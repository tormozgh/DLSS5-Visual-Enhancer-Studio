from __future__ import annotations

import traceback
from pathlib import Path

try:
    import gradio as gr
except ImportError:
    class _DummyProgress:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs): pass
    class _DummyGradio:
        Progress = _DummyProgress
        Error = RuntimeError
    gr = _DummyGradio()  # type: ignore

from ...core.jobs import Cancelled
from ...core.ffmpeg.preview import (
    is_browser_playable, make_browser_preview, normalize_preview_encoding,
    resolve_final_preview, resolve_preview_codec, wants_compat_preview,
)
from ...settings.models import coerce_hdr_mode, parse_automatic_mask
from ...settings.storage import current_preview_encoding, processing_gpu_settings
from .models import ConversionOptions
from .processor import convert_video

PREVIEW_SECONDS = 3.0

def _process_video(
    input_path: str | None,
    nr_style: str,
    nr_intensity: float,
    nr_passes: float,
    local_tone_strength: float,
    local_structure_strength: float,
    skin_structure_strength: float,
    upscaling_factor: float,
    automatic_mask: str,
    nr_color_strength: float,
    tone_preservation: float,
    face_skin_protection: float,
    grain_preservation: float,
    mask_feather: float,
    shimmer_suppression: float,
    nr_mask: object | None,
    nr_gpu_mode: bool,
    codec: str,
    container: str,
    quality: str,
    hdr_mode: bool,
    progress,
    preview_seconds: float | None,
    preview_frames: int | None,
    *,
    output_dir=None,
    controller=None,
    ephemeral_preview: bool = False,
) -> tuple[str | None, str]:
    if not input_path:
        raise gr.Error("Choose a video first.")
    is_preview = preview_seconds is not None or preview_frames is not None
    ai_gpu_uuid, video_gpu_uuid = processing_gpu_settings()
    try:
        preview_mode = current_preview_encoding()
    except Exception:
        preview_mode = "Auto"
    preview_mode = normalize_preview_encoding(preview_mode)
    if is_preview:
        effective_codec, effective_container = resolve_preview_codec(
            codec, container, preview_mode
        )
        compat_preview = wants_compat_preview(codec, container, preview_mode)
    else:
        effective_codec, effective_container = codec, container
        compat_preview = False
    # HDR Mode only for allowed codecs. Compat (forced H.264) previews stay SDR
    # 8-bit; user-encoded previews preserve the HDR choice.
    effective_hdr = coerce_hdr_mode(effective_codec, hdr_mode) and (
        not is_preview or not compat_preview
    )
    options = ConversionOptions(
        ai_gpu_uuid=ai_gpu_uuid,
        video_gpu_uuid=video_gpu_uuid,
        nr_style=nr_style,
        nr_intensity=nr_intensity,
        nr_passes=int(nr_passes),
        local_tone_strength=local_tone_strength,
        local_structure_strength=local_structure_strength,
        skin_structure_strength=skin_structure_strength,
        automatic_mask=parse_automatic_mask(automatic_mask),
        nr_color_strength=float(nr_color_strength),
        tone_preservation=float(tone_preservation),
        face_skin_protection=float(face_skin_protection),
        grain_preservation=float(grain_preservation),
        shimmer_suppression=float(shimmer_suppression),
        mask_feather=int(mask_feather),
        nr_mask=nr_mask,
        nr_gpu_mode=nr_gpu_mode,
        upscaling_factor=upscaling_factor,
        codec=effective_codec,
        container=effective_container,
        quality=quality,
        preserve_hdr=effective_hdr,
        preview_seconds=preview_seconds,
        preview_frames=preview_frames,
        preview_compat=compat_preview,
    )

    def report(value: float, message: str) -> None:
        progress(value, desc=message)

    try:
        result = convert_video(
            input_path, options, progress=report, output_dir=output_dir,
            controller=controller,
        )
    except Cancelled:
        return None, "Preview cancelled."
    except Exception as exc:
        traceback.print_exc()
        return None, f"Failed: {exc}"

    def finish(media_path: str | None, status: str) -> tuple[str | None, str]:
        if ephemeral_preview:
            # report_path now points at the shared session log (or a kept
            # .err file): only remove legacy per-job report JSONs.
            if Path(result.report_path).name.endswith(".report.json"):
                Path(result.report_path).unlink(missing_ok=True)
            if media_path and Path(media_path).resolve() != Path(result.output_path).resolve():
                Path(result.output_path).unlink(missing_ok=True)
        return media_path, status

    source_name = Path(input_path).name
    if is_preview:
        # Truncated previews normally show the encoded file directly. In Auto
        # mode the pre-check may have selected the user encoding but the actual
        # file can still probe as unplayable (rare); then derive one H.264 file.
        output_preview = result.output_path
        derived_note = ""
        if preview_mode == "Auto" and not compat_preview:
            try:
                playable = is_browser_playable(result.output_path)
            except Exception:
                playable = False
            if not playable:
                try:
                    output_preview = make_browser_preview(
                        result.output_path, dest_dir=output_dir, controller=controller,
                    )
                    derived_note = " (browser preview transcoded to H.264)"
                except Exception:
                    output_preview = result.output_path
        if preview_frames is not None:
            return finish(output_preview, (
                f"One-frame preview complete for {source_name} on {result.gpu} "
                f"in {result.elapsed_seconds:.1f}s. "
                f"Neural dimensions {result.render_width}×{result.render_height}; "
                f"{result.resize_method}, {result.memory_path}. Feature 18 confirmed."
                f"{derived_note}"
            ))
        return finish(output_preview, (
            f"Preview complete for {source_name}: {result.frames} frames from the first "
            f"{PREVIEW_SECONDS:g} seconds processed "
            f"on {result.gpu} in {result.elapsed_seconds:.1f}s. Neural dimensions "
            f"{result.render_width}×{result.render_height}; {result.resize_method}, "
            f"{result.memory_path}. All frames returned feature-18 success."
            f"{derived_note}"
        ))
    output_preview, used_derivative = resolve_final_preview(
        result.output_path, preview_mode, bounded_proxy=True
    )
    status = (
        f"Complete: {result.frames} frames processed on {result.gpu} in {result.elapsed_seconds:.1f}s. "
        f"All {result.nr_count_evidence} frames returned feature-18 success. "
        f"Neural dimensions {result.render_width}×{result.render_height}; "
        f"{result.resize_method}, {result.memory_path}."
    )
    if used_derivative:
        status += " A short H.264 browser proxy was created; the complete original output is unchanged."
    elif output_preview is None:
        status += f" {effective_container} output was created successfully, but browser preview is unavailable."
    return finish(output_preview, status)

def normalize_video_paths(paths: list[str] | str | None) -> list[str]:
    if not paths:
        return []
    return [paths] if isinstance(paths, str) else list(paths)


def first_video_path(paths: list[str] | str | None) -> str | None:
    normalized = normalize_video_paths(paths)
    return normalized[0] if normalized else None


def update_video_preview_mode(paths: list[str] | str | None):
    normalized = normalize_video_paths(paths)
    available = bool(normalized)
    single = len(normalized) == 1
    input_value = normalized[0] if available else None
    input_label = "Input video preview" if len(normalized) <= 1 else f"Input video preview (first of {len(normalized)})"
    return (
        gr.update(value=input_value, visible=True if available else "hidden", label=input_label),
        gr.update(value=None, visible=True),
        gr.update(visible=single),
        gr.update(visible=single),
    )

def preview_video(
    input_path: list[str] | str | None,
    nr_style: str,
    nr_intensity: float,
    nr_passes: float,
    local_tone_strength: float,
    local_structure_strength: float,
    skin_structure_strength: float,
    upscaling_factor: float,
    automatic_mask: str,
    nr_color_strength: float,
    tone_preservation: float,
    face_skin_protection: float,
    grain_preservation: float,
    mask_feather: float,
    shimmer_suppression: float,
    nr_mask: object | None,
    nr_gpu_mode: bool,
    codec: str,
    container: str,
    quality: str,
    hdr_mode: bool = False,
    progress=gr.Progress(track_tqdm=False),
):
    selected = first_video_path(input_path)
    return _process_video(
        selected, nr_style, nr_intensity, nr_passes, local_tone_strength, local_structure_strength,
        skin_structure_strength, upscaling_factor, automatic_mask,
        nr_color_strength, tone_preservation, face_skin_protection, grain_preservation,
        mask_feather, shimmer_suppression, nr_mask, nr_gpu_mode,
        codec, container, quality, hdr_mode,
        progress, PREVIEW_SECONDS, None
    )


def preview_one_frame(
    input_path: list[str] | str | None,
    nr_style: str,
    nr_intensity: float,
    nr_passes: float,
    local_tone_strength: float,
    local_structure_strength: float,
    skin_structure_strength: float,
    upscaling_factor: float,
    automatic_mask: str,
    nr_color_strength: float,
    tone_preservation: float,
    face_skin_protection: float,
    grain_preservation: float,
    mask_feather: float,
    shimmer_suppression: float,
    nr_mask: object | None,
    nr_gpu_mode: bool,
    codec: str,
    container: str,
    quality: str,
    hdr_mode: bool = False,
    progress=gr.Progress(track_tqdm=False),
    *, output_dir=None, controller=None, ephemeral_preview=False,
):
    selected = first_video_path(input_path)
    return _process_video(
        selected, nr_style, nr_intensity, nr_passes, local_tone_strength, local_structure_strength,
        skin_structure_strength, upscaling_factor, automatic_mask,
        nr_color_strength, tone_preservation, face_skin_protection, grain_preservation,
        mask_feather, shimmer_suppression, nr_mask, nr_gpu_mode,
        codec, container, quality, hdr_mode,
        progress, None, 1, output_dir=output_dir, controller=controller,
        ephemeral_preview=ephemeral_preview,
    )
