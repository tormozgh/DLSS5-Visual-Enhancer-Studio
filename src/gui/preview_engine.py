"""Real-time preview engine supporting both images and video scrubbing with persistent GPU sessions."""

from __future__ import annotations

import contextlib
import ctypes
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure Windows System32 modern d3dcompiler_47.dll is loaded
if sys.platform == "win32":
    system32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    sys_d3d = os.path.join(system32, "d3dcompiler_47.dll")
    if os.path.isfile(sys_d3d):
        with contextlib.suppress(Exception):
            ctypes.WinDLL(sys_d3d)

import cv2
import numpy as np
from PIL import Image
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtGui import QImage

from ..core.gpu_selection import resolve_runtime_ai_gpu
from ..core.jobs import JobController
from ..core.runtime import (
    DLSSFrameSession,
    prepare_runtime,
    resize_fit,
    resolve_native_settings,
    resolve_output_size,
    resolve_upscaling_mode,
)
from ..neural_rendering.image.decoder import decode_image
from ..neural_rendering.image.models import ImageConversionOptions

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".flv"}


@dataclass
class PreviewParameters:
    """Live Neural Rendering parameter snapshot."""

    nr_style: str = "Default"
    upscaling_factor: float = 1.0
    nr_intensity: float = 1.0
    nr_passes: int = 1
    local_tone_strength: float = 1.0
    local_structure_strength: float = 1.0
    skin_structure_strength: float = -1.0
    nr_color_strength: float = 1.0
    tone_preservation: float = 0.0
    face_skin_protection: float = 0.0
    grain_preservation: float = 0.0
    mask_feather: int = 0
    nr_mask: Any = None
    automatic_mask: bool = False
    ai_gpu_uuid: str = "auto"
    nr_gpu_mode: bool = True
    shimmer_suppression: float = 0.0
    codec: str = "H.264 (NVIDIA NVENC)"
    container: str = "MP4"
    quality: str = "Auto (Default)"
    hdr_mode: bool = False

    def to_conversion_options(self) -> ImageConversionOptions:
        return ImageConversionOptions(
            nr_style=self.nr_style,
            upscaling_factor=self.upscaling_factor,
            nr_intensity=self.nr_intensity,
            nr_passes=self.nr_passes,
            local_tone_strength=self.local_tone_strength,
            local_structure_strength=self.local_structure_strength,
            skin_structure_strength=self.skin_structure_strength,
            nr_color_strength=self.nr_color_strength,
            tone_preservation=self.tone_preservation,
            face_skin_protection=self.face_skin_protection,
            grain_preservation=self.grain_preservation,
            mask_feather=self.mask_feather,
            nr_mask=self.nr_mask,
            automatic_mask=self.automatic_mask,
            ai_gpu_uuid=self.ai_gpu_uuid,
            nr_gpu_mode=self.nr_gpu_mode,
        )


def _numpy_to_qimage(arr: np.ndarray) -> QImage:
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError("Expected RGBA array (H, W, 4)")
    h, w = arr.shape[:2]
    contiguous = np.ascontiguousarray(arr, dtype=np.uint8)
    image = QImage(
        contiguous.data,
        w,
        h,
        w * 4,
        QImage.Format.Format_RGBA8888,
    )
    return image.copy()


class PreviewWorker(QObject):
    """Background worker executing persistent DLSS 5 sessions, video frame extraction, and previews."""

    sourceLoaded = pyqtSignal(QImage, int, int)
    videoLoaded = pyqtSignal(int, float, float)  # total_frames, fps, duration_sec
    previewReady = pyqtSignal(QImage, float, str)
    errorOccurred = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._source_path: Path | None = None
        self._is_video: bool = False
        self._video_cap: cv2.VideoCapture | None = None
        self._video_total_frames: int = 0
        self._video_fps: float = 30.0
        self._video_current_frame: int = 0

        self._source_rgba: np.ndarray | None = None
        self._source_alpha: np.ndarray | None = None
        self._params: PreviewParameters = PreviewParameters()

        self._pending_params: PreviewParameters | None = None
        self._is_rendering: bool = False
        self._lock = threading.Lock()
        self._active_session: DLSSFrameSession | None = None
        self._frame_index: int = 0
        self._use_native_bridge: bool = True
        self._native_failed_logged: bool = False

    def load_source(self, path_str: str) -> None:
        """Load image or video source from disk."""
        path = Path(path_str).resolve()
        if not path.is_file():
            self.errorOccurred.emit(f"File not found: {path}")
            return

        # Close any previous video capture
        if self._video_cap is not None:
            self._video_cap.release()
            self._video_cap = None

        self._source_path = path
        suffix = path.suffix.lower()
        self._is_video = suffix in VIDEO_EXTENSIONS

        try:
            if self._is_video:
                self._load_video(path)
            else:
                self._load_image(path)

            self._close_session()
            self.request_render(self._params)
        except Exception as exc:
            self.errorOccurred.emit(f"Failed to load media: {exc}")

    def _load_image(self, path: Path) -> None:
        decoded = decode_image(path)
        self._source_rgba = decoded.rgba
        self._source_alpha = decoded.alpha
        h, w = decoded.rgba.shape[:2]
        qimg_before = _numpy_to_qimage(decoded.rgba)
        self.sourceLoaded.emit(qimg_before, w, h)

    def _load_video(self, path: Path) -> None:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {path.name}")

        self._video_cap = cap
        self._video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
        self._video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        duration_sec = self._video_total_frames / max(1.0, self._video_fps)

        self.videoLoaded.emit(self._video_total_frames, self._video_fps, duration_sec)
        self.seek_video_frame(0)

    def seek_video_frame(self, frame_index: int) -> None:
        """Seek video to a specific frame number and refresh preview."""
        if not self._is_video or self._video_cap is None:
            return

        frame_index = max(0, min(self._video_total_frames - 1, frame_index))
        is_consecutive = (frame_index == self._video_current_frame + 1)
        self._video_current_frame = frame_index
        self._is_consecutive_playback = is_consecutive
        self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, bgr = self._video_cap.read()
        if not ret or bgr is None:
            return

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        rgba = np.empty((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = 255

        self._source_rgba = rgba
        self._source_alpha = None

        qimg_before = _numpy_to_qimage(rgba)
        self.sourceLoaded.emit(qimg_before, w, h)
        self.request_render(self._params)

    def request_render(self, params: PreviewParameters) -> None:
        with self._lock:
            self._pending_params = params
            if self._is_rendering:
                return
            self._is_rendering = True

        self._run_render_loop()

    def _close_session(self) -> None:
        self._frame_index = 0
        if self._active_session is not None:
            with contextlib.suppress(Exception):
                self._active_session.close()
            self._active_session = None

    def _run_render_loop(self) -> None:
        while True:
            with self._lock:
                params = self._pending_params
                self._pending_params = None
                if params is None:
                    self._is_rendering = False
                    break

            if self._source_rgba is None:
                continue

            self._params = params
            started = time.perf_counter()

            try:
                # 1. Primary path: Official Native D3D12/NGX DLSS 5 Engine on GPU
                if self._use_native_bridge:
                    try:
                        result_rgba, details = self._render_dlss_native(params)
                        elapsed_ms = (time.perf_counter() - started) * 1000.0
                        qimg = _numpy_to_qimage(result_rgba)
                        self.previewReady.emit(qimg, elapsed_ms, details)
                        continue
                    except Exception as bridge_exc:
                        self._close_session()
                        import traceback
                        print(f"DLSS 5 Native execution error: {bridge_exc}")
                        traceback.print_exc()

                # 2. Fallback only if hardware bridge is unavailable
                result_rgba, details = self._render_simulation(params)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                qimg = _numpy_to_qimage(result_rgba)
                self.previewReady.emit(qimg, elapsed_ms, f"{details} (Preview)")

            except Exception as exc:
                self.errorOccurred.emit(f"Preview error: {exc}")

    def _render_dlss_native(self, params: PreviewParameters) -> tuple[np.ndarray, str]:
        assert self._source_rgba is not None
        source_h, source_w = self._source_rgba.shape[:2]

        options = params.to_conversion_options()
        factor, mode = resolve_upscaling_mode(options.upscaling_factor)
        out_w, out_h = resolve_output_size(source_w, source_h, options.upscaling_factor)
        native = resolve_native_settings(options)

        prepared = prepare_runtime()
        gpu = resolve_runtime_ai_gpu(prepared.gpus, prepared.runtime_bundle, options.ai_gpu_uuid)

        session = self._active_session
        needs_new_session = (
            session is None
            or session.closed
            or session.output_width != out_w
            or session.output_height != out_h
            or session.factor != factor
            or int(session.native_settings.get("nr_passes", 1)) != int(params.nr_passes)
            or int(session.native_settings.get("style", 0)) != int(native.get("style", 0))
        )

        if needs_new_session:
            self._close_session()
            controller = JobController()
            session = DLSSFrameSession(
                input_width=source_w,
                input_height=source_h,
                output_width=out_w,
                output_height=out_h,
                frame_count=None,
                warmup_frames=0,
                factor=factor,
                mode=mode,
                native_settings=native,
                composition_mask=options.nr_mask,
                gpu=gpu,
                runtime_bundle=prepared.runtime_bundle,
                controller=controller,
            )
            self._active_session = session
            self._frame_index = 0
        else:
            session.native_settings.update(native)
            session._host_bridge_settings.update(native)

        render_rgba = resize_fit(self._source_rgba, session.render_width, session.render_height)
        idx = self._frame_index
        self._frame_index += 1
        is_consec = getattr(self, "_is_consecutive_playback", False)
        reset_history = not is_consec
        processed, _pts = session.process(index=idx, rgba=render_rgba, reset=reset_history, pts=idx)

        if self._source_alpha is None:
            processed[..., 3] = 255
        elif self._source_alpha.shape == (out_h, out_w):
            processed[..., 3] = self._source_alpha
        else:
            processed[..., 3] = cv2.resize(self._source_alpha, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

        gpu_name = str(gpu.get("display_name", "RTX GPU"))
        details = f"DLSS 5 Active | {out_w}×{out_h} | Style: {params.nr_style} | Passes: {params.nr_passes} | GPU: {gpu_name}"
        return processed, details

    def _render_simulation(self, params: PreviewParameters) -> tuple[np.ndarray, str]:
        """Clean edge-preserving neural enhancement simulation with zero underflow/overflow artifacts."""
        assert self._source_rgba is not None
        source_h, source_w = self._source_rgba.shape[:2]
        factor = float(params.upscaling_factor)
        out_w = max(64, int(round(source_w * factor)))
        out_h = max(64, int(round(source_h * factor)))

        if factor != 1.0:
            resized = cv2.resize(self._source_rgba, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            resized = self._source_rgba.copy()

        # Work entirely in normalized [0.0, 1.0] float32 to prevent any clipping/wrapping artifacts
        rgb_norm = np.clip(resized[..., :3].astype(np.float32) / 255.0, 0.0, 1.0)

        intensity = float(params.nr_intensity)
        structure = float(params.local_structure_strength)
        passes = max(1, min(4, int(params.nr_passes)))
        skin_strength = float(params.skin_structure_strength)
        tone = float(params.local_tone_strength)
        color_strength = float(params.nr_color_strength)
        tone_preservation = float(params.tone_preservation)

        # 1. Edge-preserving bilateral decomposition for clean, high-fidelity neural detail
        base = cv2.bilateralFilter(rgb_norm, d=5, sigmaColor=0.12, sigmaSpace=5.0)
        detail = rgb_norm - base

        # Gentle, progressive neural detail gain
        gain = 1.0 + (intensity - 1.0) * 0.35 + (structure - 1.0) * 0.25 + (passes - 1) * 0.12
        enhanced = base + detail * max(0.0, gain)
        enhanced = np.clip(enhanced, 0.0, 1.0)

        # 2. Skin structure smoothing (smooth flat tones while strictly preserving edges)
        if skin_strength != 0.0:
            smooth_base = cv2.bilateralFilter(enhanced, d=7, sigmaColor=0.10, sigmaSpace=7.0)
            blend_skin = min(0.5, abs(skin_strength) * 0.25)
            if skin_strength < 0:
                enhanced = enhanced * (1.0 - blend_skin) + smooth_base * blend_skin

        # 3. Local Tone Curve (smooth gamma curvature without clipping highlights or shadows)
        if tone != 1.0:
            gamma = 1.0 / max(0.1, (1.0 + (tone - 1.0) * 0.35))
            enhanced = np.power(np.clip(enhanced, 1e-5, 1.0), gamma)

        # 4. Color Strength
        if color_strength < 1.0:
            # Rec.709 relative luminance
            luminance = 0.2126 * enhanced[..., 0] + 0.7152 * enhanced[..., 1] + 0.0722 * enhanced[..., 2]
            lum_3d = np.repeat(luminance[..., np.newaxis], 3, axis=2)
            enhanced = enhanced * color_strength + lum_3d * (1.0 - color_strength)

        # 5. Tone Preservation
        if tone_preservation > 0.0:
            enhanced = enhanced * (1.0 - tone_preservation) + rgb_norm * tone_preservation

        # Final safe cast to uint8 with guaranteed bounds
        output = np.empty_like(resized)
        output[..., :3] = (np.clip(enhanced, 0.0, 1.0) * 255.0).astype(np.uint8)
        output[..., 3] = resized[..., 3]

        details = f"{out_w}×{out_h} | Style: {params.nr_style} | Intensity: {intensity:.2f} | Passes: {passes}"
        return output, details

    def stop(self) -> None:
        if self._video_cap is not None:
            self._video_cap.release()
            self._video_cap = None
        self._close_session()


class PreviewEngine(QObject):
    """Public interface managing preview worker thread, video seeking, and render signals."""

    sourceLoaded = pyqtSignal(QImage, int, int)
    videoLoaded = pyqtSignal(int, float, float)
    previewReady = pyqtSignal(QImage, float, str)
    errorOccurred = pyqtSignal(str)

    _sig_load_source = pyqtSignal(str)
    _sig_seek_video = pyqtSignal(int)
    _sig_request_render = pyqtSignal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread = QThread()
        self._worker = PreviewWorker()
        self._worker.moveToThread(self._thread)

        self._sig_load_source.connect(self._worker.load_source)
        self._sig_seek_video.connect(self._worker.seek_video_frame)
        self._sig_request_render.connect(self._worker.request_render)

        self._worker.sourceLoaded.connect(self.sourceLoaded)
        self._worker.videoLoaded.connect(self.videoLoaded)
        self._worker.previewReady.connect(self.previewReady)
        self._worker.errorOccurred.connect(self.errorOccurred)

        self._thread.start()

    def load_source(self, path: str) -> None:
        self._sig_load_source.emit(path)

    def seek_video(self, frame_index: int) -> None:
        self._sig_seek_video.emit(frame_index)

    def update_params(self, params: PreviewParameters) -> None:
        self._sig_request_render.emit(params)

    def shutdown(self) -> None:
        self._worker.stop()
        self._thread.quit()
        self._thread.wait(2000)
