"""Timeline Multi-Track Compositor with DLSS 5 and ReShade FX Adjustment Layer Support."""

from __future__ import annotations

import cv2
import numpy as np

from .frame_cache import VideoFrameCache
from .models import TimelineProject
from .processor import DLSS5TimelineProcessor


class TimelineCompositor:
    """Composites video tracks from bottom to top and executes DLSS 5 / ReShade FX Adjustment Layers."""

    def __init__(self, project: TimelineProject, frame_cache: VideoFrameCache | None = None) -> None:
        self.project = project
        self.frame_cache = frame_cache or VideoFrameCache()
        self.processor = DLSS5TimelineProcessor()

    def render_frame(
        self,
        frame_idx: int,
        target_size: tuple[int, int] | None = None,
        is_export: bool = False,
        split_ratio: float | None = None,
        return_raw: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Render composite timeline at frame_idx.

        Args:
            frame_idx: Current timeline frame index.
            target_size: (width, height) of final output. Defaults to project resolution.
            is_export: True for production export, False for preview scrubbing.
            split_ratio: Legacy split-screen ratio (if used directly).
            return_raw: If True, returns (enhanced_canvas, raw_canvas) for interactive SplitCanvas.

        Returns:
            BGR uint8 numpy array, or (enhanced_canvas, raw_canvas) if return_raw is True.
        """
        width, height = target_size if target_size else (self.project.width, self.project.height)

        # Base background canvas (pitch black)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Query all active video clips at this frame sorted from bottom (V1) to top (V_n)
        active_clips = self.project.get_active_video_clips_at(frame_idx)
        if not active_clips:
            empty_raw = canvas.copy()
            return (canvas, empty_raw) if return_raw else canvas

        # Separate media clips from FX/Adjustment layers
        media_clips = [(t, c) for t, c in active_clips if c.clip_type == "media"]
        fx_clips = [(t, c) for t, c in active_clips if c.clip_type in ("adjustment_layer", "fx_layer")]

        # 1. Render all active media footage layers onto base canvas
        for track, clip in media_clips:
            if not clip.media_path:
                continue

            # FPS Conforming: map timeline frame to source video frame
            source_fps = getattr(clip, "asset_fps", self.project.fps) or self.project.fps
            seq_fps = self.project.fps if self.project.fps > 0 else 30.0
            time_sec = (frame_idx - clip.timeline_in) / seq_fps
            source_frame_idx = clip.source_in + int(round(time_sec * source_fps))

            src_frame = self.frame_cache.get_frame(clip.media_path, source_frame_idx)
            if src_frame is None or src_frame.size == 0:
                continue

            h_src, w_src = src_frame.shape[:2]
            scale_mode = getattr(clip, "scale_mode", "fit")

            # Scale and conform footage to sequence canvas
            if w_src == width and h_src == height:
                fitted = src_frame
            elif scale_mode == "fill":
                scale = max(width / w_src, height / h_src)
                nw, nh = int(round(w_src * scale)), int(round(h_src * scale))
                resized = cv2.resize(src_frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
                ox = max(0, (nw - width) // 2)
                oy = max(0, (nh - height) // 2)
                fitted = resized[oy : oy + height, ox : ox + width]
            elif scale_mode == "stretch":
                fitted = cv2.resize(src_frame, (width, height), interpolation=cv2.INTER_LINEAR)
            else:
                # Default: "fit" preserving aspect ratio with letterbox/pillarbox
                scale = min(width / w_src, height / h_src)
                nw, nh = int(round(w_src * scale)), int(round(h_src * scale))
                resized = cv2.resize(src_frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
                fitted = np.zeros((height, width, 3), dtype=np.uint8)
                ox = (width - nw) // 2
                oy = (height - nh) // 2
                fitted[oy : oy + nh, ox : ox + nw] = resized

            # Alpha blend footage onto canvas
            if clip.opacity >= 0.999:
                canvas = fitted
            else:
                alpha = max(0.0, min(1.0, clip.opacity))
                canvas = cv2.addWeighted(fitted, alpha, canvas, 1.0 - alpha, 0)

        # Snapshot the raw composite footage BEFORE applying any FX / Adjustment layer!
        # This guarantees that the ORIGINAL view in Split Preview is NEVER black.
        raw_pre_adjustment_canvas = canvas.copy()

        # 2. Render all active FX / Adjustment layers on top of the composite
        has_fx_layer = len(fx_clips) > 0
        for track, clip in fx_clips:
            fx_type = getattr(clip, "fx_type", "dlss5")
            enhanced = None

            if fx_type == "reshade" and hasattr(clip, "reshade_params") and clip.reshade_params.enabled:
                enhanced = self.processor.process_reshade_frame(
                    canvas, clip.reshade_params, (width, height), is_export
                )
            elif (fx_type == "dlss5" or clip.clip_type == "adjustment_layer") and clip.dlss_params.enabled:
                enhanced = self.processor.process_frame(
                    canvas, clip.dlss_params, (width, height), is_export, frame_idx=frame_idx
                )

            if enhanced is not None:
                if clip.opacity >= 0.999:
                    canvas = enhanced
                else:
                    alpha = max(0.0, min(1.0, clip.opacity))
                    canvas = cv2.addWeighted(enhanced, alpha, canvas, 1.0 - alpha, 0)

        # Legacy split divider if requested
        if split_ratio is not None and 0.0 < split_ratio < 1.0 and has_fx_layer:
            split_x = int(width * split_ratio)
            comparison = canvas.copy()
            comparison[:, :split_x] = raw_pre_adjustment_canvas[:, :split_x]
            cv2.line(comparison, (split_x, 0), (split_x, height), (0, 255, 255), 2)
            return (comparison, raw_pre_adjustment_canvas) if return_raw else comparison

        if return_raw:
            return canvas, raw_pre_adjustment_canvas
        return canvas
