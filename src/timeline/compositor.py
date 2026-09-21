"""Timeline Multi-Track Compositor with DLSS 5 Adjustment Layer Support."""

from __future__ import annotations

import cv2
import numpy as np

from .frame_cache import VideoFrameCache
from .models import TimelineProject
from .processor import DLSS5TimelineProcessor


class TimelineCompositor:
    """Composites video tracks from bottom to top and executes DLSS 5 Adjustment Layers."""

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
    ) -> np.ndarray:
        """Render composite timeline at frame_idx.

        Args:
            frame_idx: Current timeline frame index.
            target_size: (width, height) of final output. Defaults to project resolution.
            is_export: True for production export, False for preview scrubbing.
            split_ratio: If set (0.0 to 1.0), renders a split-screen before/after comparison.

        Returns:
            BGR uint8 numpy array of the composited timeline.
        """
        width, height = target_size if target_size else (self.project.width, self.project.height)

        # Base background canvas (pitch black)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        raw_pre_adjustment_canvas = canvas.copy()

        # Query all active video clips at this frame sorted from bottom (V1) to top (V_n)
        active_clips = self.project.get_active_video_clips_at(frame_idx)
        if not active_clips:
            return canvas

        has_adjustment_layer = any(c.clip_type == "adjustment_layer" for _, c in active_clips)

        for track, clip in active_clips:
            if clip.clip_type == "media":
                if not clip.media_path:
                    continue

                source_frame_idx = clip.source_in + (frame_idx - clip.timeline_in)
                src_frame = self.frame_cache.get_frame(clip.media_path, source_frame_idx)
                if src_frame is None or src_frame.size == 0:
                    continue

                # Scale / fit frame to canvas dimensions
                h_src, w_src = src_frame.shape[:2]
                if w_src != width or h_src != height:
                    fitted = cv2.resize(src_frame, (width, height), interpolation=cv2.INTER_LINEAR)
                else:
                    fitted = src_frame

                # Apply clip-level DLSS if explicitly enabled on this individual clip
                if clip.dlss_params.enabled:
                    fitted = self.processor.process_frame(fitted, clip.dlss_params, (width, height), is_export)

                # Alpha blend onto canvas
                if clip.opacity >= 0.999:
                    canvas = fitted
                else:
                    alpha = max(0.0, min(1.0, clip.opacity))
                    canvas = cv2.addWeighted(fitted, alpha, canvas, 1.0 - alpha, 0)

            elif clip.clip_type == "adjustment_layer":
                # Save snapshot of pre-adjustment canvas for split-screen comparison
                raw_pre_adjustment_canvas = canvas.copy()

                # Adjustment Layer acts upon the entire composite beneath it!
                if clip.dlss_params.enabled and canvas is not None and canvas.size > 0:
                    enhanced = self.processor.process_frame(canvas, clip.dlss_params, (width, height), is_export)
                    if clip.opacity >= 0.999:
                        canvas = enhanced
                    else:
                        alpha = max(0.0, min(1.0, clip.opacity))
                        canvas = cv2.addWeighted(enhanced, alpha, canvas, 1.0 - alpha, 0)

        # Handle split-screen comparison mode if requested
        if split_ratio is not None and 0.0 < split_ratio < 1.0 and has_adjustment_layer:
            split_x = int(width * split_ratio)
            comparison = canvas.copy()
            comparison[:, :split_x] = raw_pre_adjustment_canvas[:, :split_x]
            # Draw vertical divider line
            cv2.line(comparison, (split_x, 0), (split_x, height), (0, 255, 255), 2)
            return comparison

        return canvas
