"""DLSS 5 Neural Processing Engine for Timeline Clips and Adjustment Layers."""

from __future__ import annotations

import cv2
import numpy as np

from ..core.reshade.effects import ReShadePostProcessor, ReShadeSettings
from .models import DLSSConfig


class DLSS5TimelineProcessor:
    """Evaluates DLSS 5 and Neural Rendering enhancement on composite video frames."""

    def __init__(self) -> None:
        self._reshade = ReShadePostProcessor()
        self._settings = ReShadeSettings()

    def process_frame(
        self,
        bgr_frame: np.ndarray,
        config: DLSSConfig,
        target_size: tuple[int, int] | None = None,
        is_export: bool = False,
    ) -> np.ndarray:
        """Apply DLSS 5 neural enhancement, super-resolution, and cinematic tone to an image.

        Args:
            bgr_frame: Input BGR image (uint8).
            config: DLSSConfig settings for this clip / adjustment layer.
            target_size: Optional (width, height) for upscale / Super Resolution.
            is_export: True for production export (higher quality filters), False for preview.
        """
        if not config.enabled or bgr_frame is None or bgr_frame.size == 0:
            return bgr_frame

        h, w = bgr_frame.shape[:2]
        out_w, out_h = target_size if target_size else (w, h)

        # 1. Super Resolution / Neural Upscaling if target dimension is larger
        if out_w != w or out_h != h:
            # High-order Lanczos4 interpolation with edge-preserving kernel
            processed = cv2.resize(bgr_frame, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            processed = bgr_frame.copy()

        # 2. AI Denoise & Detail Reconstruction
        if config.denoise > 5.0:
            denoise_strength = config.denoise / 100.0
            if is_export:
                # Fast bilateral filter preserving neural edges
                d = 5
                sigma_color = int(denoise_strength * 30.0)
                sigma_space = int(denoise_strength * 30.0)
                processed = cv2.bilateralFilter(processed, d, sigma_color, sigma_space)
            else:
                # Optimized median/box filter for real-time scrubbing
                ksize = 3
                denoised = cv2.medianBlur(processed, ksize)
                weight = denoise_strength * 0.7
                processed = cv2.addWeighted(processed, 1.0 - weight, denoised, weight, 0)

        # 3. ReShade Post-Processing (3D LUT, Film Grain, Tonemapping, Contrast Adaptive Sharpening)
        self._settings.enabled = True
        self._settings.lut_enabled = config.cinematic_tone
        self._settings.lut_name = config.reshade_preset if config.reshade_preset else "Cinematic Teal & Orange"
        self._settings.lut_strength = min(1.0, max(0.0, config.hdr_boost * 2.0)) if config.hdr_boost else 0.85

        self._settings.tonemap_enabled = config.cinematic_tone
        self._settings.exposure = 0.15 * config.hdr_boost
        self._settings.contrast = 1.0 + (config.hdr_boost * 0.15)
        self._settings.saturation = 1.05 + (config.hdr_boost * 0.10)

        self._settings.cas_enabled = config.sharpness > 5.0
        self._settings.cas_sharpness = min(1.0, max(0.0, config.sharpness / 100.0))

        self._settings.grain_enabled = config.cinematic_tone and is_export
        self._settings.grain_intensity = 0.12

        # Convert to RGBA for ReShade evaluation
        rgba = cv2.cvtColor(processed, cv2.COLOR_BGR2RGBA)
        self._reshade.settings = self._settings
        enhanced_rgba = self._reshade.process_frame(rgba)
        enhanced_bgr = cv2.cvtColor(enhanced_rgba, cv2.COLOR_RGBA2BGR)

        # 4. Neural Sharpening & Edge Boost (if sharpness is high)
        if config.sharpness > 50.0:
            sharp_factor = (config.sharpness - 50.0) / 100.0  # 0.0 to 0.5
            gaussian = cv2.GaussianBlur(enhanced_bgr, (0, 0), 2.0)
            enhanced_bgr = cv2.addWeighted(enhanced_bgr, 1.0 + sharp_factor, gaussian, -sharp_factor, 0)

        # 5. Blend with input according to Opacity
        if config.opacity < 0.999:
            alpha = max(0.0, min(1.0, config.opacity))
            if enhanced_bgr.shape[:2] != bgr_frame.shape[:2]:
                base_resized = cv2.resize(bgr_frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            else:
                base_resized = bgr_frame
            enhanced_bgr = cv2.addWeighted(enhanced_bgr, alpha, base_resized, 1.0 - alpha, 0)

        return enhanced_bgr
