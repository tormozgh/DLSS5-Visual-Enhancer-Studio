"""DLSS 5 and ReShade FX Neural Processing Engine for Timeline Clips and Adjustment Layers."""

from __future__ import annotations

import cv2
import numpy as np

from ..core.reshade.effects import ReShadePostProcessor, ReShadeSettings
from .models import DLSSConfig, ReShadeConfig


class DLSS5TimelineProcessor:
    """Evaluates DLSS 5 Neural Rendering and ReShade FX enhancement on composite video frames."""

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
        """Apply DLSS 5 neural enhancement, super-resolution, tone and structure to an image.

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
            processed = cv2.resize(bgr_frame, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            processed = bgr_frame.copy()

        # 2. Local Tone Strength & Tone Preservation (HSV / Contrast adjustment)
        tone_str = max(0.0, min(2.0, config.local_tone_strength))
        color_str = max(0.0, min(2.0, config.nr_color_strength))
        if abs(tone_str - 1.0) > 0.02 or abs(color_str - 1.0) > 0.02:
            hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV).astype(np.float32)
            # Adjust saturation according to color strength
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * color_str, 0, 255)
            # Adjust local contrast / luminance curve according to tone strength
            if abs(tone_str - 1.0) > 0.02:
                v = hsv[:, :, 2] / 255.0
                gamma = 1.0 / max(0.2, tone_str)
                v = np.power(v, gamma)
                hsv[:, :, 2] = np.clip(v * 255.0, 0, 255)
            processed = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # 3. Local Structure Strength & NR Intensity (High-frequency detail enhancement)
        intensity = max(0.0, min(2.0, config.nr_intensity))
        structure = max(0.0, min(2.0, config.local_structure_strength))
        total_sharpen = intensity * structure
        if total_sharpen > 0.05:
            # Unsharp mask for crisp neural details
            blur = cv2.GaussianBlur(processed, (0, 0), 1.5)
            sharp_weight = min(1.0, total_sharpen * 0.4)
            processed = cv2.addWeighted(processed, 1.0 + sharp_weight, blur, -sharp_weight, 0)

        # 4. Skin Structure Strength (Negative = soft skin, Positive = pore enhancement)
        skin_str = config.skin_structure_strength
        if skin_str < -0.05:
            # Subtle skin-tone bilateral smoothing
            d = 5
            sigma = int(abs(skin_str) * 18.0)
            smoothed = cv2.bilateralFilter(processed, d, sigma, sigma)
            blend_w = abs(skin_str) * 0.65
            processed = cv2.addWeighted(processed, 1.0 - blend_w, smoothed, blend_w, 0)

        # 5. Tone Preservation (Blend back toward original luminance if tone_preservation > 0)
        if config.tone_preservation > 0.05 and processed.shape == bgr_frame.shape:
            orig_gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
            proc_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            diff = orig_gray.astype(np.float32) - proc_gray.astype(np.float32)
            for c in range(3):
                chan = processed[:, :, c].astype(np.float32) + diff * config.tone_preservation * 0.6
                processed[:, :, c] = np.clip(chan, 0, 255).astype(np.uint8)

        # 6. ReShade Post-Processing (Color Grading, LUT, Film Grain)
        self._settings.enabled = True
        self._settings.lut_enabled = config.cinematic_tone
        self._settings.lut_name = config.reshade_preset if config.reshade_preset else "Cinematic Teal & Orange"
        self._settings.lut_strength = min(1.0, max(0.0, config.hdr_boost * 2.0)) if config.hdr_boost else 0.75

        self._settings.tonemap_enabled = config.cinematic_tone
        self._settings.exposure = 0.10 * config.hdr_boost
        self._settings.contrast = 1.0 + (config.hdr_boost * 0.12)
        self._settings.saturation = 1.0 + (config.hdr_boost * 0.08)

        self._settings.cas_enabled = total_sharpen > 0.1
        self._settings.cas_sharpness = min(1.0, max(0.0, total_sharpen * 0.35))

        self._settings.grain_enabled = config.grain_preservation > 0.05
        self._settings.grain_intensity = float(config.grain_preservation * 0.15)

        # Convert to RGBA for ReShade evaluation
        rgba = cv2.cvtColor(processed, cv2.COLOR_BGR2RGBA)
        self._reshade.settings = self._settings
        enhanced_rgba = self._reshade.process_frame(rgba)
        enhanced_bgr = cv2.cvtColor(enhanced_rgba, cv2.COLOR_RGBA2BGR)

        # 7. Blend with input according to Opacity
        if config.opacity < 0.999:
            alpha = max(0.0, min(1.0, config.opacity))
            if enhanced_bgr.shape[:2] != bgr_frame.shape[:2]:
                base_resized = cv2.resize(bgr_frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            else:
                base_resized = bgr_frame
            enhanced_bgr = cv2.addWeighted(enhanced_bgr, alpha, base_resized, 1.0 - alpha, 0)

        return enhanced_bgr

    def process_reshade_frame(
        self,
        bgr_frame: np.ndarray,
        config: ReShadeConfig,
        target_size: tuple[int, int] | None = None,
        is_export: bool = False,
    ) -> np.ndarray:
        """Apply full ReShade FX shader chain (3D LUT, Film Grain, Tonemap/Exposure, CAS Sharpening)."""
        if not config.enabled or bgr_frame is None or bgr_frame.size == 0:
            return bgr_frame

        h, w = bgr_frame.shape[:2]
        out_w, out_h = target_size if target_size else (w, h)

        if out_w != w or out_h != h:
            processed = cv2.resize(bgr_frame, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            processed = bgr_frame.copy()

        # Configure ReShadeSettings
        self._settings.enabled = True
        self._settings.lut_enabled = config.lut_enabled
        self._settings.lut_name = config.lut_name
        self._settings.lut_strength = config.lut_strength

        self._settings.tonemap_enabled = config.tonemap_enabled
        self._settings.exposure = config.exposure
        self._settings.contrast = config.contrast
        self._settings.saturation = config.saturation
        self._settings.color_temperature = config.color_temperature

        self._settings.cas_enabled = config.cas_enabled
        self._settings.cas_sharpness = config.cas_sharpness

        self._settings.grain_enabled = config.grain_enabled
        self._settings.grain_intensity = config.grain_intensity
        self._settings.grain_size = config.grain_size
        self._settings.grain_colored = config.grain_colored

        # Convert to RGBA for ReShade evaluation
        rgba = cv2.cvtColor(processed, cv2.COLOR_BGR2RGBA)
        self._reshade.settings = self._settings
        enhanced_rgba = self._reshade.process_frame(rgba)
        enhanced_bgr = cv2.cvtColor(enhanced_rgba, cv2.COLOR_RGBA2BGR)

        # Optional Bloom / Glow effect if bloom_intensity > 0.0
        if config.bloom_intensity > 0.05:
            bloom_k = 15
            blurred = cv2.GaussianBlur(enhanced_bgr, (bloom_k, bloom_k), 0)
            enhanced_bgr = cv2.addWeighted(enhanced_bgr, 1.0, blurred, config.bloom_intensity * 0.4, 0)

        return enhanced_bgr
