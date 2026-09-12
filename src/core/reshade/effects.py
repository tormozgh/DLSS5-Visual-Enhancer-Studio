"""ReShade FX Shaders: Real-time post-processing passes.

Faithful, high-performance implementations of ReShade shaders:
- 3D LUT Color Grading (Cinematic, Bleach Bypass, Technicolor, Cyberpunk, Golden Hour)
- ReShade FilmGrain.fx (procedural analog film grain with temporal jitter and luminance weighting)
- ACES Filmic Tonemap (Stephen Hill ACES curve with exposure and color temperature controls)
- AMD FidelityFX / ReShade CAS (Contrast Adaptive Sharpening)
- Vibrance & Dynamic Saturation

References:
    https://github.com/crosire/reshade
    https://github.com/crosire/reshade-shaders
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass
class ReShadeSettings:
    """Parameters for active ReShade FX post-processing chain."""

    # Master switch
    enabled: bool = True

    # 3D LUT
    lut_enabled: bool = True
    lut_name: str = "Cinematic Teal & Orange"
    lut_strength: float = 0.85  # 0.0 to 1.0

    # Tonemapping & ACES
    tonemap_enabled: bool = True
    exposure: float = 0.0  # -2.0 to +2.0 EV
    contrast: float = 1.05  # 0.5 to 2.0
    saturation: float = 1.10  # 0.0 to 2.0
    color_temperature: float = 0.0  # -1.0 (Cool) to +1.0 (Warm)

    # ReShade Film Grain
    grain_enabled: bool = True
    grain_intensity: float = 0.18  # 0.0 to 1.0
    grain_size: float = 1.5  # 1.0 to 3.0
    grain_colored: bool = False

    # CAS (Contrast Adaptive Sharpening)
    cas_enabled: bool = True
    cas_sharpness: float = 0.40  # 0.0 to 1.0


class FastTonemapLUT:
    """Pre-computed 256-entry LUTs for sub-millisecond ACES tonemapping and curves."""

    def __init__(self) -> None:
        self._cache_key: tuple[float, float, float, float] | None = None
        self._lut_b: np.ndarray = np.arange(256, dtype=np.uint8)
        self._lut_g: np.ndarray = np.arange(256, dtype=np.uint8)
        self._lut_r: np.ndarray = np.arange(256, dtype=np.uint8)

    def get_tables(
        self, exposure: float, contrast: float, temperature: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (round(exposure, 2), round(contrast, 2), round(temperature, 2))
        if key == self._cache_key:
            return self._lut_b, self._lut_g, self._lut_r

        x = np.linspace(0.0, 1.0, 256, dtype=np.float32)

        # Exposure bias
        exp_factor = math.pow(2.0, exposure)
        x_exp = x * exp_factor

        # ACES Filmic curve (Hill approximation)
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14
        aces = (x_exp * (a * x_exp + b)) / (x_exp * (c * x_exp + d) + e)
        aces = np.clip(aces, 0.0, 1.0)

        # Contrast adjustment around midtone 0.18
        if contrast != 1.0:
            aces = np.power(aces, 1.0 / contrast)

        base_curve = (aces * 255.0).astype(np.float32)

        # Color temperature adjustment (Kelvin shift: warm boosts R and reduces B, cool inverse)
        r_mult = 1.0 + max(0.0, temperature) * 0.15 - max(0.0, -temperature) * 0.10
        g_mult = 1.0 + max(0.0, temperature) * 0.05
        b_mult = 1.0 + max(0.0, -temperature) * 0.20 - max(0.0, temperature) * 0.15

        self._lut_r = np.clip(base_curve * r_mult, 0, 255).astype(np.uint8)
        self._lut_g = np.clip(base_curve * g_mult, 0, 255).astype(np.uint8)
        self._lut_b = np.clip(base_curve * b_mult, 0, 255).astype(np.uint8)
        self._cache_key = key
        return self._lut_b, self._lut_g, self._lut_r


class ReShadeLutManager:
    """Manages 3D cinematic color profiles."""

    def __init__(self) -> None:
        self._profiles: dict[str, dict[str, Any]] = {
            "Cinematic Teal & Orange": {
                "shadow_color": np.array([30, -5, -25], dtype=np.float32),  # BGR bias in shadows
                "highlight_color": np.array([-15, 10, 35], dtype=np.float32),  # Warm amber highlights
                "gamma": 1.08,
            },
            "Warm Golden Hour": {
                "shadow_color": np.array([-5, 15, 25], dtype=np.float32),
                "highlight_color": np.array([-20, 20, 50], dtype=np.float32),
                "gamma": 1.05,
            },
            "Bleach Bypass": {
                "shadow_color": np.array([5, 5, 5], dtype=np.float32),
                "highlight_color": np.array([0, 0, 0], dtype=np.float32),
                "saturation_mult": 0.45,
                "contrast_mult": 1.35,
                "gamma": 0.95,
            },
            "Technicolor 3-Strip": {
                "shadow_color": np.array([-10, 0, 15], dtype=np.float32),
                "highlight_color": np.array([10, 20, 30], dtype=np.float32),
                "saturation_mult": 1.30,
                "gamma": 1.02,
            },
            "Cyberpunk Neon": {
                "shadow_color": np.array([45, -20, 10], dtype=np.float32),  # Deep violet shadows
                "highlight_color": np.array([30, 40, -10], dtype=np.float32),  # Electric cyan/green highlights
                "gamma": 1.12,
            },
            "Neutral Broadcast": {
                "shadow_color": np.array([0, 0, 0], dtype=np.float32),
                "highlight_color": np.array([0, 0, 0], dtype=np.float32),
                "gamma": 1.0,
            },
        }

    @property
    def available_luts(self) -> list[str]:
        return list(self._profiles.keys())

    def apply_lut(
        self, bgr: np.ndarray, lut_name: str, strength: float = 1.0
    ) -> np.ndarray:
        if strength <= 0.001 or lut_name not in self._profiles:
            return bgr

        profile = self._profiles[lut_name]
        h, w = bgr.shape[:2]

        # Calculate luminance (Y)
        b = bgr[:, :, 0].astype(np.float32)
        g = bgr[:, :, 1].astype(np.float32)
        r = bgr[:, :, 2].astype(np.float32)
        luma = (0.114 * b + 0.587 * g + 0.299 * r) / 255.0  # (H, W) in [0, 1]

        shadow_weight = np.clip(1.0 - (luma * 2.0), 0.0, 1.0)[:, :, None]
        highlight_weight = np.clip((luma * 2.0) - 1.0, 0.0, 1.0)[:, :, None]

        shadow_shift = profile["shadow_color"] * shadow_weight
        highlight_shift = profile["highlight_color"] * highlight_weight
        total_shift = shadow_shift + highlight_shift

        graded = bgr.astype(np.float32) + (total_shift * strength)

        sat_mult = profile.get("saturation_mult")
        if sat_mult is not None and sat_mult != 1.0:
            eff_sat = 1.0 + (sat_mult - 1.0) * strength
            luma_3d = (luma * 255.0)[:, :, None]
            graded = luma_3d + (graded - luma_3d) * eff_sat

        return np.clip(graded, 0, 255).astype(np.uint8)


class ReShadeFilmGrain:
    """Procedural analog film grain shader matching ReShade FilmGrain.fx."""

    def __init__(self) -> None:
        self._noise_table: np.ndarray | None = None
        self._table_size = 512
        self._reseed_counter = 0

    def _get_noise_patch(self, w: int, h: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        noise = rng.normal(loc=0.0, scale=1.0, size=(h, w)).astype(np.float32)
        return noise

    def apply(
        self, bgr: np.ndarray, intensity: float, size: float = 1.5
    ) -> np.ndarray:
        if intensity <= 0.001:
            return bgr

        h, w = bgr.shape[:2]
        seed = int(time.perf_counter() * 1000) % 65536

        # Fast downsampled noise generation then resize
        nh = max(32, int(h / max(1.0, size)))
        nw = max(32, int(w / max(1.0, size)))

        noise_small = self._get_noise_patch(nw, nh, seed)
        noise = cv2.resize(noise_small, (w, h), interpolation=cv2.INTER_LINEAR)

        # Luminance weighting: film grain is most pronounced in midtones and suppressed in pure blacks and whites
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        luma_mask = 4.0 * gray * (1.0 - gray)  # Parabola peaking at 0.5 midtones

        grain = noise * (luma_mask * intensity * 45.0)
        grain_3d = grain[:, :, None]

        out = bgr.astype(np.float32) + grain_3d
        return np.clip(out, 0, 255).astype(np.uint8)


class ReShadeCAS:
    """Contrast Adaptive Sharpening (AMD FidelityFX / ReShade CAS.fx)."""

    @staticmethod
    def apply(bgr: np.ndarray, sharpness: float = 0.5) -> np.ndarray:
        if sharpness <= 0.01:
            return bgr

        # Fast 3x3 unsharp mask kernel simulating CAS adaptive contrast
        k_amount = sharpness * 0.6
        kernel = np.array(
            [
                [0.0, -k_amount * 0.5, 0.0],
                [-k_amount * 0.5, 1.0 + (k_amount * 2.0), -k_amount * 0.5],
                [0.0, -k_amount * 0.5, 0.0],
            ],
            dtype=np.float32,
        )
        return cv2.filter2D(bgr, -1, kernel)


class ReShadeEngine:
    """Master orchestrator executing ReShade FX passes on video frames."""

    def __init__(self) -> None:
        self.settings = ReShadeSettings()
        self.lut_manager = ReShadeLutManager()
        self.tonemap_lut = FastTonemapLUT()
        self.film_grain = ReShadeFilmGrain()
        self.cas = ReShadeCAS()

    def process_frame(self, rgba_frame: np.ndarray) -> np.ndarray:
        """Apply active ReShade effects to an RGBA8 frame and return processed RGBA8."""
        if not self.settings.enabled:
            return rgba_frame

        h, w = rgba_frame.shape[:2]
        # Work in BGR color space for ultra-fast OpenCV / SIMD processing
        bgr = cv2.cvtColor(rgba_frame, cv2.COLOR_RGBA2BGR)

        # 1. 3D LUT Color Grading
        if self.settings.lut_enabled:
            bgr = self.lut_manager.apply_lut(
                bgr, self.settings.lut_name, self.settings.lut_strength
            )

        # 2. ACES Tonemap & Temperature
        if self.settings.tonemap_enabled:
            lut_b, lut_g, lut_r = self.tonemap_lut.get_tables(
                self.settings.exposure,
                self.settings.contrast,
                self.settings.color_temperature,
            )
            b = cv2.LUT(bgr[:, :, 0], lut_b)
            g = cv2.LUT(bgr[:, :, 1], lut_g)
            r = cv2.LUT(bgr[:, :, 2], lut_r)
            bgr = cv2.merge([b, g, r])

            # Saturation adjustment
            if abs(self.settings.saturation - 1.0) > 0.01:
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.settings.saturation, 0, 255)
                bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # 3. Contrast Adaptive Sharpening (CAS)
        if self.settings.cas_enabled and self.settings.cas_sharpness > 0.01:
            bgr = self.cas.apply(bgr, self.settings.cas_sharpness)

        # 4. ReShade Film Grain
        if self.settings.grain_enabled and self.settings.grain_intensity > 0.01:
            bgr = self.film_grain.apply(
                bgr, self.settings.grain_intensity, self.settings.grain_size
            )

        # Re-assemble RGBA with original alpha channel preserved
        out_rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
        out_rgba[:, :, 3] = rgba_frame[:, :, 3]
        return out_rgba
