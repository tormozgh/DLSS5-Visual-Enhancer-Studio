"""ReShade FX Preset Management: Built-in cinematic presets and custom preset I/O."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .effects import ReShadeSettings


DEFAULT_PRESETS: dict[str, dict[str, Any]] = {
    "Broadcast Clean": {
        "enabled": True,
        "lut_enabled": False,
        "lut_name": "Neutral Broadcast",
        "lut_strength": 0.5,
        "tonemap_enabled": True,
        "exposure": 0.0,
        "contrast": 1.05,
        "saturation": 1.05,
        "color_temperature": 0.0,
        "grain_enabled": False,
        "grain_intensity": 0.0,
        "grain_size": 1.5,
        "grain_colored": False,
        "cas_enabled": True,
        "cas_sharpness": 0.35,
    },
    "Cinematic Teal & Orange": {
        "enabled": True,
        "lut_enabled": True,
        "lut_name": "Cinematic Teal & Orange",
        "lut_strength": 0.85,
        "tonemap_enabled": True,
        "exposure": 0.05,
        "contrast": 1.10,
        "saturation": 1.15,
        "color_temperature": 0.05,
        "grain_enabled": True,
        "grain_intensity": 0.15,
        "grain_size": 1.5,
        "grain_colored": False,
        "cas_enabled": True,
        "cas_sharpness": 0.40,
    },
    "Warm Golden Hour": {
        "enabled": True,
        "lut_enabled": True,
        "lut_name": "Warm Golden Hour",
        "lut_strength": 0.80,
        "tonemap_enabled": True,
        "exposure": 0.10,
        "contrast": 1.08,
        "saturation": 1.20,
        "color_temperature": 0.25,
        "grain_enabled": True,
        "grain_intensity": 0.12,
        "grain_size": 1.5,
        "grain_colored": False,
        "cas_enabled": True,
        "cas_sharpness": 0.30,
    },
    "Bleach Bypass": {
        "enabled": True,
        "lut_enabled": True,
        "lut_name": "Bleach Bypass",
        "lut_strength": 0.90,
        "tonemap_enabled": True,
        "exposure": -0.05,
        "contrast": 1.25,
        "saturation": 0.60,
        "color_temperature": -0.10,
        "grain_enabled": True,
        "grain_intensity": 0.22,
        "grain_size": 1.8,
        "grain_colored": False,
        "cas_enabled": True,
        "cas_sharpness": 0.50,
    },
    "Cyberpunk Neon": {
        "enabled": True,
        "lut_enabled": True,
        "lut_name": "Cyberpunk Neon",
        "lut_strength": 0.90,
        "tonemap_enabled": True,
        "exposure": 0.0,
        "contrast": 1.18,
        "saturation": 1.35,
        "color_temperature": -0.15,
        "grain_enabled": False,
        "grain_intensity": 0.0,
        "grain_size": 1.5,
        "grain_colored": False,
        "cas_enabled": True,
        "cas_sharpness": 0.45,
    },
    "Technicolor Vintage": {
        "enabled": True,
        "lut_enabled": True,
        "lut_name": "Technicolor 3-Strip",
        "lut_strength": 0.80,
        "tonemap_enabled": True,
        "exposure": 0.0,
        "contrast": 1.05,
        "saturation": 1.25,
        "color_temperature": 0.10,
        "grain_enabled": True,
        "grain_intensity": 0.20,
        "grain_size": 1.6,
        "grain_colored": False,
        "cas_enabled": True,
        "cas_sharpness": 0.30,
    },
}


class ReShadePresetManager:
    """Manages preset loading, saving, and selection."""

    def __init__(self, presets_dir: Path | None = None) -> None:
        self.presets_dir = presets_dir or Path("presets") / "reshade"
        self.presets_dir.mkdir(parents=True, exist_ok=True)

    def get_preset_names(self) -> list[str]:
        builtins = list(DEFAULT_PRESETS.keys())
        customs = [p.stem for p in self.presets_dir.glob("*.json")]
        return builtins + [f"Custom: {c}" for c in customs if c not in builtins]

    def load_preset(self, name: str) -> ReShadeSettings:
        clean_name = name.removeprefix("Custom: ").strip()
        if clean_name in DEFAULT_PRESETS:
            return ReShadeSettings(**DEFAULT_PRESETS[clean_name])

        custom_path = self.presets_dir / f"{clean_name}.json"
        if custom_path.is_file():
            try:
                data = json.loads(custom_path.read_text(encoding="utf-8"))
                return ReShadeSettings(**data)
            except Exception:
                pass
        return ReShadeSettings()

    def save_preset(self, name: str, settings: ReShadeSettings) -> Path:
        clean_name = name.removeprefix("Custom: ").strip()
        target = self.presets_dir / f"{clean_name}.json"
        data = asdict(settings)
        target.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return target
