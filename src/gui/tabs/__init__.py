"""GUI tab modules."""

from .about_tab import AboutTab
from .frame_interpolation_tab import FrameInterpolationTab
from .neural_rendering_tab import NeuralRenderingTab
from .realtime_rendering_tab import RealtimeRenderingTab
from .realtime_upscale_tab import RealtimeUpscaleTab
from .settings_tab import SettingsTab
from .upscale_tab import UpscaleTab

__all__ = [
    "AboutTab",
    "FrameInterpolationTab",
    "NeuralRenderingTab",
    "RealtimeRenderingTab",
    "RealtimeUpscaleTab",
    "SettingsTab",
    "UpscaleTab",
]
