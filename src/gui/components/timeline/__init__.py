"""GUI Components for Timeline Studio."""

from .inspector import ClipInspectorWidget
from .media_pool import MediaPoolWidget
from .monitor import TimelineMonitorWidget
from .time_ruler import TimeRulerWidget
from .timeline_bar import ClickableSlider, VideoTimelineBar
from .track_canvas import MultiTrackCanvas

__all__ = [
    "ClipInspectorWidget",
    "ClickableSlider",
    "MediaPoolWidget",
    "MultiTrackCanvas",
    "TimelineMonitorWidget",
    "TimeRulerWidget",
    "VideoTimelineBar",
]
