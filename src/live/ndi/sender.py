"""NDI 6 Stream Sender: Broadcasts enhanced frames back onto the local network."""

from __future__ import annotations

import ctypes
import threading
import time

import numpy as np

from .runtime import (
    NDIlib_FourCC_video_type_e,
    NDIlib_frame_format_type_e,
    NDIlib_send_create_t,
    NDIlib_tally_t,
    NDIlib_video_frame_v2_t,
    get_ndi_lib,
)


class NdiSender:
    """Broadcasts video and audio frames as a named NDI network stream."""

    def __init__(self, stream_name: str = "DLSS 5 Visual Enhancer Studio") -> None:
        self._stream_name = stream_name
        self._lock = threading.Lock()
        self._instance: int | None = None
        self._frame_count = 0

    @property
    def stream_name(self) -> str:
        return self._stream_name

    @property
    def is_broadcasting(self) -> bool:
        with self._lock:
            return self._instance is not None

    def start(self) -> None:
        """Create the NDI sender and advertise stream onto the network."""
        with self._lock:
            if self._instance is not None:
                return

            lib = get_ndi_lib()
            settings = NDIlib_send_create_t(
                p_ndi_name=self._stream_name.encode("utf-8"),
                p_groups=None,
                clock_video=False,  # Video paced by rendering loop
                clock_audio=False,
            )
            instance = lib.NDIlib_send_create(ctypes.byref(settings))
            if not instance:
                raise RuntimeError(f"Failed to create NDI sender for '{self._stream_name}'.")
            self._instance = instance

    def stop(self) -> None:
        """Destroy sender instance and remove stream from network."""
        with self._lock:
            instance = self._instance
            self._instance = None

        if instance:
            lib = get_ndi_lib()
            lib.NDIlib_send_destroy(instance)

    def send_video_frame(self, rgba_frame: np.ndarray, fps: float = 60.0) -> None:
        """Dispatch a single RGBA8 frame to NDI receivers."""
        with self._lock:
            if not self._instance:
                return

            if rgba_frame.dtype != np.uint8 or rgba_frame.ndim != 3 or rgba_frame.shape[2] != 4:
                raise ValueError("NDI frame must be uint8 RGBA of shape (H, W, 4).")

            if not rgba_frame.flags.c_contiguous:
                rgba_frame = np.ascontiguousarray(rgba_frame)

            h, w = rgba_frame.shape[:2]
            stride = int(rgba_frame.strides[0])

            fps_n = int(round(fps * 1000))
            fps_d = 1000

            lib = get_ndi_lib()
            frame = NDIlib_video_frame_v2_t()
            frame.xres = w
            frame.yres = h
            frame.FourCC = NDIlib_FourCC_video_type_e.NDIlib_FourCC_type_RGBA
            frame.frame_rate_N = fps_n
            frame.frame_rate_D = fps_d
            frame.picture_aspect_ratio = float(w) / float(h) if h > 0 else 1.7777
            frame.frame_format_type = NDIlib_frame_format_type_e.NDIlib_frame_format_type_progressive
            frame.timecode = int(time.perf_counter() * 10_000_000)
            frame.p_data = ctypes.c_void_p(rgba_frame.ctypes.data)
            frame.line_stride_in_bytes = stride
            frame.p_metadata = None
            frame.timestamp = 0

            lib.NDIlib_send_send_video_v2(self._instance, ctypes.byref(frame))
            self._frame_count += 1

    def get_tally(self) -> tuple[bool, bool]:
        """Return (on_program, on_preview) tally status indicating if a switcher is displaying us."""
        with self._lock:
            if not self._instance:
                return False, False

            lib = get_ndi_lib()
            tally = NDIlib_tally_t()
            if lib.NDIlib_send_get_tally(self._instance, ctypes.byref(tally), 0):
                return bool(tally.on_program), bool(tally.on_preview)
            return False, False
