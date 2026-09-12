"""NDI 6 Stream Receiver: Captures progressive video & audio frames from network sources."""

from __future__ import annotations

import ctypes
import threading
import time
from typing import Callable

import numpy as np

from .runtime import (
    NDIlib_FourCC_video_type_e,
    NDIlib_audio_frame_v2_t,
    NDIlib_frame_type_e,
    NDIlib_metadata_frame_t,
    NDIlib_recv_bandwidth_e,
    NDIlib_recv_color_format_e,
    NDIlib_recv_create_v3_t,
    NDIlib_source_t,
    NDIlib_video_frame_v2_t,
    get_ndi_lib,
)


class NdiReceiver:
    """Connects to an NDI source and yields live progressive video frames."""

    def __init__(
        self,
        source_name: str,
        url_address: str | None = None,
        on_video_frame: Callable[[np.ndarray, int, float], None] | None = None,
        on_audio_frame: Callable[[NDIlib_audio_frame_v2_t], None] | None = None,
    ) -> None:
        self._source_name = source_name
        self._url_address = url_address
        self._on_video = on_video_frame
        self._on_audio = on_audio_frame
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._instance: int | None = None
        self._current_fps: float = 0.0
        self._frame_count: int = 0
        self._last_width: int = 0
        self._last_height: int = 0

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._running and self._instance is not None

    @property
    def resolution(self) -> tuple[int, int]:
        return self._last_width, self._last_height

    @property
    def fps(self) -> float:
        return self._current_fps

    def start(self) -> None:
        """Connect to NDI source and launch capture thread."""
        with self._lock:
            if self._running:
                return
            self._running = True

            lib = get_ndi_lib()
            p_url = self._url_address.encode("utf-8") if self._url_address else None
            src = NDIlib_source_t(
                p_ndi_name=self._source_name.encode("utf-8"),
                p_url_address=p_url,
            )

            # Prefer RGBX/RGBA color space for direct Direct3D 12 / neural bridge compatibility
            settings = NDIlib_recv_create_v3_t(
                source_to_connect_to=src,
                color_format=NDIlib_recv_color_format_e.NDIlib_recv_color_format_RGBX_RGBA,
                bandwidth=NDIlib_recv_bandwidth_e.NDIlib_recv_bandwidth_highest,
                allow_video_fields=False,  # Force progressive deinterlace
                p_ndi_recv_name=b"DLSS 5 Studio Receiver",
            )

            instance = lib.NDIlib_recv_create_v3(ctypes.byref(settings))
            if not instance:
                self._running = False
                raise RuntimeError(f"Failed to connect to NDI source '{self._source_name}'.")
            self._instance = instance

            self._thread = threading.Thread(
                target=self._capture_loop,
                name="dlss5-ndi-receiver",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """Disconnect and release receiver resources."""
        with self._lock:
            self._running = False
            instance = self._instance
            self._instance = None

        if instance:
            lib = get_ndi_lib()
            lib.NDIlib_recv_destroy(instance)

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None

    def _capture_loop(self) -> None:
        lib = get_ndi_lib()
        v_frame = NDIlib_video_frame_v2_t()
        a_frame = NDIlib_audio_frame_v2_t()
        m_frame = NDIlib_metadata_frame_t()

        while self._running and self._instance:
            frame_type = lib.NDIlib_recv_capture_v2(
                self._instance,
                ctypes.byref(v_frame),
                ctypes.byref(a_frame),
                ctypes.byref(m_frame),
                100,  # 100ms timeout
            )

            if not self._running or not self._instance:
                break

            if frame_type == NDIlib_frame_type_e.NDIlib_frame_type_video:
                w = v_frame.xres
                h = v_frame.yres
                stride = v_frame.line_stride_in_bytes or (w * 4)
                fps_n = v_frame.frame_rate_N
                fps_d = v_frame.frame_rate_D or 1
                fps = fps_n / fps_d if fps_n > 0 else 30.0

                self._last_width = w
                self._last_height = h
                self._current_fps = fps

                # Convert raw memory pointer into NumPy RGBA array with minimal copy
                if v_frame.p_data:
                    buf_size = stride * h
                    raw_bytes = (ctypes.c_uint8 * buf_size).from_address(v_frame.p_data)
                    arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((h, stride // 4, 4))
                    if stride != w * 4:
                        arr = arr[:, :w, :]
                    frame_copy = arr.copy()
                else:
                    frame_copy = np.zeros((h, w, 4), dtype=np.uint8)

                timestamp = v_frame.timestamp or int(time.perf_counter() * 10_000_000)
                lib.NDIlib_recv_free_video_v2(self._instance, ctypes.byref(v_frame))

                self._frame_count += 1
                if self._on_video:
                    try:
                        self._on_video(frame_copy, timestamp, fps)
                    except Exception:
                        pass

            elif frame_type == NDIlib_frame_type_e.NDIlib_frame_type_audio:
                if self._on_audio:
                    try:
                        self._on_audio(a_frame)
                    except Exception:
                        pass
                lib.NDIlib_recv_free_audio_v2(self._instance, ctypes.byref(a_frame))

            elif frame_type == NDIlib_frame_type_e.NDIlib_frame_type_none:
                continue
