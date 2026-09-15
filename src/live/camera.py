"""Hardware DirectShow Camera and Video Capture Card Ingest Engine.

Provides asynchronous, low-latency video frame acquisition from USB webcams,
HDMI capture cards (e.g. Elgato, Magewell), and virtual camera devices via DirectShow.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np


@dataclass
class CameraDeviceInfo:
    index: int
    name: str
    width: int = 1920
    height: int = 1080
    fps: float = 60.0

    @property
    def display_name(self) -> str:
        return f"Camera {self.index} ({self.width}x{self.height} @ {int(round(self.fps))} FPS)"


class WebcamReceiver:
    """Asynchronous DirectShow webcam and capture card frame ingest."""

    _cached_devices: list[CameraDeviceInfo] | None = None
    _cache_time: float = 0.0

    def __init__(
        self,
        device_index: int = 0,
        target_width: int = 1920,
        target_height: int = 1080,
        target_fps: float = 60.0,
        on_video_frame: Callable[[np.ndarray, int, float], None] | None = None,
    ) -> None:
        self.device_index = device_index
        self.target_width = target_width
        self.target_height = target_height
        self.target_fps = target_fps
        self.on_video_frame = on_video_frame

        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None

        self._actual_width = 0
        self._actual_height = 0
        self._actual_fps = 0.0

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def resolution(self) -> tuple[int, int]:
        with self._lock:
            return self._actual_width, self._actual_height

    @property
    def fps(self) -> float:
        with self._lock:
            return self._actual_fps

    @classmethod
    def list_cameras(cls, max_probe: int = 6, force_refresh: bool = False) -> list[CameraDeviceInfo]:
        """Enumerate available video capture hardware devices via DirectShow."""
        now = time.time()
        if not force_refresh and cls._cached_devices is not None and (now - cls._cache_time < 30.0):
            return cls._cached_devices

        devices: list[CameraDeviceInfo] = []
        for idx in range(max_probe):
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if cap.isOpened():
                    # Probe actual capabilities
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                    cap.set(cv2.CAP_PROP_FPS, 60.0)

                    ret, frame = cap.read()
                    if ret and frame is not None:
                        h, w = frame.shape[:2]
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        if fps <= 0 or fps > 240:
                            fps = 60.0
                        devices.append(CameraDeviceInfo(index=idx, name=f"Camera {idx}", width=w, height=h, fps=fps))
                    cap.release()
            except Exception:
                pass

        cls._cached_devices = devices
        cls._cache_time = now
        return devices

    def start(self) -> None:
        """Start asynchronous camera capture thread."""
        with self._lock:
            if self._running:
                return
            self._running = True

            cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                self._running = False
                raise RuntimeError(f"Failed to open camera device index {self.device_index}")

            # Request desired video format
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)

            # Read one frame to verify actual dimensions
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                self._running = False
                raise RuntimeError(f"Cannot capture initial frame from camera index {self.device_index}")

            self._actual_height, self._actual_width = frame.shape[:2]
            reported_fps = cap.get(cv2.CAP_PROP_FPS)
            self._actual_fps = reported_fps if (reported_fps and reported_fps > 10.0) else self.target_fps
            self._cap = cap

            self._thread = threading.Thread(
                target=self._capture_loop,
                name=f"dlss5-camera-worker-{self.device_index}",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop capture thread and release camera device."""
        with self._lock:
            self._running = False
            cap = self._cap
            self._cap = None

        if cap:
            try:
                cap.release()
            except Exception:
                pass

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None

    def _capture_loop(self) -> None:
        cap = self._cap
        if not cap:
            return

        frame_interval = 1.0 / max(1.0, self._actual_fps)
        last_frame_time = time.perf_counter()

        while self._running:
            ret, bgr_frame = cap.read()
            if not ret or bgr_frame is None:
                time.sleep(0.005)
                continue

            # Convert BGR to RGBA for direct pipeline compatibility
            rgba_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGBA)
            now = time.perf_counter()
            ts = int(now * 1000)

            dt = now - last_frame_time
            curr_fps = (1.0 / dt) if dt > 0.001 else self._actual_fps
            last_frame_time = now

            if self.on_video_frame and self._running:
                try:
                    self.on_video_frame(rgba_frame, ts, curr_fps)
                except Exception:
                    pass
