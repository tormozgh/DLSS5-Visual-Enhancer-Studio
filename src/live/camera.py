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
        if self.name and self.name != f"Camera {self.index}":
            return f"[{self.index}] {self.name}"
        return f"Camera {self.index} ({self.width}x{self.height})"


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

        self._actual_width = target_width
        self._actual_height = target_height
        self._actual_fps = target_fps

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
    def list_cameras(cls, max_probe: int = 8, force_refresh: bool = False) -> list[CameraDeviceInfo]:
        """Enumerate available video capture hardware devices via DirectShow Registry instantaneously."""
        now = time.time()
        if not force_refresh and cls._cached_devices is not None and (now - cls._cache_time < 2.0):
            return cls._cached_devices

        devices: list[CameraDeviceInfo] = []

        # DirectShow Video Input Device Category CLSID
        # Fast query via Windows Registry without hardware initialization stalls (< 1ms)
        try:
            import winreg

            seen_names: set[str] = set()
            idx = 0
            for root_key in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
                reg_paths = [
                    r"SOFTWARE\Classes\CLSID\{860BB310-5D01-11d0-BD3B-00A0C911CE86}\Instance",
                    r"SOFTWARE\WOW6432Node\Classes\CLSID\{860BB310-5D01-11d0-BD3B-00A0C911CE86}\Instance",
                ]
                for path in reg_paths:
                    try:
                        key = winreg.OpenKey(root_key, path)
                    except OSError:
                        continue

                    sub_idx = 0
                    while True:
                        try:
                            subkey_name = winreg.EnumKey(key, sub_idx)
                            subkey = winreg.OpenKey(key, subkey_name)
                            try:
                                friendly_name, _ = winreg.QueryValueEx(subkey, "FriendlyName")
                            except FileNotFoundError:
                                friendly_name = f"Capture Device {idx}"

                            if friendly_name and friendly_name not in seen_names:
                                seen_names.add(friendly_name)
                                devices.append(
                                    CameraDeviceInfo(
                                        index=idx,
                                        name=str(friendly_name),
                                        width=1920,
                                        height=1080,
                                        fps=60.0,
                                    )
                                )
                                idx += 1
                            sub_idx += 1
                        except OSError:
                            break

            if devices:
                cls._cached_devices = devices
                cls._cache_time = now
                return devices
        except Exception:
            pass

        # Fallback if registry query unsupported (e.g. non-Windows)
        for idx in range(min(max_probe, 2)):
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if cap.isOpened():
                    devices.append(CameraDeviceInfo(index=idx, name=f"Camera {idx}", width=1920, height=1080, fps=60.0))
                    cap.release()
            except Exception:
                pass

        cls._cached_devices = devices
        cls._cache_time = now
        return devices

    def start(self) -> None:
        """Start asynchronous camera capture thread non-blockingly."""
        with self._lock:
            if self._running:
                return
            self._running = True

            self._thread = threading.Thread(
                target=self._capture_worker,
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

    def _capture_worker(self) -> None:
        cap: cv2.VideoCapture | None = None
        try:
            cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                with self._lock:
                    self._running = False
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)

            # Read initial frame to latch actual hardware dimensions
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                with self._lock:
                    self._running = False
                return

            with self._lock:
                self._actual_height, self._actual_width = frame.shape[:2]
                rep_fps = cap.get(cv2.CAP_PROP_FPS)
                self._actual_fps = rep_fps if (rep_fps and rep_fps > 10.0) else self.target_fps
                self._cap = cap

            # Process first frame immediately
            rgba_first = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            now = time.perf_counter()
            if self.on_video_frame and self._running:
                try:
                    self.on_video_frame(rgba_first, int(now * 1000), self._actual_fps)
                except Exception:
                    pass

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
        finally:
            with self._lock:
                if cap and cap.isOpened():
                    try:
                        cap.release()
                    except Exception:
                        pass
                self._cap = None
                self._running = False
