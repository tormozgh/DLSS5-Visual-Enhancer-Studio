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
    is_hardware: bool = True

    @property
    def display_name(self) -> str:
        tag = "" if self.is_hardware else " [Virtual]"
        if self.name and self.name != f"Camera {self.index}":
            return f"{self.name} ({self.width}x{self.height}){tag}"
        return f"Camera {self.index} ({self.width}x{self.height}){tag}"


class WebcamReceiver:
    """Asynchronous DirectShow webcam and capture card frame ingest."""

    _cached_devices: list[CameraDeviceInfo] | None = None
    _cache_time: float = 0.0
    _scan_lock: threading.Lock = threading.Lock()
    _is_scanning: bool = False

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
    def _enum_dshow_monikers_ctypes(cls) -> list[tuple[str, str]]:
        """Fallback DirectShow device enumeration using pure Windows COM ctypes."""
        import ctypes
        from ctypes import wintypes, POINTER, byref, c_void_p, Structure, cast

        ole32 = ctypes.windll.ole32
        ole32.CoInitialize(None)

        class GUID(Structure):
            _fields_ = [
                ("Data1", wintypes.DWORD),
                ("Data2", wintypes.WORD),
                ("Data3", wintypes.WORD),
                ("Data4", wintypes.BYTE * 8),
            ]

            def __init__(self, s):
                super().__init__()
                ole32.CLSIDFromString(ctypes.c_wchar_p(s), byref(self))

        CLSID_SystemDeviceEnum = GUID("{62BE5D10-60EB-11D0-BD3B-00A0C911CE86}")
        CLSID_VideoInputDeviceCategory = GUID("{860BB310-5D01-11D0-BD3B-00A0C911CE86}")
        IID_ICreateDevEnum = GUID("{29840822-5B84-11D0-BD3B-00A0C911CE86}")
        IID_IPropertyBag = GUID("{55272A00-42CB-11CE-8135-00AA004BB851}")

        pDevEnum = c_void_p()
        hr = ole32.CoCreateInstance(
            byref(CLSID_SystemDeviceEnum), None, 1, byref(IID_ICreateDevEnum), byref(pDevEnum)
        )
        if hr != 0 or not pDevEnum.value:
            return []

        vtable = cast(pDevEnum, POINTER(POINTER(c_void_p))).contents
        CreateClassEnumerator = ctypes.WINFUNCTYPE(
            ctypes.c_long, c_void_p, POINTER(GUID), POINTER(c_void_p), wintypes.DWORD
        )(vtable[3])

        pEnum = c_void_p()
        hr_enum = CreateClassEnumerator(pDevEnum, byref(CLSID_VideoInputDeviceCategory), byref(pEnum), 0)
        if hr_enum != 0 or not pEnum.value:
            return []

        enum_vtable = cast(pEnum, POINTER(POINTER(c_void_p))).contents
        Next_func = ctypes.WINFUNCTYPE(
            ctypes.c_long, c_void_p, wintypes.ULONG, POINTER(c_void_p), POINTER(wintypes.ULONG)
        )(enum_vtable[3])

        class VARIANT(Structure):
            _fields_ = [
                ("vt", wintypes.WORD),
                ("wReserved1", wintypes.WORD),
                ("wReserved2", wintypes.WORD),
                ("wReserved3", wintypes.WORD),
                ("bstrVal", c_void_p),
                ("dummy", wintypes.BYTE * 8),
            ]

        results = []
        idx = 0
        while True:
            pMoniker = c_void_p()
            fetched = wintypes.ULONG(0)
            if Next_func(pEnum, 1, byref(pMoniker), byref(fetched)) != 0 or fetched.value == 0:
                break

            mon_vtable = cast(pMoniker, POINTER(POINTER(c_void_p))).contents
            BindToStorage_func = ctypes.WINFUNCTYPE(
                ctypes.c_long, c_void_p, c_void_p, c_void_p, POINTER(GUID), POINTER(c_void_p)
            )(mon_vtable[9])

            pPropBag = c_void_p()
            name = f"Camera {idx}"
            path = ""
            if BindToStorage_func(pMoniker, None, None, byref(IID_IPropertyBag), byref(pPropBag)) == 0 and pPropBag.value:
                prop_vtable = cast(pPropBag, POINTER(POINTER(c_void_p))).contents
                Read_func = ctypes.WINFUNCTYPE(
                    ctypes.c_long, c_void_p, ctypes.c_wchar_p, POINTER(VARIANT), c_void_p
                )(prop_vtable[3])

                var = VARIANT()
                if Read_func(pPropBag, "FriendlyName", byref(var), None) == 0:
                    if var.vt == 8:
                        name = ctypes.wstring_at(var.bstrVal)

                var2 = VARIANT()
                if Read_func(pPropBag, "DevicePath", byref(var2), None) == 0:
                    if var2.vt == 8:
                        path = ctypes.wstring_at(var2.bstrVal)

                ctypes.WINFUNCTYPE(wintypes.ULONG, c_void_p)(prop_vtable[2])(pPropBag)

            results.append((name, path))
            ctypes.WINFUNCTYPE(wintypes.ULONG, c_void_p)(mon_vtable[2])(pMoniker)
            idx += 1

        ctypes.WINFUNCTYPE(wintypes.ULONG, c_void_p)(enum_vtable[2])(pEnum)
        ctypes.WINFUNCTYPE(wintypes.ULONG, c_void_p)(vtable[2])(pDevEnum)
        return results

    @classmethod
    def list_cameras(cls, max_probe: int = 10, force_refresh: bool = False) -> list[CameraDeviceInfo]:
        """Enumerate video capture devices dynamically on the host system via DirectShow.

        Directly maps Windows DirectShow device monikers to their exact OpenCV capture index (0..N).
        Filters and prioritizes physical hardware webcams and HDMI capture cards over virtual devices.
        """
        now = time.time()
        if not force_refresh and cls._cached_devices is not None and (now - cls._cache_time < 3.0):
            return cls._cached_devices

        with cls._scan_lock:
            if not force_refresh and cls._cached_devices is not None and (time.time() - cls._cache_time < 3.0):
                return cls._cached_devices

            raw_devices: list[tuple[str, str]] = []
            try:
                from cv2_enumerate_cameras._windows_backend import DSHOW_enumerate_cameras
                raw_devices = DSHOW_enumerate_cameras()
            except Exception:
                pass

            if not raw_devices:
                try:
                    raw_devices = cls._enum_dshow_monikers_ctypes()
                except Exception:
                    raw_devices = []

            all_devices: list[CameraDeviceInfo] = []
            for idx, (name, path) in enumerate(raw_devices):
                name_l = name.lower()
                path_l = path.lower() if path else ""

                # Identify virtual camera filters (OBS Virtual Camera, NDI, Spout, Broadcast, etc.)
                is_virtual = (
                    any(k in name_l for k in [
                        "obs virtual", "virtual camera", "virtualcam",
                        "broadcast", "nvidia broadcast", "ndi webcam",
                        "spoutcam", "manycam", "xsplit", "vmix",
                        "epoccam", "droidcam", "snap camera", "screen capture"
                    ])
                    or "root#media" in path_l
                    or not path_l
                )
                is_hardware = not is_virtual and any(
                    bus in path_l for bus in ["usb", "pci", "acpi", "uvc", "pnp"]
                )

                # Sensible resolution heuristics based on device capabilities
                w, h, fps = 1920, 1080, 60.0
                if any(k in name_l for k in ["c310", "c270", "720"]):
                    w, h, fps = 1280, 720, 30.0
                elif any(k in name_l for k in ["4k", "brio", "4k60", "pro capture"]):
                    w, h, fps = 3840, 2160, 60.0
                elif not is_hardware:
                    fps = 30.0

                all_devices.append(
                    CameraDeviceInfo(
                        index=idx,
                        name=name,
                        width=w,
                        height=h,
                        fps=fps,
                        is_hardware=is_hardware,
                    )
                )

            # Fallback if no DirectShow monikers were found (e.g. non-Windows or broken COM)
            if not all_devices:
                for idx in range(min(max_probe, 4)):
                    try:
                        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                        if cap.isOpened():
                            all_devices.append(
                                CameraDeviceInfo(
                                    index=idx,
                                    name=f"Camera {idx}",
                                    width=1920,
                                    height=1080,
                                    fps=30.0,
                                    is_hardware=True,
                                )
                            )
                            cap.release()
                    except Exception:
                        pass

            # Prioritize real physical webcams and capture cards
            hardware_devices = [d for d in all_devices if d.is_hardware]
            final_devices = hardware_devices if hardware_devices else all_devices

            cls._cached_devices = final_devices
            cls._cache_time = now
            return final_devices

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

            # Format negotiation for DirectShow:
            # If target resolution is > 640x480 (e.g. 720p or 1080p), MJPG is required
            # on USB 2.0 webcams (such as Logitech C310) to avoid USB bandwidth limits.
            # CRITICAL: Do NOT call cap.set(cv2.CAP_PROP_FPS, ...) because OpenCV's
            # DShow implementation resets FourCC back to uncompressed YUY2 when setting FPS!
            if self.target_width > 640 or self.target_height > 480:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            else:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)

            try:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            except Exception:
                pass

            # Validate initial frame reception via grab/retrieve
            initial_frame: np.ndarray | None = None
            for _ in range(3):
                if not self._running:
                    break
                t_init = time.perf_counter()
                grabbed = cap.grab()
                if not grabbed:
                    time.sleep(0.01)
                    continue
                ret, frame = cap.retrieve()
                dt_init = time.perf_counter() - t_init
                if ret and frame is not None and frame.size > 0 and dt_init < 0.85:
                    initial_frame = frame
                    break
                time.sleep(0.01)

            # If high-res or MJPG failed or timed out, fallback to native default resolution (640x480)
            if initial_frame is None and self._running:
                cap.release()
                cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    with self._lock:
                        self._running = False
                    return
                for _ in range(3):
                    if not self._running:
                        break
                    t_init = time.perf_counter()
                    grabbed = cap.grab()
                    if not grabbed:
                        time.sleep(0.01)
                        continue
                    ret, frame = cap.retrieve()
                    dt_init = time.perf_counter() - t_init
                    if ret and frame is not None and frame.size > 0 and dt_init < 0.85:
                        initial_frame = frame
                        break
                    time.sleep(0.01)

            if initial_frame is not None:
                with self._lock:
                    self._actual_height, self._actual_width = initial_frame.shape[:2]
                    rep_fps = cap.get(cv2.CAP_PROP_FPS)
                    self._actual_fps = rep_fps if (rep_fps and rep_fps > 10.0) else self.target_fps
                    self._cap = cap

                rgba_first = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2RGBA)
                now = time.perf_counter()
                if self.on_video_frame and self._running:
                    try:
                        self.on_video_frame(rgba_first, int(now * 1000), self._actual_fps)
                    except Exception:
                        pass
            else:
                with self._lock:
                    self._cap = cap

            last_frame_time = time.perf_counter()

            while self._running:
                t_grab_start = time.perf_counter()
                grabbed = cap.grab()
                if not grabbed:
                    time.sleep(0.005)
                    continue

                ret, bgr_frame = cap.retrieve()
                grab_dt = time.perf_counter() - t_grab_start

                # Strict frame validation:
                # If retrieve returned False or took >= 0.85s (indicating DirectShow driver timeout),
                # drop this frame to prevent injecting 1 FPS black screens into the pipeline
                if not ret or bgr_frame is None or bgr_frame.size == 0 or grab_dt >= 0.85:
                    time.sleep(0.005)
                    continue

                h, w = bgr_frame.shape[:2]
                if w != self._actual_width or h != self._actual_height:
                    with self._lock:
                        self._actual_width = w
                        self._actual_height = h

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

