"""NDI 6 C-ABI runtime bindings and structure definitions."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any

# ==============================================================================
# 1. Enums & Constants
# ==============================================================================

class NDIlib_frame_type_e:
    NDIlib_frame_type_none = 0
    NDIlib_frame_type_video = 1
    NDIlib_frame_type_audio = 2
    NDIlib_frame_type_metadata = 3
    NDIlib_frame_type_error = 4
    NDIlib_frame_type_status_change = 100

class NDIlib_FourCC_video_type_e:
    NDIlib_FourCC_type_UYVY = 0x59565955
    NDIlib_FourCC_type_BGRA = 0x41524742
    NDIlib_FourCC_type_BGRX = 0x58524742
    NDIlib_FourCC_type_RGBA = 0x41424752
    NDIlib_FourCC_type_RGBX = 0x58424752
    NDIlib_FourCC_type_NV12 = 0x3231564E

class NDIlib_frame_format_type_e:
    NDIlib_frame_format_type_progressive = 1
    NDIlib_frame_format_type_interleaved = 2
    NDIlib_frame_format_type_field_0 = 3
    NDIlib_frame_format_type_field_1 = 4

class NDIlib_recv_bandwidth_e:
    NDIlib_recv_bandwidth_metadata_only = -10
    NDIlib_recv_bandwidth_audio_only = 10
    NDIlib_recv_bandwidth_lowest = 0
    NDIlib_recv_bandwidth_highest = 100

class NDIlib_recv_color_format_e:
    NDIlib_recv_color_format_BGRX_BGRA = 0
    NDIlib_recv_color_format_UYVY_BGRA = 1
    NDIlib_recv_color_format_RGBX_RGBA = 2
    NDIlib_recv_color_format_UYVY_RGBA = 3
    NDIlib_recv_color_format_fastest = 100
    NDIlib_recv_color_format_best = 101

# ==============================================================================
# 2. C Structure Definitions
# ==============================================================================

class NDIlib_source_t(ctypes.Structure):
    _fields_ = [
        ("p_ndi_name", ctypes.c_char_p),
        ("p_url_address", ctypes.c_char_p),
    ]

    def name(self) -> str:
        return self.p_ndi_name.decode("utf-8", errors="ignore") if self.p_ndi_name else ""

    def address(self) -> str:
        return self.p_url_address.decode("utf-8", errors="ignore") if self.p_url_address else ""

class NDIlib_video_frame_v2_t(ctypes.Structure):
    _fields_ = [
        ("xres", ctypes.c_int),
        ("yres", ctypes.c_int),
        ("FourCC", ctypes.c_uint32),
        ("frame_rate_N", ctypes.c_int),
        ("frame_rate_D", ctypes.c_int),
        ("picture_aspect_ratio", ctypes.c_float),
        ("frame_format_type", ctypes.c_int),
        ("timecode", ctypes.c_int64),
        ("p_data", ctypes.c_void_p),
        ("line_stride_in_bytes", ctypes.c_int),
        ("p_metadata", ctypes.c_char_p),
        ("timestamp", ctypes.c_int64),
    ]

class NDIlib_audio_frame_v2_t(ctypes.Structure):
    _fields_ = [
        ("sample_rate", ctypes.c_int),
        ("no_channels", ctypes.c_int),
        ("no_samples", ctypes.c_int),
        ("timecode", ctypes.c_int64),
        ("p_data", ctypes.c_void_p),
        ("channel_stride_in_bytes", ctypes.c_int),
        ("p_metadata", ctypes.c_char_p),
        ("timestamp", ctypes.c_int64),
    ]

class NDIlib_metadata_frame_t(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_int),
        ("timecode", ctypes.c_int64),
        ("p_data", ctypes.c_char_p),
    ]

class NDIlib_find_create_t(ctypes.Structure):
    _fields_ = [
        ("show_local_sources", ctypes.c_bool),
        ("p_groups", ctypes.c_char_p),
        ("p_extra_ips", ctypes.c_char_p),
    ]

class NDIlib_recv_create_v3_t(ctypes.Structure):
    _fields_ = [
        ("source_to_connect_to", NDIlib_source_t),
        ("color_format", ctypes.c_int),
        ("bandwidth", ctypes.c_int),
        ("allow_video_fields", ctypes.c_bool),
        ("p_ndi_recv_name", ctypes.c_char_p),
    ]

class NDIlib_send_create_t(ctypes.Structure):
    _fields_ = [
        ("p_ndi_name", ctypes.c_char_p),
        ("p_groups", ctypes.c_char_p),
        ("clock_video", ctypes.c_bool),
        ("clock_audio", ctypes.c_bool),
    ]

class NDIlib_tally_t(ctypes.Structure):
    _fields_ = [
        ("on_program", ctypes.c_bool),
        ("on_preview", ctypes.c_bool),
    ]

# ==============================================================================
# 3. Dynamic Library Loader & Function Binding
# ==============================================================================

_NDI_LIB: ctypes.WinDLL | None = None
_IS_INITIALIZED: bool = False

def _find_ndi_dll() -> str | None:
    candidates = [
        Path("bin/Processing.NDI.Lib.x64.dll").resolve(),
        Path(r"C:\Program Files\NDI\NDI 6 Runtime\v6\Processing.NDI.Lib.x64.dll"),
        Path(r"C:\Program Files\NDI\NDI 5 Runtime\v5\Processing.NDI.Lib.x64.dll"),
        Path(r"C:\Program Files\NDI\NDI 6 SDK\Lib\x64\Processing.NDI.Lib.x64.dll"),
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return "Processing.NDI.Lib.x64.dll"

def get_ndi_lib() -> ctypes.WinDLL:
    """Load and initialize NDI 6 DLL once with complete C prototypes."""
    global _NDI_LIB, _IS_INITIALIZED
    if _NDI_LIB is not None:
        return _NDI_LIB

    dll_path = _find_ndi_dll()
    try:
        lib = ctypes.WinDLL(dll_path)
    except OSError as exc:
        raise RuntimeError(
            f"Failed to load NDI library from '{dll_path}'. "
            "Please ensure NDI 6 Runtime or SDK is installed."
        ) from exc

    # Lifecycle prototypes
    lib.NDIlib_initialize.argtypes = []
    lib.NDIlib_initialize.restype = ctypes.c_bool

    lib.NDIlib_destroy.argtypes = []
    lib.NDIlib_destroy.restype = None

    lib.NDIlib_version.argtypes = []
    lib.NDIlib_version.restype = ctypes.c_char_p

    # Finder prototypes
    lib.NDIlib_find_create_v2.argtypes = [ctypes.POINTER(NDIlib_find_create_t)]
    lib.NDIlib_find_create_v2.restype = ctypes.c_void_p

    lib.NDIlib_find_destroy.argtypes = [ctypes.c_void_p]
    lib.NDIlib_find_destroy.restype = None

    lib.NDIlib_find_get_current_sources.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    lib.NDIlib_find_get_current_sources.restype = ctypes.POINTER(NDIlib_source_t)

    lib.NDIlib_find_wait_for_sources.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.NDIlib_find_wait_for_sources.restype = ctypes.c_bool

    # Receiver prototypes
    lib.NDIlib_recv_create_v3.argtypes = [ctypes.POINTER(NDIlib_recv_create_v3_t)]
    lib.NDIlib_recv_create_v3.restype = ctypes.c_void_p

    lib.NDIlib_recv_destroy.argtypes = [ctypes.c_void_p]
    lib.NDIlib_recv_destroy.restype = None

    lib.NDIlib_recv_connect.argtypes = [ctypes.c_void_p, ctypes.POINTER(NDIlib_source_t)]
    lib.NDIlib_recv_connect.restype = None

    lib.NDIlib_recv_capture_v2.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(NDIlib_video_frame_v2_t),
        ctypes.POINTER(NDIlib_audio_frame_v2_t),
        ctypes.POINTER(NDIlib_metadata_frame_t),
        ctypes.c_uint32,
    ]
    lib.NDIlib_recv_capture_v2.restype = ctypes.c_int

    lib.NDIlib_recv_free_video_v2.argtypes = [ctypes.c_void_p, ctypes.POINTER(NDIlib_video_frame_v2_t)]
    lib.NDIlib_recv_free_video_v2.restype = None

    lib.NDIlib_recv_free_audio_v2.argtypes = [ctypes.c_void_p, ctypes.POINTER(NDIlib_audio_frame_v2_t)]
    lib.NDIlib_recv_free_audio_v2.restype = None

    lib.NDIlib_recv_set_tally.argtypes = [ctypes.c_void_p, ctypes.POINTER(NDIlib_tally_t)]
    lib.NDIlib_recv_set_tally.restype = ctypes.c_bool

    # Sender prototypes
    lib.NDIlib_send_create.argtypes = [ctypes.POINTER(NDIlib_send_create_t)]
    lib.NDIlib_send_create.restype = ctypes.c_void_p

    lib.NDIlib_send_destroy.argtypes = [ctypes.c_void_p]
    lib.NDIlib_send_destroy.restype = None

    lib.NDIlib_send_send_video_v2.argtypes = [ctypes.c_void_p, ctypes.POINTER(NDIlib_video_frame_v2_t)]
    lib.NDIlib_send_send_video_v2.restype = None

    lib.NDIlib_send_send_video_async_v2.argtypes = [ctypes.c_void_p, ctypes.POINTER(NDIlib_video_frame_v2_t)]
    lib.NDIlib_send_send_video_async_v2.restype = None

    lib.NDIlib_send_send_audio_v2.argtypes = [ctypes.c_void_p, ctypes.POINTER(NDIlib_audio_frame_v2_t)]
    lib.NDIlib_send_send_audio_v2.restype = None

    lib.NDIlib_send_get_tally.argtypes = [ctypes.c_void_p, ctypes.POINTER(NDIlib_tally_t), ctypes.c_uint32]
    lib.NDIlib_send_get_tally.restype = ctypes.c_bool

    if not _IS_INITIALIZED:
        if not lib.NDIlib_initialize():
            raise RuntimeError("NDIlib_initialize failed to initialize NDI runtime.")
        _IS_INITIALIZED = True

    _NDI_LIB = lib
    return _NDI_LIB

def is_ndi_available() -> bool:
    try:
        get_ndi_lib()
        return True
    except Exception:
        return False
