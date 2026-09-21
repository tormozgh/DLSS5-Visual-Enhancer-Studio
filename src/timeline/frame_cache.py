"""High-speed frame cache and video reader pool for responsive timeline scrubbing."""

from __future__ import annotations

import collections
import os
import threading
from typing import OrderedDict

import cv2
import numpy as np


class VideoFrameCache:
    """Thread-safe LRU frame cache and VideoCapture instance pool."""

    def __init__(self, max_cache_frames: int = 180) -> None:
        self.max_cache_frames = max_cache_frames
        self._cache: OrderedDict[tuple[str, int], np.ndarray] = collections.OrderedDict()
        self._caps: dict[str, cv2.VideoCapture] = {}
        self._last_read_frame: dict[str, int] = {}
        self._lock = threading.Lock()

    def get_frame(self, file_path: str, frame_idx: int) -> np.ndarray | None:
        """Retrieve a specific frame (in BGR format) from cache or source file."""
        if not file_path or not os.path.exists(file_path):
            return None

        key = (file_path, frame_idx)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

            cap = self._caps.get(file_path)
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(file_path)
                if not cap.isOpened():
                    return None
                self._caps[file_path] = cap
                self._last_read_frame[file_path] = -1

            last_frame = self._last_read_frame.get(file_path, -1)
            # Optimize sequential reading: if target is the immediate next frame, avoid slow seek
            if frame_idx != last_frame + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            ret, frame = cap.read()
            if not ret or frame is None:
                return None

            self._last_read_frame[file_path] = frame_idx

            # Store in LRU cache
            self._cache[key] = frame
            if len(self._cache) > self.max_cache_frames:
                self._cache.popitem(last=False)

            return frame

    def generate_thumbnail(self, file_path: str, thumb_size: tuple[int, int] = (160, 90)) -> np.ndarray | None:
        """Extract a clean representative thumbnail frame from a media file."""
        frame = self.get_frame(file_path, 0)
        if frame is None:
            return None
        return cv2.resize(frame, thumb_size, interpolation=cv2.INTER_AREA)

    def get_media_metadata(self, file_path: str) -> tuple[int, float, float, int, int] | None:
        """Query duration_frames, duration_sec, fps, width, height for a file."""
        if not os.path.exists(file_path):
            return None
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0.0:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / fps if total_frames > 0 else 0.0
        cap.release()
        return total_frames, duration_sec, fps, width, height

    def clear(self) -> None:
        """Release all open video captures and flush memory cache."""
        with self._lock:
            for cap in self._caps.values():
                try:
                    cap.release()
                except Exception:
                    pass
            self._caps.clear()
            self._last_read_frame.clear()
            self._cache.clear()
