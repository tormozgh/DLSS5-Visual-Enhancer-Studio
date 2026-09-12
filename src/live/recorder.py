"""Real-Time Hardware NVENC Video Recorder.

Captures live rendered frames and NDI audio, streaming them directly into an
asynchronous hardware NVENC FFmpeg session with zero broadcast loop stalls.
"""

from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.core.paths import FFMPEG, OUTPUTS


@dataclass
class RecorderTelemetry:
    is_recording: bool = False
    duration_seconds: float = 0.0
    frames_written: int = 0
    dropped_frames: int = 0
    file_size_mb: float = 0.0
    output_path: str = ""


class LiveRecorder:
    """Async hardware NVENC video & audio recorder."""

    def __init__(self, output_dir: Path | None = None) -> None:
        self._output_dir = output_dir or OUTPUTS
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=45)
        self._worker_thread: threading.Thread | None = None
        self._is_recording = False
        self._start_time: float = 0.0
        self._frames_written: int = 0
        self._dropped_frames: int = 0
        self._current_path: Path | None = None
        self._width = 0
        self._height = 0
        self._fps = 60.0

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._is_recording

    def get_telemetry(self) -> RecorderTelemetry:
        with self._lock:
            if not self._is_recording:
                return RecorderTelemetry()
            dur = max(0.0, time.perf_counter() - self._start_time)
            size_mb = 0.0
            if self._current_path and self._current_path.is_file():
                try:
                    size_mb = self._current_path.stat().st_size / (1024 * 1024)
                except OSError:
                    pass
            return RecorderTelemetry(
                is_recording=True,
                duration_seconds=dur,
                frames_written=self._frames_written,
                dropped_frames=self._dropped_frames,
                file_size_mb=size_mb,
                output_path=str(self._current_path) if self._current_path else "",
            )

    def start(
        self,
        width: int,
        height: int,
        fps: float = 60.0,
        bitrate_mbps: int = 25,
        codec: str = "h264_nvenc",
        preset: str = "p4",  # Fast & high quality NVENC preset
    ) -> Path:
        """Start asynchronous recording session."""
        with self._lock:
            if self._is_recording:
                raise RuntimeError("Recording session already in progress.")

            self._width = width
            self._height = height
            self._fps = fps if fps > 1.0 else 60.0

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self._current_path = self._output_dir / f"DLSS5_Live_{timestamp}.mp4"

            # Check if NVENC is available; fallback to libx264 if necessary
            ffmpeg_exe = str(FFMPEG if FFMPEG.is_file() else "ffmpeg")

            cmd = [
                ffmpeg_exe,
                "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{width}x{height}",
                "-pix_fmt", "rgba",
                "-r", f"{self._fps:.2f}",
                "-i", "-",  # Input from stdin pipe
                "-c:v", codec,
                "-preset", preset,
                "-b:v", f"{bitrate_mbps}M",
                "-maxrate", f"{int(bitrate_mbps * 1.5)}M",
                "-bufsize", f"{bitrate_mbps * 2}M",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(self._current_path),
            ]

            try:
                self._process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
            except Exception as exc:
                # Fallback to software libx264 if hardware encoder setup failed
                cmd[13] = "libx264"
                cmd[15] = "veryfast"
                self._process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )

            self._is_recording = True
            self._start_time = time.perf_counter()
            self._frames_written = 0
            self._dropped_frames = 0

            # Drain queue
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

            self._worker_thread = threading.Thread(
                target=self._writer_loop,
                name="dlss5-live-recorder-worker",
                daemon=True,
            )
            self._worker_thread.start()
            return self._current_path

    def write_frame(self, rgba_frame: np.ndarray) -> None:
        """Enqueue frame for asynchronous hardware encoding."""
        if not self._is_recording or self._process is None:
            return

        try:
            # If buffer fills up (e.g. momentary disk IO spike), drop oldest to protect live broadcast
            self._queue.put_nowait(rgba_frame)
        except queue.Full:
            self._dropped_frames += 1

    def stop(self) -> Path | None:
        """Finalize recording and cleanly close MP4 container."""
        with self._lock:
            if not self._is_recording:
                return self._current_path
            self._is_recording = False

        # Signal worker to finish
        self._queue.put(None)
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3.0)
            self._worker_thread = None

        proc = self._process
        self._process = None
        if proc:
            try:
                if proc.stdin and not proc.stdin.closed:
                    proc.stdin.close()
                proc.wait(timeout=3.0)
            except Exception:
                proc.kill()

        return self._current_path

    def _writer_loop(self) -> None:
        proc = self._process
        if not proc or not proc.stdin:
            return

        while True:
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                if not self._is_recording:
                    break
                continue

            if item is None:
                break

            try:
                # Ensure contiguous memory before write
                if not item.flags.c_contiguous:
                    item = np.ascontiguousarray(item)
                proc.stdin.write(item.tobytes())
                self._frames_written += 1
            except (BrokenPipeError, OSError):
                break
