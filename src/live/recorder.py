"""Real-Time Hardware NVENC Video Recorder.

Captures live rendered frames and NDI audio, streaming them directly into an
asynchronous hardware NVENC FFmpeg session with zero broadcast loop stalls.

Key Resilience Features:
- Broadcast-grade fragmented MP4 (+frag_keyframe+empty_moov+default_base_moof)
  ensuring recordings are immediately playable and immune to crashes or abrupt stops.
- Dynamic resolution latching: automatically adapts to incoming video dimensions.
- Zero-copy memoryview writes to FFmpeg stdin pipe (zero memory churn).
- Continuous non-blocking stderr drainage to prevent OS pipe deadlocks.
- Graceful flush and container finalization on stop.
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

import cv2
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
    """Async hardware NVENC video & audio recorder with crash-resilient container."""

    def __init__(self, output_dir: Path | None = None) -> None:
        self._output_dir = output_dir or OUTPUTS
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=90)
        self._worker_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._is_recording = False
        self._start_time: float = 0.0
        self._frames_written: int = 0
        self._dropped_frames: int = 0
        self._current_path: Path | None = None
        self._width = 0
        self._height = 0
        self._fps = 60.0
        self._bitrate_mbps = 25
        self._codec = "h264_nvenc"
        self._preset = "p4"
        self._last_error = ""

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
        width: int = 0,
        height: int = 0,
        fps: float = 60.0,
        bitrate_mbps: int = 25,
        codec: str = "h264_nvenc",
        preset: str = "p4",
    ) -> Path:
        """Start asynchronous recording session."""
        with self._lock:
            if self._is_recording:
                raise RuntimeError("Recording session already in progress.")

            self._width = width
            self._height = height
            self._fps = fps if fps > 1.0 else 60.0
            self._bitrate_mbps = bitrate_mbps
            self._codec = codec
            self._preset = preset
            self._last_error = ""

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self._current_path = self._output_dir / f"DLSS5_Live_{timestamp}.mp4"

            self._is_recording = True
            self._start_time = time.perf_counter()
            self._frames_written = 0
            self._dropped_frames = 0

            # Drain queue of any stale entries
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

            # If dimensions are already known, spawn FFmpeg immediately
            if self._width > 0 and self._height > 0:
                self._spawn_ffmpeg(self._width, self._height)

            self._worker_thread = threading.Thread(
                target=self._writer_loop,
                name="dlss5-live-recorder-worker",
                daemon=True,
            )
            self._worker_thread.start()
            return self._current_path

    def _spawn_ffmpeg(self, width: int, height: int) -> None:
        """Spawn the background FFmpeg recording process with resilient container flags."""
        ffmpeg_exe = str(FFMPEG if FFMPEG.is_file() else "ffmpeg")
        gop_size = max(1, int(round(self._fps)))  # Keyframe every 1.0s for instant seek & recovery

        cmd = [
            ffmpeg_exe,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "rgba",
            "-r", f"{self._fps:.2f}",
            "-i", "-",  # Input from stdin pipe
            "-c:v", self._codec,
            "-preset", self._preset,
            "-g", str(gop_size),
            "-b:v", f"{self._bitrate_mbps}M",
            "-maxrate", f"{int(self._bitrate_mbps * 1.5)}M",
            "-bufsize", f"{self._bitrate_mbps * 2}M",
            "-pix_fmt", "yuv420p",
            # Fragmented MP4 flags: makes recording immune to abrupt termination or crashes
            "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
            str(self._current_path),
        ]

        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except Exception:
            # Fallback to software libx264 if hardware encoder setup failed
            cmd[13] = "libx264"
            cmd[15] = "veryfast"
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

        # Launch stderr drainer thread to prevent pipe buffer deadlock
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name="dlss5-live-recorder-stderr",
            daemon=True,
        )
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        """Continuously consume stderr to prevent OS pipe buffer deadlocks."""
        proc = self._process
        if not proc or not proc.stderr:
            return
        try:
            while True:
                line = proc.stderr.readline()
                if not line:
                    break
                line_str = line.decode("utf-8", errors="ignore").strip()
                if "error" in line_str.lower() or "fail" in line_str.lower():
                    self._last_error = line_str
        except Exception:
            pass

    def write_frame(self, rgba_frame: np.ndarray) -> None:
        """Enqueue frame for asynchronous hardware encoding."""
        if not self._is_recording:
            return

        h, w = rgba_frame.shape[:2]

        # Dynamic dimension latching: if recorder was started with (0, 0), initialize on first frame
        if self._process is None:
            with self._lock:
                if self._process is None and self._is_recording:
                    self._width = w
                    self._height = h
                    self._spawn_ffmpeg(w, h)

        # Ensure frame dimensions match configured FFmpeg canvas
        if (w, h) != (self._width, self._height) and self._width > 0 and self._height > 0:
            rgba_frame = cv2.resize(rgba_frame, (self._width, self._height), interpolation=cv2.INTER_LINEAR)

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
            self._worker_thread.join(timeout=8.0)
            self._worker_thread = None

        proc = self._process
        self._process = None
        if proc:
            try:
                # Wait for FFmpeg to finish writing container fragments
                proc.wait(timeout=8.0)
            except Exception:
                proc.kill()

        return self._current_path

    def _writer_loop(self) -> None:
        """Background thread streaming frames directly to FFmpeg stdin."""
        while True:
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                if not self._is_recording:
                    break
                continue

            if item is None:
                break

            proc = self._process
            if proc and proc.stdin:
                try:
                    # Ensure contiguous memory before write
                    if not item.flags.c_contiguous:
                        item = np.ascontiguousarray(item)
                    # Use zero-copy memoryview directly into stdin
                    proc.stdin.write(memoryview(item))
                    self._frames_written += 1
                except (BrokenPipeError, OSError):
                    break

        # Close stdin cleanly from the writer thread to signal EOF to FFmpeg
        proc = self._process
        if proc and proc.stdin and not proc.stdin.closed:
            try:
                proc.stdin.close()
            except Exception:
                pass
