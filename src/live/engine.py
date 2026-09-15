"""Real-Time Rendering Master Pipeline Coordinator.

Glues together:
- NDI 6 Receiver (Network Ingest)
- NVIDIA Streamline 2.13 Engine (DLSS-NR + Optical Flow + DLSS-G Frame Gen)
- ReShade FX Engine (3D LUTs + Film Grain + ACES + CAS)
- NDI 6 Broadcast Sender ('DLSS 5 Visual Enhancer Studio' + Tally)
- Asynchronous NVENC Live Hardware Recorder
- Viewport callback emitters for PyQt6 Canvas
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from src.core.reshade import ReShadeEngine
from src.core.streamline import StreamlineHostEngine
from src.live.ndi import NdiReceiver, NdiSender, NdiSourceFinder
from src.live.recorder import LiveRecorder, RecorderTelemetry


@dataclass
class PipelineTelemetry:
    is_running: bool = False
    source_name: str = "None"
    input_resolution: tuple[int, int] = (0, 0)
    output_resolution: tuple[int, int] = (0, 0)
    input_fps: float = 0.0
    render_fps: float = 0.0
    broadcast_fps: float = 0.0
    latency_ms: float = 0.0
    frame_count: int = 0
    tally_program: bool = False
    tally_preview: bool = False
    recorder: RecorderTelemetry = field(default_factory=RecorderTelemetry)


class RealtimePipeline:
    """Master real-time rendering and broadcast pipeline."""

    def __init__(
        self,
        sender_name: str = "DLSS 5 Visual Enhancer Studio",
        on_frame_ready: Callable[[np.ndarray, np.ndarray], None] | None = None,
        on_telemetry: Callable[[PipelineTelemetry], None] | None = None,
    ) -> None:
        self.sender_name = sender_name
        self.on_frame_ready = on_frame_ready  # (original_rgba, enhanced_rgba)
        self.on_telemetry = on_telemetry

        self.streamline = StreamlineHostEngine()
        self.reshade = ReShadeEngine()
        self.recorder = LiveRecorder()
        self.finder = NdiSourceFinder()

        self._receiver: NdiReceiver | None = None
        self._sender: NdiSender | None = None
        self._lock = threading.Lock()
        self._is_running = False

        self._fps_tracker_time = time.perf_counter()
        self._fps_frame_count = 0
        self._render_fps = 0.0
        self._last_latency = 0.0
        self._total_frames = 0
        self._current_source_name = "None"
        self._current_url: str | None = None

        # Start background finder so available streams are ready in UI
        self.finder.start()

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._is_running

    def get_sources(self) -> list[tuple[str, str]]:
        """Return discovered NDI sources on local network."""
        return self.finder.get_sources()

    def start_pipeline(
        self,
        source_name: str,
        url_address: str | None = None,
        enable_ndi_out: bool = True,
    ) -> None:
        """Start ingesting from NDI source and executing real-time pipeline."""
        with self._lock:
            if self._is_running:
                self.stop_pipeline()

            self._current_source_name = source_name
            self._current_url = url_address
            self._is_running = True
            self._fps_tracker_time = time.perf_counter()
            self._fps_frame_count = 0
            self._total_frames = 0

            # 1. Setup NDI Sender if broadcasting enabled
            if enable_ndi_out:
                self._sender = NdiSender(self.sender_name)
                self._sender.start()
            else:
                self._sender = None

            # 2. Setup NDI Receiver
            self._receiver = NdiReceiver(
                source_name=source_name,
                url_address=url_address,
                on_video_frame=self._on_incoming_video_frame,
            )
            self._receiver.start()

    def stop_pipeline(self) -> None:
        """Stop receiver, sender, and recording."""
        with self._lock:
            self._is_running = False

            if self._receiver:
                try:
                    self._receiver.stop()
                except Exception:
                    pass
                self._receiver = None

            if self._sender:
                try:
                    self._sender.stop()
                except Exception:
                    pass
                self._sender = None

            if self.recorder.is_recording:
                try:
                    self.recorder.stop()
                except Exception:
                    pass

    def start_recording(
        self,
        bitrate_mbps: int = 25,
        format_ext: str = "mp4",
        target_resolution: tuple[int, int] = (0, 0),
        output_dir: Path | None = None,
        filename_prefix: str = "DLSS5_Live",
    ) -> Path | None:
        """Start hardware live recording with custom parameters."""
        with self._lock:
            if not self._is_running or not self._receiver:
                return None
            w, h = target_resolution
            if w <= 0 or h <= 0:
                w, h = self._receiver.resolution
            fps = self._receiver.fps or 60.0
            if self.streamline.config.enable_frame_gen:
                fps *= 2.0
            return self.recorder.start(
                width=w,
                height=h,
                fps=fps,
                bitrate_mbps=bitrate_mbps,
                format_ext=format_ext,
                output_dir=output_dir,
                filename_prefix=filename_prefix,
                target_resolution=target_resolution,
            )

    def stop_recording(self) -> Path | None:
        """Stop hardware live recording."""
        return self.recorder.stop()

    def _on_incoming_video_frame(
        self, original_rgba: np.ndarray, timestamp: int, fps: float
    ) -> None:
        """Main real-time processing loop triggered on each NDI video frame."""
        if not self._is_running:
            return

        t_start = time.perf_counter()
        h, w = original_rgba.shape[:2]

        # 1. NVIDIA Streamline 2.13 (DLSS-NR & Frame Generation)
        enhanced_frames = self.streamline.process_frame(original_rgba, fps)

        # 2. ReShade FX Post-Processing Pass
        final_frames: list[np.ndarray] = []
        for frame in enhanced_frames:
            shaded = self.reshade.process_frame(frame)
            final_frames.append(shaded)

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        # 3. Output to NDI Broadcast Sender
        sender = self._sender
        out_fps = fps * (2.0 if self.streamline.config.enable_frame_gen else 1.0)
        if sender and sender.is_broadcasting:
            for frame in final_frames:
                sender.send_video_frame(frame, fps=out_fps)

        # 4. Output to Live NVENC Recorder
        if self.recorder.is_recording:
            for frame in final_frames:
                self.recorder.write_frame(frame)

        # 5. Telemetry calculation
        self._fps_frame_count += len(final_frames)
        self._total_frames += len(final_frames)
        now = time.perf_counter()
        dt = now - self._fps_tracker_time
        if dt >= 0.5:
            self._render_fps = self._fps_frame_count / dt
            self._fps_frame_count = 0
            self._fps_tracker_time = now
            self._last_latency = latency_ms

            if self.on_telemetry:
                prog, prev = (sender.get_tally() if sender else (False, False))
                telem = PipelineTelemetry(
                    is_running=True,
                    source_name=self._current_source_name,
                    input_resolution=(w, h),
                    output_resolution=(w, h),
                    input_fps=fps,
                    render_fps=self._render_fps,
                    broadcast_fps=out_fps if sender else 0.0,
                    latency_ms=self._last_latency,
                    frame_count=self._total_frames,
                    tally_program=prog,
                    tally_preview=prev,
                    recorder=self.recorder.get_telemetry(),
                )
                try:
                    self.on_telemetry(telem)
                except Exception:
                    pass

        # 6. Emit to UI Canvas Viewport
        if self.on_frame_ready and len(final_frames) > 0:
            try:
                self.on_frame_ready(original_rgba, final_frames[-1])
            except Exception:
                pass

    def close(self) -> None:
        """Full cleanup."""
        self.stop_pipeline()
        self.finder.stop()
