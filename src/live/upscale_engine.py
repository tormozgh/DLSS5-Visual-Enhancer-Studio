"""Dedicated Real-Time Upscale Pipeline Coordinator.

Decoupled execution engine dedicated strictly to spatial edge upscaling and directional
sharpening (NVIDIA NIS / CAS / Catmull-Rom Bicubic). 

Architecture Isolation:
- Strictly independent from StreamlineHostEngine and ReShadeEngine.
- Ingests RAW NDI, DirectShow Webcams/Capture Cards, or In-Memory Internal Render feed.
- Non-blocking atomic frame handoff for Internal feed (zero memory accumulation, zero drop cascade).
- Full hardware NVENC recording locked to upscaled target resolution (e.g. 4K UHD).
- Low-latency NDI 6 Broadcast sender.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from src.live.camera import CameraDeviceInfo, WebcamReceiver
from src.live.engine import PipelineTelemetry
from src.live.ndi import NdiReceiver, NdiSender, NdiSourceFinder
from src.live.recorder import LiveRecorder, RecorderTelemetry
from src.live.upscaler import RealtimeNISUpscaler


class RealtimeUpscalePipeline:
    """Dedicated low-latency pipeline coordinator for Real-Time Upscale Studio."""

    def __init__(
        self,
        sender_name: str = "DLSS 5 Real-Time Upscale Studio",
        on_frame_ready: Callable[[np.ndarray, np.ndarray], None] | None = None,
        on_telemetry: Callable[[PipelineTelemetry], None] | None = None,
    ) -> None:
        self.sender_name = sender_name
        self.on_frame_ready = on_frame_ready  # (original_rgba, upscaled_rgba)
        self.on_telemetry = on_telemetry

        self.upscaler = RealtimeNISUpscaler()
        self.recorder = LiveRecorder()
        self.finder = NdiSourceFinder()

        self._receiver: NdiReceiver | None = None
        self._camera_receiver: WebcamReceiver | None = None
        self._sender: NdiSender | None = None

        self._lock = threading.Lock()
        self._is_running = False
        self._source_type = "internal"  # 'internal', 'ndi', 'webcam'
        self._current_source_name = "None"
        self._current_url: str | None = None

        # Internal feed thread-safe atomic buffer
        self._internal_lock = threading.Lock()
        self._pending_internal_frame: tuple[np.ndarray, np.ndarray] | None = None
        self._internal_event = threading.Event()
        self._internal_thread: threading.Thread | None = None

        # Telemetry and state tracking
        self._last_raw_frame: np.ndarray | None = None
        self._last_upscaled_frame: np.ndarray | None = None
        self._last_fps: float = 60.0
        self._last_timestamp: int = 0
        self._last_dimensions: tuple[int, int] = (1920, 1080)

        self._fps_tracker_time = time.perf_counter()
        self._fps_frame_count = 0
        self._render_fps = 0.0
        self._last_latency = 0.0
        self._total_frames = 0

        # Start NDI discovery in background
        self.finder.start()

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._is_running

    def get_sources(self) -> list[tuple[str, str]]:
        """Return discovered NDI sources on local network."""
        return self.finder.get_sources()

    def get_cameras(self) -> list[CameraDeviceInfo]:
        """Return discovered video capture hardware devices."""
        return WebcamReceiver.list_cameras()

    def start_ndi_pipeline(
        self,
        source_name: str,
        url_address: str | None = None,
        enable_ndi_out: bool = True,
    ) -> None:
        """Start ingesting from RAW NDI stream with isolated upscaling."""
        with self._lock:
            if self._is_running:
                self.stop_pipeline()

            self._current_source_name = source_name
            self._current_url = url_address
            self._source_type = "ndi"
            self._is_running = True
            self._fps_tracker_time = time.perf_counter()
            self._fps_frame_count = 0
            self._total_frames = 0

            if enable_ndi_out:
                self._sender = NdiSender(self.sender_name)
                self._sender.start()
            else:
                self._sender = None

            self._receiver = NdiReceiver(
                source_name=source_name,
                url_address=url_address,
                on_video_frame=self._on_incoming_video_frame,
            )
            self._receiver.start()

    def start_webcam_pipeline(
        self,
        device_index: int = 0,
        width: int = 1920,
        height: int = 1080,
        fps: float = 60.0,
        enable_ndi_out: bool = True,
    ) -> None:
        """Start capturing from DirectShow webcam or capture card."""
        with self._lock:
            if self._is_running:
                self.stop_pipeline()

            # Find friendly camera name
            cams = WebcamReceiver.list_cameras()
            cam_name = f"Camera {device_index}"
            for c in cams:
                if c.index == device_index:
                    cam_name = c.name
                    break

            self._current_source_name = cam_name
            self._source_type = "webcam"
            self._is_running = True
            self._fps_tracker_time = time.perf_counter()
            self._fps_frame_count = 0
            self._total_frames = 0

            if enable_ndi_out:
                self._sender = NdiSender(self.sender_name)
                self._sender.start()
            else:
                self._sender = None

            self._camera_receiver = WebcamReceiver(
                device_index=device_index,
                target_width=width,
                target_height=height,
                target_fps=fps,
                on_video_frame=self._on_incoming_video_frame,
            )
            self._camera_receiver.start()

    def start_internal_pipeline(
        self,
        source_name: str = "Real-Time Rendering Feed",
        enable_ndi_out: bool = True,
    ) -> None:
        """Start non-blocking internal in-memory feed ingestion."""
        with self._lock:
            if self._is_running:
                self.stop_pipeline()

            self._current_source_name = source_name
            self._source_type = "internal"
            self._is_running = True
            self._fps_tracker_time = time.perf_counter()
            self._fps_frame_count = 0
            self._total_frames = 0

            if enable_ndi_out:
                self._sender = NdiSender(self.sender_name)
                self._sender.start()
            else:
                self._sender = None

            self._internal_event.clear()
            self._internal_thread = threading.Thread(
                target=self._internal_worker_loop,
                name="dlss5-internal-feed-worker",
                daemon=True,
            )
            self._internal_thread.start()

    def feed_internal_frame(self, original_rgba: np.ndarray, enhanced_rgba: np.ndarray) -> None:
        """Atomic non-blocking frame deposit from Real-Time Rendering tab.
        
        Zero copy overhead and zero queue growth: replaces any waiting unconsumed
        frame in O(1) time without blocking the caller thread.
        """
        if not self._is_running or self._source_type != "internal":
            return

        with self._internal_lock:
            self._pending_internal_frame = (original_rgba, enhanced_rgba)
        self._internal_event.set()

    def _internal_worker_loop(self) -> None:
        """Dedicated consumer thread for internal in-memory frames."""
        while self._is_running and self._source_type == "internal":
            signaled = self._internal_event.wait(timeout=0.033)
            if not self._is_running or self._source_type != "internal":
                break

            pair = None
            with self._internal_lock:
                if self._pending_internal_frame is not None:
                    pair = self._pending_internal_frame
                    self._pending_internal_frame = None
                self._internal_event.clear()

            if pair is not None:
                orig, enh = pair
                now = time.perf_counter()
                ts = int(now * 1000)
                # Process the rendered frame through upscaling
                self._process_frame_core(enh, orig, ts, 60.0)

    def stop_pipeline(self) -> None:
        """Stop all receivers, workers, senders, and active recordings."""
        with self._lock:
            self._is_running = False

            if self._internal_thread and self._internal_thread.is_alive():
                self._internal_event.set()
                self._internal_thread.join(timeout=0.5)
                self._internal_thread = None

            with self._internal_lock:
                self._pending_internal_frame = None

            if self._receiver:
                try:
                    self._receiver.stop()
                except Exception:
                    pass
                self._receiver = None

            if self._camera_receiver:
                try:
                    self._camera_receiver.stop()
                except Exception:
                    pass
                self._camera_receiver = None

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

            try:
                self.upscaler.close()
            except Exception:
                pass

    def start_recording(
        self,
        bitrate_mbps: int = 25,
        format_ext: str = "mp4",
        output_dir: Path | None = None,
        filename_prefix: str = "DLSS5_Upscale",
    ) -> Path | None:
        """Start hardware NVENC recording locked to the upscaled resolution."""
        with self._lock:
            if not self._is_running:
                return None

            in_w, in_h = self._last_dimensions
            target_w, target_h = self.upscaler.calculate_target_dimensions(in_w, in_h)
            fps = self._last_fps if self._last_fps > 10.0 else 60.0

            return self.recorder.start(
                width=target_w,
                height=target_h,
                fps=fps,
                bitrate_mbps=bitrate_mbps,
                format_ext=format_ext,
                output_dir=output_dir,
                filename_prefix=filename_prefix,
                target_resolution=(target_w, target_h),
            )

    def stop_recording(self) -> Path | None:
        """Stop live recording and finalize file."""
        return self.recorder.stop()

    def _on_incoming_video_frame(self, raw_rgba: np.ndarray, timestamp: int, fps: float) -> None:
        """Callback from NdiReceiver or WebcamReceiver."""
        if not self._is_running:
            return
        self._process_frame_core(raw_rgba, raw_rgba, timestamp, fps)

    def _process_frame_core(
        self,
        process_rgba: np.ndarray,
        comparison_original_rgba: np.ndarray,
        timestamp: int,
        fps: float,
    ) -> None:
        """High-performance core upscaling loop."""
        in_h, in_w = process_rgba.shape[:2]
        self._last_dimensions = (in_w, in_h)

        with self._lock:
            self._last_raw_frame = process_rgba
            self._last_fps = fps
            self._last_timestamp = timestamp

        t_start = time.perf_counter()

        # Real-Time Spatial Upscale & Directional Sharpening (NIS / CAS / Bicubic)
        upscaled = self.upscaler.process(process_rgba)

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        with self._lock:
            self._last_upscaled_frame = upscaled

        # 1. Output to NDI Broadcast Sender
        sender = self._sender
        if sender and sender.is_broadcasting:
            sender.send_video_frame(upscaled, fps=fps)

        # 2. Output to Hardware NVENC Live Recorder at UPSCALED Resolution
        if self.recorder.is_recording:
            self.recorder.write_frame(upscaled)

        # 3. Telemetry Tracking
        self._fps_frame_count += 1
        self._total_frames += 1
        now = time.perf_counter()
        dt = now - self._fps_tracker_time
        if dt >= 0.5:
            self._render_fps = self._fps_frame_count / dt
            self._fps_frame_count = 0
            self._fps_tracker_time = now
            self._last_latency = latency_ms

            if self.on_telemetry:
                prog, prev = (sender.get_tally() if sender else (False, False))
                out_h, out_w = upscaled.shape[:2]
                telem = PipelineTelemetry(
                    is_running=True,
                    source_name=self._current_source_name,
                    input_resolution=(in_w, in_h),
                    output_resolution=(out_w, out_h),
                    input_fps=fps,
                    render_fps=self._render_fps,
                    broadcast_fps=fps if sender else 0.0,
                    latency_ms=self._last_latency,
                    frame_count=self._total_frames,
                    tally_program=prog,
                    tally_preview=prev,
                    recorder=self.recorder.get_telemetry(),
                )
                self.on_telemetry(telem)

        # 4. Viewport emit for SplitCanvas (original vs upscaled)
        if self.on_frame_ready:
            self.on_frame_ready(comparison_original_rgba, upscaled)

    def reprocess_last_frame(self) -> None:
        """Reprocess current frame when paused or changing parameters."""
        raw = self._last_raw_frame
        if raw is not None:
            self._process_frame_core(raw, raw, self._last_timestamp, self._last_fps)

    def close(self) -> None:
        """Cleanly terminate pipeline and background resources."""
        self.stop_pipeline()
        self.finder.stop()
        try:
            self.upscaler.close()
        except Exception:
            pass
