"""Background export worker rendering timeline frames and encoding with FFmpeg / NVENC."""

from __future__ import annotations

import os
import shutil
import subprocess
import time

from PyQt6.QtCore import QThread, pyqtSignal

from .compositor import TimelineCompositor


class TimelineExportWorker(QThread):
    """Background rendering worker for exporting the entire timeline or In/Out range."""

    progressChanged = pyqtSignal(int, str)  # percent, status message
    statsUpdated = pyqtSignal(float, float, str)  # fps, elapsed, eta
    finishedExport = pyqtSignal(str)  # output_path
    failedExport = pyqtSignal(str)  # error_message

    def __init__(
        self,
        compositor: TimelineCompositor,
        start_frame: int,
        end_frame: int,
        output_path: str,
        target_resolution: tuple[int, int] = (1920, 1080),
        target_fps: float = 30.0,
        codec: str = "h264_nvenc",
        bitrate_mbps: float = 24.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.compositor = compositor
        self.start_frame = max(0, start_frame)
        self.end_frame = max(self.start_frame + 1, end_frame)
        self.output_path = output_path
        self.target_resolution = target_resolution
        self.target_fps = target_fps
        self.codec = codec
        self.bitrate_mbps = bitrate_mbps

        self._cancelled = False

    def cancel(self) -> None:
        """Signal thread to cancel export immediately."""
        self._cancelled = True

    def run(self) -> None:
        width, height = self.target_resolution
        total_frames = self.end_frame - self.start_frame
        if total_frames <= 0:
            self.failedExport.emit("Invalid frame range specified.")
            return

        ffmpeg_exe = shutil.which("ffmpeg") or "ffmpeg"

        # Determine video encoder and fallback
        vcodec = self.codec
        # Test if nvenc requested
        extra_args = []
        if "nvenc" in vcodec:
            extra_args = ["-preset", "p6", "-tune", "hq", "-rc", "vbr", "-b:v", f"{self.bitrate_mbps}M"]
        elif "prores" in vcodec:
            vcodec = "prores_ks"
            extra_args = ["-profile:v", "3", "-vendor", "apl0", "-bits_per_mb", "8000"]
        else:
            vcodec = "libx264"
            extra_args = ["-preset", "medium", "-crf", "18", "-b:v", f"{self.bitrate_mbps}M"]

        temp_video = self.output_path + ".temp_video.mp4"
        cmd = [
            ffmpeg_exe,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "bgr24",
            "-r", str(self.target_fps),
            "-i", "-",
            "-c:v", vcodec,
            *extra_args,
            "-pix_fmt", "yuv420p",
            temp_video,
        ]

        proc = None
        try:
            self.progressChanged.emit(0, "Initializing video encoding pipeline...")
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**7,
            )

            start_time = time.perf_counter()
            last_stats_time = start_time
            frames_rendered = 0

            for frame_idx in range(self.start_frame, self.end_frame):
                if self._cancelled:
                    if proc and proc.stdin:
                        proc.stdin.close()
                    proc.kill()
                    if os.path.exists(temp_video):
                        os.remove(temp_video)
                    self.failedExport.emit("Export cancelled by user.")
                    return

                # Render composited frame through DLSS 5 pipeline
                frame = self.compositor.render_frame(
                    frame_idx=frame_idx,
                    target_size=self.target_resolution,
                    is_export=True,
                )

                if frame is not None and frame.size > 0:
                    try:
                        proc.stdin.write(frame.tobytes())
                    except Exception as exc:
                        raise RuntimeError(f"Error writing to FFmpeg encoder: {exc}")

                frames_rendered += 1
                now = time.perf_counter()

                # Update progress every ~0.25s
                if now - last_stats_time >= 0.25 or frames_rendered == total_frames:
                    elapsed = now - start_time
                    fps = frames_rendered / max(0.001, elapsed)
                    remaining_frames = total_frames - frames_rendered
                    eta_sec = remaining_frames / max(0.001, fps)
                    eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_sec))

                    pct = int((frames_rendered / total_frames) * 100)
                    status = f"Rendering Frame {frames_rendered}/{total_frames} ({fps:.1f} FPS)"
                    self.progressChanged.emit(pct, status)
                    self.statsUpdated.emit(fps, elapsed, eta_str)
                    last_stats_time = now

            if proc and proc.stdin:
                proc.stdin.close()
            proc.wait()

            # Move temp video to final output path or mux audio if present
            if os.path.exists(self.output_path):
                try:
                    os.remove(self.output_path)
                except Exception:
                    pass
            shutil.move(temp_video, self.output_path)

            self.progressChanged.emit(100, "Export completed successfully!")
            self.finishedExport.emit(self.output_path)

        except Exception as exc:
            if proc:
                try:
                    proc.kill()
                except Exception:
                    pass
            if os.path.exists(temp_video):
                try:
                    os.remove(temp_video)
                except Exception:
                    pass
            self.failedExport.emit(f"Export failed: {exc}")
