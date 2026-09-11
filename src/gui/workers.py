"""Background task workers for rendering video Neural Rendering, Upscale, and Frame Interpolation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QThread, pyqtSignal

from ..core.jobs import Cancelled, JobController
from ..core.paths import OUTPUTS
from ..frame_interpolation.batch import interpolate_videos
from ..frame_interpolation.models import FrameInterpolationOptions
from ..neural_rendering.video.models import ConversionOptions
from ..neural_rendering.video.processor import convert_video
from ..upscale.image.batch import upscale_images
from ..upscale.image.models import ImageUpscaleOptions
from ..upscale.video.batch import upscale_videos
from ..upscale.video.models import UpscaleOptions


class VideoRenderWorker(QThread):
    """Executes Neural Rendering on video files in background."""

    progressChanged = pyqtSignal(float, str)  # 0.0 - 1.0, message
    finished = pyqtSignal(str, float)         # output_path, elapsed_seconds
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        source_path: Path,
        options: ConversionOptions,
        output_dir: Path = OUTPUTS,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self.source_path = source_path
        self.options = options
        self.output_dir = output_dir
        self.controller = JobController()

    def cancel(self) -> None:
        self.controller.stop()

    def run(self) -> None:
        def on_progress(fraction: float, message: str) -> None:
            self.progressChanged.emit(max(0.0, min(1.0, fraction)), message)

        try:
            res = convert_video(
                self.source_path,
                self.options,
                progress=on_progress,
                output_dir=self.output_dir,
                controller=self.controller,
            )
            self.finished.emit(str(res.output_path), float(res.elapsed_seconds))
        except Cancelled:
            self.cancelled.emit()
        except Exception as exc:
            if self.controller.cancel.is_set():
                self.cancelled.emit()
            else:
                self.failed.emit(str(exc))


class UpscaleWorker(QThread):
    """Executes RTX Video Super Resolution (VSR) and HDR on images or videos."""

    progressChanged = pyqtSignal(float, str)
    finished = pyqtSignal(str, float)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        source_path: Path,
        is_video: bool,
        options: ImageUpscaleOptions | UpscaleOptions,
        output_dir: Path = OUTPUTS,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self.source_path = source_path
        self.is_video = is_video
        self.options = options
        self.output_dir = output_dir
        self.controller = JobController()

    def cancel(self) -> None:
        self.controller.stop()

    def run(self) -> None:
        started = time.perf_counter()

        def on_progress(fraction: float, message: str) -> None:
            self.progressChanged.emit(max(0.0, min(1.0, fraction)), message)

        try:
            if self.is_video:
                batch_res = upscale_videos(
                    [self.source_path],
                    self.options,  # type: ignore
                    progress=on_progress,
                    output_dir=self.output_dir,
                    controller=self.controller,
                )
                elapsed = time.perf_counter() - started
                if batch_res.cancelled or self.controller.cancel.is_set():
                    self.cancelled.emit()
                    return
                if batch_res.failures:
                    self.failed.emit(batch_res.failures[0].error)
                    return
                if batch_res.successes:
                    out_path = str(batch_res.successes[0].result.output_path)
                    self.finished.emit(out_path, elapsed)
                else:
                    self.failed.emit("No output file generated.")
            else:
                batch_res = upscale_images(
                    [self.source_path],
                    self.options,  # type: ignore
                    progress=on_progress,
                    output_dir=self.output_dir,
                    controller=self.controller,
                )
                elapsed = time.perf_counter() - started
                if batch_res.cancelled or self.controller.cancel.is_set():
                    self.cancelled.emit()
                    return
                if batch_res.failures:
                    self.failed.emit(batch_res.failures[0].error)
                    return
                if batch_res.successes:
                    out_path = str(batch_res.successes[0].output_path)
                    self.finished.emit(out_path, elapsed)
                else:
                    self.failed.emit("No output file generated.")
        except Cancelled:
            self.cancelled.emit()
        except Exception as exc:
            if self.controller.cancel.is_set():
                self.cancelled.emit()
            else:
                self.failed.emit(str(exc))


class InterpolationWorker(QThread):
    """Executes DLSS Frame Generation (DLSSG) multi-frame video interpolation."""

    progressChanged = pyqtSignal(float, str)
    finished = pyqtSignal(str, float)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        source_path: Path,
        options: FrameInterpolationOptions,
        output_dir: Path = OUTPUTS,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self.source_path = source_path
        self.options = options
        self.output_dir = output_dir
        self.controller = JobController()

    def cancel(self) -> None:
        self.controller.stop()

    def run(self) -> None:
        started = time.perf_counter()

        def on_progress(fraction: float, message: str) -> None:
            self.progressChanged.emit(max(0.0, min(1.0, fraction)), message)

        try:
            batch_res = interpolate_videos(
                [self.source_path],
                self.options,
                progress=on_progress,
                output_dir=self.output_dir,
                controller=self.controller,
            )
            elapsed = time.perf_counter() - started
            if batch_res.cancelled or self.controller.cancel.is_set():
                self.cancelled.emit()
                return
            if batch_res.failures:
                self.failed.emit(batch_res.failures[0].error)
                return
            if batch_res.successes:
                out_path = str(batch_res.successes[0].result.output_path)
                self.finished.emit(out_path, elapsed)
            else:
                self.failed.emit("No output file generated.")
        except Cancelled:
            self.cancelled.emit()
        except Exception as exc:
            if self.controller.cancel.is_set():
                self.cancelled.emit()
            else:
                self.failed.emit(str(exc))
