"""Program Monitor (Preview Viewport) displaying composited timeline frames with playback controls."""

from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class TimelineMonitorWidget(QFrame):
    """Program Monitor widget rendering real-time composite frames and transport controls."""

    seekRequested = pyqtSignal(int)
    playToggled = pyqtSignal(bool)
    markInRequested = pyqtSignal()
    markOutRequested = pyqtSignal()
    splitRatioChanged = pyqtSignal(float)

    def __init__(self, fps: float = 30.0, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("timelineMonitorWidget")
        self.fps = max(1.0, fps)
        self.current_frame = 0
        self.total_frames = 900
        self.is_playing = False
        self.loop = True
        self.split_enabled = False
        self.split_ratio = 0.5

        self._current_pixmap: QPixmap | None = None

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._on_play_tick)

        self.setStyleSheet(
            "QFrame#timelineMonitorWidget {"
            "  background-color: #0f1013;"
            "  border-bottom: 1px solid #23252a;"
            "}"
            "QPushButton {"
            "  background-color: #1e2026;"
            "  color: #e2e8f0;"
            "  border: 1px solid #2d3039;"
            "  border-radius: 4px;"
            "  padding: 4px 8px;"
            "  font-size: 11px;"
            "  font-weight: 600;"
            "}"
            "QPushButton:hover {"
            "  background-color: #2b2e38;"
            "}"
            "QPushButton:checked {"
            "  background-color: #2563eb;"
            "  color: #ffffff;"
            "}"
        )

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        # Video Viewport Display Area
        self.viewport = QLabel()
        self.viewport.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.viewport.setStyleSheet("background-color: #000000; border: 1px solid #1e2025; border-radius: 4px;")
        self.viewport.setMinimumSize(480, 270)
        layout.addWidget(self.viewport, 1)

        # Controls & Timecode Bar
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)

        # Transport Buttons
        self.btn_first = QPushButton("|<")
        self.btn_first.setToolTip("Go to Start (Home)")
        self.btn_first.clicked.connect(self.seek_start)
        controls_layout.addWidget(self.btn_first)

        self.btn_prev = QPushButton("<")
        self.btn_prev.setToolTip("Previous Frame (Left Arrow)")
        self.btn_prev.clicked.connect(self.step_backward)
        controls_layout.addWidget(self.btn_prev)

        self.btn_play = QPushButton("Play")
        self.btn_play.setFixedWidth(64)
        self.btn_play.setToolTip("Play / Pause (Space)")
        self.btn_play.clicked.connect(self.toggle_play)
        controls_layout.addWidget(self.btn_play)

        self.btn_next = QPushButton(">")
        self.btn_next.setToolTip("Next Frame (Right Arrow)")
        self.btn_next.clicked.connect(self.step_forward)
        controls_layout.addWidget(self.btn_next)

        self.btn_last = QPushButton(">|")
        self.btn_last.setToolTip("Go to End (End)")
        self.btn_last.clicked.connect(self.seek_end)
        controls_layout.addWidget(self.btn_last)

        # Mark In / Out buttons
        self.btn_mark_in = QPushButton("Mark In (I)")
        self.btn_mark_in.setToolTip("Set Work Area In Point")
        self.btn_mark_in.clicked.connect(self.markInRequested.emit)
        controls_layout.addWidget(self.btn_mark_in)

        self.btn_mark_out = QPushButton("Mark Out (O)")
        self.btn_mark_out.setToolTip("Set Work Area Out Point")
        self.btn_mark_out.clicked.connect(self.markOutRequested.emit)
        controls_layout.addWidget(self.btn_mark_out)

        # Loop Checkbox
        self.chk_loop = QCheckBox("Loop")
        self.chk_loop.setChecked(True)
        self.chk_loop.setStyleSheet("color: #94a3b8; font-size: 11px;")
        self.chk_loop.toggled.connect(self._on_loop_toggled)
        controls_layout.addWidget(self.chk_loop)

        controls_layout.addStretch()

        # Split Preview Toggle & Slider
        self.btn_split = QPushButton("Split Preview")
        self.btn_split.setCheckable(True)
        self.btn_split.setToolTip("Compare Raw Composite vs DLSS 5 Adjustment Layer")
        self.btn_split.toggled.connect(self._on_split_toggled)
        controls_layout.addWidget(self.btn_split)

        self.slider_split = QSlider(Qt.Orientation.Horizontal)
        self.slider_split.setRange(5, 95)
        self.slider_split.setValue(50)
        self.slider_split.setFixedWidth(90)
        self.slider_split.setToolTip("Split comparison position")
        self.slider_split.setVisible(False)
        self.slider_split.valueChanged.connect(self._on_split_slider_changed)
        controls_layout.addWidget(self.slider_split)

        controls_layout.addSpacing(10)

        # Timecode Display
        self.lbl_timecode = QLabel("00:00:00:00 / 00:00:00:00 [F: 0]")
        self.lbl_timecode.setStyleSheet("color: #38bdf8; font-family: Consolas, monospace; font-size: 11px; font-weight: 700;")
        controls_layout.addWidget(self.lbl_timecode)

        layout.addLayout(controls_layout)

    def display_frame(self, bgr_frame: np.ndarray, current_frame: int, total_frames: int) -> None:
        """Render frame in viewport maintaining aspect ratio."""
        self.current_frame = current_frame
        self.total_frames = max(1, total_frames)

        # Update timecode
        cur_sec = current_frame / self.fps
        tot_sec = total_frames / self.fps
        tc_cur = self._format_timecode(cur_sec, current_frame)
        tc_tot = self._format_timecode(tot_sec, total_frames)
        self.lbl_timecode.setText(f"{tc_cur} / {tc_tot} [F: {current_frame}]")

        if bgr_frame is None or bgr_frame.size == 0:
            return

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)

        # Scale to viewport maintaining aspect ratio
        vp_size = self.viewport.size()
        scaled = pixmap.scaled(
            vp_size.width(),
            vp_size.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.viewport.setPixmap(scaled)

    def _format_timecode(self, seconds: float, frame_idx: int) -> str:
        s = int(seconds)
        hh = s // 3600
        mm = (s % 3600) // 60
        ss = s % 60
        ff = int(frame_idx % int(self.fps))
        return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"

    def toggle_play(self) -> None:
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def play(self) -> None:
        self.is_playing = True
        self.btn_play.setText("Pause")
        self.btn_play.setStyleSheet("background-color: #2563eb; color: white;")
        interval_ms = int(1000.0 / self.fps)
        self._play_timer.start(interval_ms)
        self.playToggled.emit(True)

    def pause(self) -> None:
        self.is_playing = False
        self.btn_play.setText("Play")
        self.btn_play.setStyleSheet("")
        self._play_timer.stop()
        self.playToggled.emit(False)

    def _on_play_tick(self) -> None:
        next_f = self.current_frame + 1
        if next_f >= self.total_frames:
            if self.loop:
                next_f = 0
            else:
                self.pause()
                return
        self.current_frame = next_f
        self.seekRequested.emit(next_f)

    def step_forward(self) -> None:
        self.pause()
        next_f = min(self.total_frames - 1, self.current_frame + 1)
        self.seekRequested.emit(next_f)

    def step_backward(self) -> None:
        self.pause()
        prev_f = max(0, self.current_frame - 1)
        self.seekRequested.emit(prev_f)

    def seek_start(self) -> None:
        self.pause()
        self.seekRequested.emit(0)

    def seek_end(self) -> None:
        self.pause()
        self.seekRequested.emit(max(0, self.total_frames - 1))

    def _on_loop_toggled(self, checked: bool) -> None:
        self.loop = checked

    def _on_split_toggled(self, checked: bool) -> None:
        self.split_enabled = checked
        self.slider_split.setVisible(checked)
        ratio = (self.slider_split.value() / 100.0) if checked else None
        self.splitRatioChanged.emit(ratio if ratio is not None else -1.0)

    def _on_split_slider_changed(self, val: int) -> None:
        if self.split_enabled:
            self.splitRatioChanged.emit(val / 100.0)
