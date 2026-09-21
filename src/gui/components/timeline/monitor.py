"""Program Monitor (Preview Viewport) displaying composited timeline frames with interactive Split View, Zoom, and Pan."""

from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..split_canvas import CanvasViewMode, SplitCanvas


class TimelineMonitorWidget(QFrame):
    """Program Monitor widget rendering real-time composite frames with transport controls and SplitCanvas."""

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
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # 1. Interactive Viewport Canvas (Split, Zoom, Pan)
        self.canvas = SplitCanvas(self)
        self.canvas.setMinimumSize(480, 270)
        layout.addWidget(self.canvas, 1)

        # 2. Viewport Toolbar (Split View / Side-by-Side / Fit / 1:1)
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(2, 0, 2, 0)
        toolbar_layout.setSpacing(6)

        self.btn_view_split = QPushButton("Split View")
        self.btn_view_split.setCheckable(True)
        self.btn_view_split.setChecked(True)
        self.btn_view_split.clicked.connect(lambda: self.canvas.set_view_mode(CanvasViewMode.SPLIT))

        self.btn_view_sbs = QPushButton("Side-by-Side")
        self.btn_view_sbs.setCheckable(True)
        self.btn_view_sbs.clicked.connect(lambda: self.canvas.set_view_mode(CanvasViewMode.SIDE_BY_SIDE))

        self.view_mode_group = QButtonGroup(self)
        self.view_mode_group.addButton(self.btn_view_split)
        self.view_mode_group.addButton(self.btn_view_sbs)

        self.btn_fit = QPushButton("Fit")
        self.btn_fit.setToolTip("Fit video to viewport (F)")
        self.btn_fit.clicked.connect(self.canvas.fit_to_view)

        self.btn_1to1 = QPushButton("1:1")
        self.btn_1to1.setToolTip("100% Original Pixel View")
        self.btn_1to1.clicked.connect(self.canvas.reset_1to1)

        self.btn_show_orig = QPushButton("Original")
        self.btn_show_orig.setToolTip("Show only original composite before DLSS 5")
        self.btn_show_orig.clicked.connect(lambda: self.canvas.set_split_ratio(1.0))

        self.btn_show_dlss = QPushButton("DLSS 5")
        self.btn_show_dlss.setToolTip("Show only DLSS 5 enhanced composite")
        self.btn_show_dlss.clicked.connect(lambda: self.canvas.set_split_ratio(0.0))

        toolbar_layout.addWidget(QLabel("View:"))
        toolbar_layout.addWidget(self.btn_view_split)
        toolbar_layout.addWidget(self.btn_view_sbs)
        toolbar_layout.addSpacing(6)
        toolbar_layout.addWidget(self.btn_fit)
        toolbar_layout.addWidget(self.btn_1to1)
        toolbar_layout.addWidget(self.btn_show_orig)
        toolbar_layout.addWidget(self.btn_show_dlss)
        toolbar_layout.addStretch()

        layout.addLayout(toolbar_layout)

        # 3. Transport Controls & Timecode Bar
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(2, 0, 2, 0)
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

        # Timecode Display
        self.lbl_timecode = QLabel("00:00:00:00 / 00:00:00:00 [F: 0]")
        self.lbl_timecode.setStyleSheet("color: #38bdf8; font-family: Consolas, monospace; font-size: 11px; font-weight: 700;")
        controls_layout.addWidget(self.lbl_timecode)

        layout.addLayout(controls_layout)

    def display_frame(
        self,
        bgr_frame: np.ndarray | None,
        raw_bgr: np.ndarray | None = None,
        current_frame: int = 0,
        total_frames: int = 1,
    ) -> None:
        """Render composite and raw frames in SplitCanvas with zoom, pan, and interactive split."""
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

        rgb_after = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_after.shape
        qimg_after = QImage(rgb_after.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()

        if raw_bgr is not None and raw_bgr.size > 0:
            rgb_before = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
            qimg_before = QImage(rgb_before.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        else:
            qimg_before = qimg_after

        self.canvas.set_images(qimg_before, qimg_after)

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
