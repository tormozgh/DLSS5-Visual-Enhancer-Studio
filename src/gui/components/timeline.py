"""Professional video timeline scrubber with playback controls and timecode display."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class ClickableSlider(QSlider):
    """Horizontal slider that instantly jumps to click position."""

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.orientation() == Qt.Orientation.Horizontal:
            width = self.width()
            if width > 0:
                val_range = self.maximum() - self.minimum()
                pos = event.position().x()
                new_val = self.minimum() + int(round(val_range * (pos / width)))
                self.setValue(max(self.minimum(), min(self.maximum(), new_val)))
                event.accept()
                return
        super().mousePressEvent(event)


class VideoTimelineBar(QFrame):
    """Professional video timeline scrubber with transport controls, loop, and timecode."""

    frameChanged = pyqtSignal(int)
    playbackToggled = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("timelineBar")
        self.setStyleSheet(
            "QFrame#timelineBar {"
            "  background-color: #16171b;"
            "  border-top: 1px solid #282a30;"
            "  border-bottom: 1px solid #282a30;"
            "  padding: 4px;"
            "}"
        )

        self._total_frames: int = 0
        self._fps: float = 30.0
        self._current_frame: int = 0
        self._is_playing: bool = False
        self._loop: bool = True

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._on_play_tick)

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        # Row 1: Scrubber slider
        self.slider = ClickableSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setValue(0)
        self.slider.setCursor(Qt.CursorShape.PointingHandCursor)
        self.slider.setStyleSheet(
            "QSlider::groove:horizontal { height: 6px; background: #282a30; border-radius: 3px; }"
            "QSlider::sub-page:horizontal { background: #505460; border-radius: 3px; }"
            "QSlider::handle:horizontal { width: 14px; height: 14px; margin: -4px 0; background: #d0d4dc; border-radius: 7px; }"
            "QSlider::handle:horizontal:hover { background: #ffffff; }"
        )
        self.slider.valueChanged.connect(self._on_slider_value_changed)
        layout.addWidget(self.slider)

        # Row 2: Transport controls and timecode
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)

        # First frame
        self.btn_first = QPushButton("|<")
        self.btn_first.setProperty("class", "mini-btn")
        self.btn_first.setToolTip("First Frame (Home)")
        self.btn_first.clicked.connect(self.seek_first)
        controls_layout.addWidget(self.btn_first)

        # Step back
        self.btn_prev = QPushButton("<")
        self.btn_prev.setProperty("class", "mini-btn")
        self.btn_prev.setToolTip("Previous Frame (Left Arrow)")
        self.btn_prev.clicked.connect(self.step_prev)
        controls_layout.addWidget(self.btn_prev)

        # Play / Pause
        self.btn_play = QPushButton("Play")
        self.btn_play.setProperty("class", "mini-btn")
        self.btn_play.setFixedWidth(56)
        self.btn_play.setToolTip("Play / Pause (Space)")
        self.btn_play.clicked.connect(self.toggle_play)
        controls_layout.addWidget(self.btn_play)

        # Step forward
        self.btn_next = QPushButton(">")
        self.btn_next.setProperty("class", "mini-btn")
        self.btn_next.setToolTip("Next Frame (Right Arrow)")
        self.btn_next.clicked.connect(self.step_next)
        controls_layout.addWidget(self.btn_next)

        # Last frame
        self.btn_last = QPushButton(">|")
        self.btn_last.setProperty("class", "mini-btn")
        self.btn_last.setToolTip("Last Frame (End)")
        self.btn_last.clicked.connect(self.seek_last)
        controls_layout.addWidget(self.btn_last)

        # Loop toggle
        self.btn_loop = QPushButton("Loop")
        self.btn_loop.setProperty("class", "mini-btn")
        self.btn_loop.setCheckable(True)
        self.btn_loop.setChecked(True)
        self.btn_loop.setToolTip("Toggle Playback Loop")
        self.btn_loop.toggled.connect(self._on_loop_toggled)
        controls_layout.addWidget(self.btn_loop)

        controls_layout.addStretch()

        # Timecode Display
        self.lbl_timecode = QLabel("00:00:00:00 / 00:00:00:00 [Frame: 0 / 0]")
        self.lbl_timecode.setStyleSheet(
            "color: #d0d4dc; font-family: Consolas, 'Courier New', monospace; font-size: 11px; font-weight: 600;"
        )
        controls_layout.addWidget(self.lbl_timecode)

        layout.addLayout(controls_layout)

    def set_media_info(self, total_frames: int, fps: float) -> None:
        """Configure timeline for a newly loaded video."""
        self.pause()
        self._total_frames = max(1, total_frames)
        self._fps = max(1.0, fps)
        self._current_frame = 0

        self.slider.blockSignals(True)
        self.slider.setRange(0, self._total_frames - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)

        interval_ms = max(10, int(round(1000.0 / self._fps)))
        self._play_timer.setInterval(interval_ms)

        self._update_timecode_display(0)

    def _format_timecode(self, frame_idx: int) -> str:
        total_sec = frame_idx / self._fps
        h = int(total_sec // 3600)
        m = int((total_sec % 3600) // 60)
        s = int(total_sec % 60)
        ff = int(round((total_sec - int(total_sec)) * self._fps))
        return f"{h:02d}:{m:02d}:{s:02d}:{ff:02d}"

    def _update_timecode_display(self, frame_idx: int) -> None:
        cur_tc = self._format_timecode(frame_idx)
        max_idx = max(0, self._total_frames - 1)
        tot_tc = self._format_timecode(max_idx)
        self.lbl_timecode.setText(f"{cur_tc} / {tot_tc} [Frame: {frame_idx + 1} / {self._total_frames}]")

    def _on_slider_value_changed(self, value: int) -> None:
        self._current_frame = value
        self._update_timecode_display(value)
        self.frameChanged.emit(value)

    def _on_loop_toggled(self, checked: bool) -> None:
        self._loop = checked

    def toggle_play(self) -> None:
        if self._is_playing:
            self.pause()
        else:
            self.play()

    def play(self) -> None:
        if self._total_frames <= 1:
            return
        self._is_playing = True
        self._frame_in_flight = False
        self.btn_play.setText("Pause")
        self._play_timer.start()
        self.playbackToggled.emit(True)

    def pause(self) -> None:
        self._is_playing = False
        self._frame_in_flight = False
        self.btn_play.setText("Play")
        self._play_timer.stop()
        self.playbackToggled.emit(False)

    def notify_frame_ready(self) -> None:
        """Called when the neural engine finishes rendering a frame to pace playback."""
        self._frame_in_flight = False

    def _on_play_tick(self) -> None:
        if self._total_frames <= 1:
            self.pause()
            return

        if self._frame_in_flight:
            # Engine is still processing the previous frame; don't pile up the queue
            return

        next_frame = self._current_frame + 1
        if next_frame >= self._total_frames:
            if self._loop:
                next_frame = 0
            else:
                self.pause()
                return

        self._frame_in_flight = True
        self.set_current_frame(next_frame)

    def set_current_frame(self, frame_idx: int) -> None:
        val = max(0, min(self._total_frames - 1, frame_idx))
        self.slider.setValue(val)

    def step_next(self) -> None:
        self.pause()
        if self._current_frame < self._total_frames - 1:
            self.set_current_frame(self._current_frame + 1)

    def step_prev(self) -> None:
        self.pause()
        if self._current_frame > 0:
            self.set_current_frame(self._current_frame - 1)

    def seek_first(self) -> None:
        self.pause()
        self.set_current_frame(0)

    def seek_last(self) -> None:
        self.pause()
        self.set_current_frame(self._total_frames - 1)

    def current_frame(self) -> int:
        return self._current_frame
