"""Time Ruler component with playhead scrubber, In/Out work area, and timecode ticks."""

from __future__ import annotations

from PyQt6.QtCore import QPoint, QRect, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPen, QPolygon
from PyQt6.QtWidgets import QWidget


class TimeRulerWidget(QWidget):
    """Interactive time ruler displaying timecode ticks, playhead needle, and In/Out points."""

    seekRequested = pyqtSignal(int)
    inPointChanged = pyqtSignal(int)
    outPointChanged = pyqtSignal(int)

    def __init__(
        self,
        fps: float = 30.0,
        total_frames: int = 900,
        pixels_per_frame: float = 2.0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.fps = max(1.0, fps)
        self.total_frames = max(30, total_frames)
        self.pixels_per_frame = max(0.1, pixels_per_frame)

        self.current_frame = 0
        self.in_point: int | None = None
        self.out_point: int | None = None

        self._dragging_playhead = False
        self._dragging_in = False
        self._dragging_out = False

        self.setFixedHeight(32)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_zoom(self, pixels_per_frame: float) -> None:
        self.pixels_per_frame = max(0.1, min(50.0, pixels_per_frame))
        self.update()

    def set_current_frame(self, frame: int) -> None:
        self.current_frame = max(0, min(self.total_frames, frame))
        self.update()

    def set_in_point(self, frame: int | None) -> None:
        self.in_point = frame
        self.inPointChanged.emit(frame if frame is not None else 0)
        self.update()

    def set_out_point(self, frame: int | None) -> None:
        self.out_point = frame
        self.outPointChanged.emit(frame if frame is not None else self.total_frames)
        self.update()

    def frame_to_x(self, frame: int) -> float:
        return frame * self.pixels_per_frame

    def x_to_frame(self, x: float) -> int:
        return max(0, min(self.total_frames, int(round(x / self.pixels_per_frame))))

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        width = self.width()
        height = self.height()

        # Background
        painter.fillRect(0, 0, width, height, QColor("#16171b"))

        # Shaded In/Out work area if set
        if self.in_point is not None or self.out_point is not None:
            in_x = self.frame_to_x(self.in_point if self.in_point is not None else 0)
            out_x = self.frame_to_x(self.out_point if self.out_point is not None else self.total_frames)
            if out_x > in_x:
                work_rect = QRectF(in_x, 0, out_x - in_x, height)
                painter.fillRect(work_rect, QColor(56, 189, 248, 35))

        # Draw timecode ticks
        painter.setPen(QPen(QColor("#475569"), 1))
        font = QFont("Consolas", 8)
        font.setBold(True)
        painter.setFont(font)

        # Decide tick interval based on zoom
        if self.pixels_per_frame >= 8.0:
            frame_step = 5
            label_step = int(self.fps)
        elif self.pixels_per_frame >= 2.0:
            frame_step = int(self.fps)
            label_step = int(self.fps * 2)
        elif self.pixels_per_frame >= 0.5:
            frame_step = int(self.fps * 5)
            label_step = int(self.fps * 10)
        else:
            frame_step = int(self.fps * 15)
            label_step = int(self.fps * 30)

        frame_step = max(1, frame_step)
        label_step = max(frame_step, label_step)

        for f in range(0, self.total_frames + 1, frame_step):
            x = self.frame_to_x(f)
            if x > width + 100:
                break

            if f % label_step == 0:
                painter.setPen(QPen(QColor("#94a3b8"), 1))
                painter.drawLine(int(x), height - 12, int(x), height)
                sec = int(f / self.fps)
                mm = sec // 60
                ss = sec % 60
                label = f"{mm:02d}:{ss:02d}"
                painter.drawText(int(x) + 3, 14, label)
            else:
                painter.setPen(QPen(QColor("#334155"), 1))
                painter.drawLine(int(x), height - 6, int(x), height)

        # Draw In Point marker ([)
        if self.in_point is not None:
            ix = int(self.frame_to_x(self.in_point))
            painter.setPen(QPen(QColor("#38bdf8"), 2))
            painter.drawLine(ix, 0, ix, height)
            painter.drawLine(ix, 0, ix + 8, 0)
            painter.drawLine(ix, height - 1, ix + 8, height - 1)

        # Draw Out Point marker (])
        if self.out_point is not None:
            ox = int(self.frame_to_x(self.out_point))
            painter.setPen(QPen(QColor("#38bdf8"), 2))
            painter.drawLine(ox, 0, ox, height)
            painter.drawLine(ox - 8, 0, ox, 0)
            painter.drawLine(ox - 8, height - 1, ox, height - 1)

        # Draw Playhead needle handle
        px = self.frame_to_x(self.current_frame)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#ef4444"))  # Vivid red playhead needle
        poly = QPolygon([
            QPoint(int(px - 7), 0),
            QPoint(int(px + 7), 0),
            QPoint(int(px + 7), height - 8),
            QPoint(int(px), height),
            QPoint(int(px - 7), height - 8),
        ])
        painter.drawPolygon(poly)

        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            x = event.position().x()
            frame = self.x_to_frame(x)
            self._dragging_playhead = True
            self.set_current_frame(frame)
            self.seekRequested.emit(frame)
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging_playhead:
            x = event.position().x()
            frame = self.x_to_frame(x)
            self.set_current_frame(frame)
            self.seekRequested.emit(frame)
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging_playhead = False
            event.accept()
