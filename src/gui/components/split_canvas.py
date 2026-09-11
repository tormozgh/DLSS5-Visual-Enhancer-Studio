"""Interactive Preview Canvas supporting both Split-Slider and Side-by-Side dual views."""

from __future__ import annotations

from enum import Enum

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QFont,
    QImage,
    QMouseEvent,
    QPaintEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PyQt6.QtWidgets import QWidget


class CanvasViewMode(str, Enum):
    SPLIT = "split"
    SIDE_BY_SIDE = "side_by_side"
    ONLY_BEFORE = "only_before"
    ONLY_AFTER = "only_after"


class SplitCanvas(QWidget):
    """Interactive Preview Canvas supporting Split Slider, Side-by-Side, pan, and zoom."""

    imageDropped = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Image state
        self._pixmap_before: QPixmap | None = None
        self._pixmap_after: QPixmap | None = None
        self._image_width = 0
        self._image_height = 0

        # View mode
        self._view_mode: CanvasViewMode = CanvasViewMode.SPLIT

        # View transform (pan & zoom)
        self._zoom: float = 1.0
        self._offset: QPointF = QPointF(0, 0)
        self._split_ratio: float = 0.5

        # Mouse interaction state
        self._dragging_split = False
        self._panning = False
        self._last_mouse_pos = QPointF(0, 0)

        # Neutral Theme Colors
        self._bg_color = QColor("#101114")
        self._grid_color = QColor("#1b1d22")
        self._divider_color = QColor("#8a909e")
        self._divider_hover_color = QColor("#d0d4dc")
        self._badge_bg = QColor(22, 24, 28, 220)
        self._badge_border = QColor("#353841")
        self._badge_text_before = QColor("#9ca0ab")
        self._badge_text_after = QColor("#e0e3ea")

        self.setMinimumSize(400, 300)

    def set_view_mode(self, mode: CanvasViewMode) -> None:
        self._view_mode = mode
        self.fit_to_view()
        self.update()

    def view_mode(self) -> CanvasViewMode:
        return self._view_mode

    def has_images(self) -> bool:
        return self._pixmap_before is not None

    def set_before_image(self, before: QImage | QPixmap | None) -> None:
        if isinstance(before, QImage):
            self._pixmap_before = QPixmap.fromImage(before)
        elif isinstance(before, QPixmap):
            self._pixmap_before = before
        else:
            self._pixmap_before = None

        if self._pixmap_after is None:
            self._pixmap_after = self._pixmap_before

        if self._pixmap_before is not None:
            self._image_width = self._pixmap_before.width()
            self._image_height = self._pixmap_before.height()

        self.update()

    def set_images(self, before: QImage | QPixmap | None, after: QImage | QPixmap | None = None) -> None:
        first_load = self._pixmap_before is None and before is not None

        if isinstance(before, QImage):
            self._pixmap_before = QPixmap.fromImage(before)
        elif isinstance(before, QPixmap):
            self._pixmap_before = before
        else:
            self._pixmap_before = None

        if isinstance(after, QImage):
            self._pixmap_after = QPixmap.fromImage(after)
        elif isinstance(after, QPixmap):
            self._pixmap_after = after
        elif before is None:
            self._pixmap_after = None
        elif self._pixmap_after is None:
            self._pixmap_after = self._pixmap_before

        if self._pixmap_before is not None:
            self._image_width = self._pixmap_before.width()
            self._image_height = self._pixmap_before.height()
            if first_load:
                self.fit_to_view()
        else:
            self._image_width = 0
            self._image_height = 0

        self.update()

    def set_after_image(self, after: QImage | QPixmap | None) -> None:
        if isinstance(after, QImage):
            self._pixmap_after = QPixmap.fromImage(after)
        elif isinstance(after, QPixmap):
            self._pixmap_after = after
        else:
            self._pixmap_after = self._pixmap_before
        self.update()

    def fit_to_view(self) -> None:
        if self._image_width <= 0 or self._image_height <= 0:
            return

        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return

        if self._view_mode == CanvasViewMode.SIDE_BY_SIDE:
            # Each side gets half width minus margins
            half_w = (w - 30) / 2.0
            scale_w = half_w / self._image_width
            scale_h = (h - 40) / self._image_height
            self._zoom = max(0.05, min(scale_w, scale_h, 1.0))
            display_w = self._image_width * self._zoom
            display_h = self._image_height * self._zoom
            # Center vertically, align relative to half pane
            self._offset = QPointF((half_w - display_w) / 2.0 + 10, (h - display_h) / 2.0)
        else:
            scale_w = (w - 40) / self._image_width
            scale_h = (h - 40) / self._image_height
            self._zoom = max(0.05, min(scale_w, scale_h, 1.0))
            display_w = self._image_width * self._zoom
            display_h = self._image_height * self._zoom
            self._offset = QPointF((w - display_w) / 2.0, (h - display_h) / 2.0)

        self.update()

    def reset_1to1(self) -> None:
        if self._image_width <= 0 or self._image_height <= 0:
            return
        self._zoom = 1.0
        w, h = self.width(), self.height()
        if self._view_mode == CanvasViewMode.SIDE_BY_SIDE:
            half_w = (w - 30) / 2.0
            self._offset = QPointF((half_w - self._image_width) / 2.0 + 10, (h - self._image_height) / 2.0)
        else:
            self._offset = QPointF((w - self._image_width) / 2.0, (h - self._image_height) / 2.0)
        self.update()

    def set_split_ratio(self, ratio: float) -> None:
        self._split_ratio = max(0.0, min(1.0, ratio))
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        painter.fillRect(self.rect(), self._bg_color)
        self._draw_grid(painter)

        if self._pixmap_before is None and self._pixmap_after is None:
            self._draw_placeholder(painter)
            return

        w = self.width()
        h = self.height()

        if self._view_mode == CanvasViewMode.SIDE_BY_SIDE:
            self._paint_side_by_side(painter, w, h)
        elif self._view_mode == CanvasViewMode.ONLY_BEFORE:
            self._paint_single(painter, self._pixmap_before, "ORIGINAL", w, h)
        elif self._view_mode == CanvasViewMode.ONLY_AFTER:
            self._paint_single(painter, self._pixmap_after, "DLSS 5 NEURAL", w, h)
        else:
            self._paint_split(painter, w, h)

    def _paint_split(self, painter: QPainter, w: int, h: int) -> None:
        split_x = int(round(w * self._split_ratio))
        display_w = self._image_width * self._zoom
        display_h = self._image_height * self._zoom
        img_rect = QRectF(self._offset.x(), self._offset.y(), display_w, display_h)

        # After (Right side)
        if self._pixmap_after is not None:
            painter.save()
            painter.setClipRect(split_x, 0, w - split_x, h)
            painter.drawPixmap(img_rect.toRect(), self._pixmap_after)
            painter.restore()

        # Before (Left side)
        if self._pixmap_before is not None:
            painter.save()
            painter.setClipRect(0, 0, split_x, h)
            painter.drawPixmap(img_rect.toRect(), self._pixmap_before)
            painter.restore()

        # Splitter Line and Handle
        self._draw_splitter(painter, split_x, h)

        # Badges
        self._draw_split_overlays(painter, split_x, w, h)

    def _paint_side_by_side(self, painter: QPainter, w: int, h: int) -> None:
        mid_x = w // 2
        display_w = self._image_width * self._zoom
        display_h = self._image_height * self._zoom

        # 1. Left Viewport (Before / Original)
        painter.save()
        painter.setClipRect(0, 0, mid_x, h)
        left_rect = QRectF(self._offset.x(), self._offset.y(), display_w, display_h)
        if self._pixmap_before is not None:
            painter.drawPixmap(left_rect.toRect(), self._pixmap_before)
        self._draw_badge(painter, 14, 14, 110, 26, "BEFORE (ORIGINAL)", self._badge_text_before)
        painter.restore()

        # 2. Right Viewport (After / DLSS 5)
        painter.save()
        painter.setClipRect(mid_x, 0, w - mid_x, h)
        # Shift offset by mid_x
        right_rect = QRectF(self._offset.x() + mid_x, self._offset.y(), display_w, display_h)
        if self._pixmap_after is not None:
            painter.drawPixmap(right_rect.toRect(), self._pixmap_after)
        self._draw_badge(painter, mid_x + 14, 14, 120, 26, "AFTER (DLSS 5)", self._badge_text_after)
        painter.restore()

        # Middle Divider Line
        painter.save()
        pen = QPen(self._badge_border, 2)
        painter.setPen(pen)
        painter.drawLine(mid_x, 0, mid_x, h)
        painter.restore()

        # Resolution badge
        self._draw_resolution_badge(painter, h)

    def _paint_single(self, painter: QPainter, pixmap: QPixmap | None, label: str, w: int, h: int) -> None:
        display_w = self._image_width * self._zoom
        display_h = self._image_height * self._zoom
        img_rect = QRectF(self._offset.x(), self._offset.y(), display_w, display_h)

        if pixmap is not None:
            painter.drawPixmap(img_rect.toRect(), pixmap)

        self._draw_badge(painter, 14, 14, 120, 26, label, self._badge_text_after)
        self._draw_resolution_badge(painter, h)

    def _draw_grid(self, painter: QPainter) -> None:
        pen = QPen(self._grid_color, 1)
        painter.setPen(pen)
        grid_size = 40
        for x in range(0, self.width(), grid_size):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), grid_size):
            painter.drawLine(0, y, self.width(), y)

    def _draw_placeholder(self, painter: QPainter) -> None:
        painter.save()
        painter.setPen(QColor("#4b505c"))
        font = QFont("Segoe UI", 14)
        font.setBold(True)
        painter.setFont(font)

        text = "Drag & Drop Image or Video Here\nor use Browse Media in the controls"
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, text)

        sub_font = QFont("Segoe UI", 10)
        painter.setFont(sub_font)
        painter.setPen(QColor("#8a909e"))
        hint_rect = QRectF(0, self.height() / 2 + 50, self.width(), 30)
        painter.drawText(hint_rect, Qt.AlignmentFlag.AlignCenter, "Supports PNG, JPG, WebP, AVIF, TIFF, MP4, MKV, MOV")
        painter.restore()

    def _draw_splitter(self, painter: QPainter, split_x: int, height: int) -> None:
        painter.save()
        pen = QPen(self._divider_color, 2)
        painter.setPen(pen)
        painter.drawLine(split_x, 0, split_x, height)

        handle_y = height // 2
        handle_w = 26
        handle_h = 42
        handle_rect = QRectF(split_x - handle_w / 2, handle_y - handle_h / 2, handle_w, handle_h)

        painter.setBrush(QColor("#1f2126"))
        painter.setPen(QPen(self._divider_color, 1.5))
        painter.drawRoundedRect(handle_rect, 5, 5)

        # Left/Right indicator arrows drawn on handle
        painter.setPen(QPen(QColor("#d0d4dc"), 1.5))
        painter.drawLine(split_x - 5, handle_y, split_x - 2, handle_y - 4)
        painter.drawLine(split_x - 5, handle_y, split_x - 2, handle_y + 4)

        painter.drawLine(split_x + 5, handle_y, split_x + 2, handle_y - 4)
        painter.drawLine(split_x + 5, handle_y, split_x + 2, handle_y + 4)
        painter.restore()

    def _draw_badge(self, painter: QPainter, x: float, y: float, w: float, h: float, text: str, text_color: QColor) -> None:
        painter.save()
        badge_rect = QRectF(x, y, w, h)
        painter.setBrush(self._badge_bg)
        painter.setPen(QPen(self._badge_border, 1))
        painter.drawRoundedRect(badge_rect, 4, 4)

        font = QFont("Segoe UI", 9)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(text_color)
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()

    def _draw_split_overlays(self, painter: QPainter, split_x: int, w: int, h: int) -> None:
        if split_x > 80:
            self._draw_badge(painter, 14, 14, 90, 24, "ORIGINAL", self._badge_text_before)
        if w - split_x > 100:
            self._draw_badge(painter, w - 124, 14, 110, 24, "DLSS 5 NEURAL", self._badge_text_after)
        self._draw_resolution_badge(painter, h)

    def _draw_resolution_badge(self, painter: QPainter, h: int) -> None:
        if self._image_width > 0:
            zoom_pct = int(round(self._zoom * 100))
            info_text = f"{self._image_width} × {self._image_height}  |  {zoom_pct}%"
            self._draw_badge(painter, 14, h - 38, 150, 24, info_text, QColor("#d0d4dc"))

    # Mouse Events
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._view_mode == CanvasViewMode.SPLIT:
            split_x = int(round(self.width() * self._split_ratio))
            dist_to_split = abs(event.position().x() - split_x)
            if event.button() == Qt.MouseButton.LeftButton and dist_to_split < 18:
                self._dragging_split = True
                self.setCursor(Qt.CursorShape.SplitHCursor)
                return

        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._panning = True
            self._last_mouse_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._view_mode == CanvasViewMode.SPLIT and self._dragging_split:
            ratio = event.position().x() / max(1, self.width())
            self.set_split_ratio(ratio)
            return

        if self._panning:
            delta = event.position() - self._last_mouse_pos
            self._offset += delta
            self._last_mouse_pos = event.position()
            self.update()
        elif self._view_mode == CanvasViewMode.SPLIT:
            split_x = int(round(self.width() * self._split_ratio))
            dist_to_split = abs(event.position().x() - split_x)
            if dist_to_split < 18:
                self.setCursor(Qt.CursorShape.SplitHCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._dragging_split = False
        self._panning = False
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._image_width <= 0:
            return

        angle = event.angleDelta().y()
        factor = 1.15 if angle > 0 else 0.85

        old_zoom = self._zoom
        new_zoom = max(0.05, min(10.0, old_zoom * factor))
        if new_zoom == old_zoom:
            return

        mouse_pos = event.position()
        self._offset = mouse_pos - (mouse_pos - self._offset) * (new_zoom / old_zoom)
        self._zoom = new_zoom
        self.update()

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path:
                self.imageDropped.emit(file_path)
