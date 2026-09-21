"""Multi-Track Timeline Canvas with interactive clip moving, trimming, splitting, and drag-and-drop."""

from __future__ import annotations

import json
from typing import Callable

from PyQt6.QtCore import QPoint, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QFont,
    QKeyEvent,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPen,
)
from PyQt6.QtWidgets import QMenu, QWidget

from ....timeline.models import DLSSConfig, MediaAsset, TimelineClip, TimelineProject, TimelineTrack


TRACK_HEADER_WIDTH = 70


class MultiTrackCanvas(QWidget):
    """Interactive visual canvas rendering timeline tracks, clips, playhead needle, and trim handles."""

    clipSelected = pyqtSignal(object)  # TimelineClip | None
    projectModified = pyqtSignal()
    seekRequested = pyqtSignal(int)
    mediaDropped = pyqtSignal(str, int, int)  # asset_id, track_id, drop_frame

    TRACK_HEIGHT = 48
    TRACK_GAP = 4
    TRACK_HEADER_WIDTH = 70

    def __init__(
        self,
        project: TimelineProject,
        pixels_per_frame: float = 2.0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.project = project
        self.pixels_per_frame = max(0.1, pixels_per_frame)
        self.selected_clip_id: str | None = None

        # Interaction state
        self._hover_clip: TimelineClip | None = None
        self._hover_mode: str = "none"  # "none" | "move" | "trim_in" | "trim_out"
        self._drag_clip: TimelineClip | None = None
        self._drag_start_pos: QPoint = QPoint()
        self._drag_orig_in: int = 0
        self._drag_orig_out: int = 0
        self._drag_orig_track_id: int = 0
        self._snap_threshold_px: float = 12.0

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAcceptDrops(True)
        self._update_min_size()

    def set_zoom(self, pixels_per_frame: float) -> None:
        self.pixels_per_frame = max(0.1, min(50.0, pixels_per_frame))
        self._update_min_size()
        self.update()

    def _update_min_size(self) -> None:
        total_tracks = len(self.project.video_tracks)
        h = max(200, total_tracks * (self.TRACK_HEIGHT + self.TRACK_GAP) + 60)
        w = max(1200, int(self.project.get_total_frames() * self.pixels_per_frame) + self.TRACK_HEADER_WIDTH + 300)
        self.setMinimumSize(w, h)

    def frame_to_x(self, frame: int) -> float:
        return self.TRACK_HEADER_WIDTH + (frame * self.pixels_per_frame)

    def x_to_frame(self, x: float) -> int:
        rel_x = max(0.0, x - self.TRACK_HEADER_WIDTH)
        return int(round(rel_x / self.pixels_per_frame))

    def _get_track_y(self, track_index: int) -> float:
        return 10.0 + track_index * (self.TRACK_HEIGHT + self.TRACK_GAP)

    def _get_all_tracks_ordered(self) -> list[TimelineTrack]:
        # Only video tracks V_n down to V1
        return sorted(self.project.video_tracks, key=lambda t: t.track_id, reverse=True)

    def _get_track_at_y(self, y: float) -> TimelineTrack | None:
        tracks = self._get_all_tracks_ordered()
        for idx, track in enumerate(tracks):
            ty = self._get_track_y(idx)
            if ty <= y <= ty + self.TRACK_HEIGHT:
                return track
        return None

    def _get_clip_rect(self, track: TimelineTrack, clip: TimelineClip) -> QRectF:
        tracks = self._get_all_tracks_ordered()
        try:
            track_idx = tracks.index(track)
        except ValueError:
            return QRectF()

        y = self._get_track_y(track_idx)
        x_in = self.frame_to_x(clip.timeline_in)
        x_out = self.frame_to_x(clip.timeline_out)
        w = max(4.0, x_out - x_in)
        return QRectF(x_in, y, w, self.TRACK_HEIGHT)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        width = self.width()
        height = self.height()

        # Canvas background
        painter.fillRect(0, 0, width, height, QColor("#121316"))

        tracks = self._get_all_tracks_ordered()

        # Draw Tracks
        for idx, track in enumerate(tracks):
            ty = self._get_track_y(idx)
            # Track lane background
            is_audio = track.track_type == "audio"
            lane_color = QColor("#16181d") if is_audio else QColor("#181a20")
            painter.fillRect(QRectF(self.TRACK_HEADER_WIDTH, ty, width - self.TRACK_HEADER_WIDTH, self.TRACK_HEIGHT), lane_color)

            # Track separator line
            painter.setPen(QPen(QColor("#262930"), 1))
            painter.drawLine(0, int(ty + self.TRACK_HEIGHT), width, int(ty + self.TRACK_HEIGHT))

            # Track Header Block
            header_rect = QRectF(0, ty, self.TRACK_HEADER_WIDTH - 2, self.TRACK_HEIGHT)
            painter.fillRect(header_rect, QColor("#1f2229"))
            painter.setPen(QColor("#cbd5e1"))
            font = QFont("Consolas", 9)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(header_rect, Qt.AlignmentFlag.AlignCenter, track.name)

            # Draw Clips on Track
            for clip in track.clips:
                rect = self._get_clip_rect(track, clip)
                is_selected = clip.clip_id == self.selected_clip_id
                self._draw_clip(painter, clip, rect, is_selected)

        # Draw Playhead vertical needle
        px = self.frame_to_x(self.project.playhead_frame)
        painter.setPen(QPen(QColor("#ef4444"), 2))
        painter.drawLine(int(px), 0, int(px), height)

        painter.end()

    def _draw_clip(self, painter: QPainter, clip: TimelineClip, rect: QRectF, is_selected: bool) -> None:
        painter.save()

        # Background gradient based on clip type
        if clip.clip_type == "adjustment_layer":
            # Distinctive vibrant Purple/Amber gradient for DLSS 5 Adjustment Layer
            grad = QLinearGradient(rect.topLeft(), rect.bottomLeft())
            grad.setColorAt(0.0, QColor("#9333ea"))
            grad.setColorAt(1.0, QColor("#6b21a8"))
        elif clip.color == "#2563eb":
            # Professional Blue/Cyan for media video clips
            grad = QLinearGradient(rect.topLeft(), rect.bottomLeft())
            grad.setColorAt(0.0, QColor("#2563eb"))
            grad.setColorAt(1.0, QColor("#1d4ed8"))
        else:
            grad = QLinearGradient(rect.topLeft(), rect.bottomLeft())
            grad.setColorAt(0.0, QColor(clip.color))
            grad.setColorAt(1.0, QColor(clip.color).darker(120))

        painter.setBrush(grad)

        # Border styling: Selected clips have thick highlight
        if is_selected:
            painter.setPen(QPen(QColor("#38bdf8"), 2))
        else:
            painter.setPen(QPen(QColor("#0f172a"), 1))

        painter.drawRoundedRect(rect, 4, 4)

        # Clip Title and Badge
        painter.setPen(QColor("#f8fafc"))
        font = QFont("Segoe UI", 8)
        font.setBold(True)
        painter.setFont(font)

        if clip.clip_type == "adjustment_layer":
            badge = f"DLSS 5 FX [{clip.dlss_params.preset}]"
            painter.drawText(rect.adjusted(6, 4, -6, -4), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, badge)
            sub_info = f"Scale: {clip.dlss_params.scale_factor:.1f}x | Sharp: {int(clip.dlss_params.sharpness)}%"
            font_sub = QFont("Segoe UI", 7)
            painter.setFont(font_sub)
            painter.drawText(rect.adjusted(6, 18, -6, -4), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, sub_info)
        else:
            painter.drawText(rect.adjusted(6, 4, -6, -4), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, clip.name)
            dur_sec = clip.duration_frames / max(1.0, self.project.fps)
            info = f"{dur_sec:.2f}s ({clip.duration_frames}f)"
            font_sub = QFont("Segoe UI", 7)
            painter.setFont(font_sub)
            painter.drawText(rect.adjusted(6, 18, -6, -4), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, info)

        # Trim handles visual hint on edges if hovered
        if clip == self._hover_clip and self._hover_mode in ("trim_in", "trim_out"):
            painter.setPen(QPen(QColor("#ffffff"), 2))
            if self._hover_mode == "trim_in":
                painter.drawLine(int(rect.left()), int(rect.top()), int(rect.left()), int(rect.bottom()))
            elif self._hover_mode == "trim_out":
                painter.drawLine(int(rect.right()), int(rect.top()), int(rect.right()), int(rect.bottom()))

        painter.restore()

    def _find_clip_at_pos(self, pos: QPoint) -> tuple[TimelineTrack, TimelineClip, str] | None:
        track = self._get_track_at_y(pos.y())
        if not track:
            return None

        x = pos.x()
        for clip in track.clips:
            rect = self._get_clip_rect(track, clip)
            if rect.contains(float(pos.x()), float(pos.y())):
                # Detect if near left edge (trim_in) or right edge (trim_out)
                if abs(x - rect.left()) <= 8.0:
                    return track, clip, "trim_in"
                elif abs(x - rect.right()) <= 8.0:
                    return track, clip, "trim_out"
                else:
                    return track, clip, "move"
        return None

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            hit = self._find_clip_at_pos(event.pos())
            if hit:
                track, clip, mode = hit
                self.selected_clip_id = clip.clip_id
                self._drag_clip = clip
                self._hover_mode = mode
                self._drag_start_pos = event.pos()
                self._drag_orig_in = clip.timeline_in
                self._drag_orig_out = clip.timeline_out
                self._drag_orig_track_id = clip.track_id
                self.clipSelected.emit(clip)
                self.update()
                event.accept()
                return
            else:
                # Clicked on empty space: seek playhead
                frame = self.x_to_frame(event.pos().x())
                self.project.playhead_frame = frame
                self.selected_clip_id = None
                self.clipSelected.emit(None)
                self.seekRequested.emit(frame)
                self.update()

        elif event.button() == Qt.MouseButton.RightButton:
            # Check if clicked on track header
            if event.pos().x() < self.TRACK_HEADER_WIDTH:
                track = self._get_track_at_y(event.pos().y())
                self._show_track_header_context_menu(track, event.globalPosition().toPoint())
                event.accept()
                return

            # Context menu (Split, Delete, Properties)
            hit = self._find_clip_at_pos(event.pos())
            if hit:
                track, clip, _ = hit
                self.selected_clip_id = clip.clip_id
                self.clipSelected.emit(clip)
                self._show_context_menu(clip, event.globalPosition().toPoint())
                event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        # Cursor change on edge hover
        if not self._drag_clip:
            hit = self._find_clip_at_pos(event.pos())
            if hit:
                _, self._hover_clip, mode = hit
                self._hover_mode = mode
                if mode in ("trim_in", "trim_out"):
                    self.setCursor(Qt.CursorShape.SizeHorCursor)
                else:
                    self.setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                self._hover_clip = None
                self._hover_mode = "none"
                self.setCursor(Qt.CursorShape.ArrowCursor)
            self.update()
            return

        # Dragging logic
        clip = self._drag_clip
        dx_pixels = event.pos().x() - self._drag_start_pos.x()
        dframes = int(round(dx_pixels / self.pixels_per_frame))

        if self._hover_mode == "move":
            # Move horizontally
            dur = self._drag_orig_out - self._drag_orig_in
            new_in = max(0, self._drag_orig_in + dframes)
            new_in = self._apply_snap(new_in)
            clip.timeline_in = new_in
            clip.timeline_out = new_in + dur

            # Move vertically across tracks
            target_track = self._get_track_at_y(event.pos().y())
            if target_track and target_track.track_type == "video" and target_track.track_id != clip.track_id:
                # Remove from old track, add to new track
                old_track = next((t for t in self.project.video_tracks if t.track_id == clip.track_id), None)
                if old_track:
                    old_track.remove_clip(clip.clip_id)
                target_track.add_clip(clip)

        elif self._hover_mode == "trim_in":
            new_in = max(0, min(clip.timeline_out - 1, self._drag_orig_in + dframes))
            new_in = self._apply_snap(new_in)
            trim_delta = new_in - clip.timeline_in
            clip.timeline_in = new_in
            clip.source_in = max(0, clip.source_in + trim_delta)

        elif self._hover_mode == "trim_out":
            new_out = max(clip.timeline_in + 1, self._drag_orig_out + dframes)
            new_out = self._apply_snap(new_out)
            trim_delta = new_out - clip.timeline_out
            clip.timeline_out = new_out
            clip.source_out = clip.source_out + trim_delta

        self._update_min_size()
        self.projectModified.emit()
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._drag_clip:
            self._drag_clip = None
            self.projectModified.emit()
            self.update()
            event.accept()

    def _apply_snap(self, target_frame: int) -> int:
        """Snap target_frame to playhead or nearby clip edges within magnetic threshold."""
        snap_frames = int(self._snap_threshold_px / self.pixels_per_frame)
        candidates = [self.project.playhead_frame]

        for track in self.project.video_tracks:
            for c in track.clips:
                if not self._drag_clip or c.clip_id != self._drag_clip.clip_id:
                    candidates.append(c.timeline_in)
                    candidates.append(c.timeline_out)

        for cand in candidates:
            if abs(cand - target_frame) <= snap_frames:
                return cand
        return target_frame

    def _show_context_menu(self, clip: TimelineClip, pos: QPoint) -> None:
        menu = QMenu(self)
        menu.setStyleSheet("background: #1e293b; color: #f8fafc; border: 1px solid #334155;")
        act_split = menu.addAction("Split at Playhead (C)")
        act_delete = menu.addAction("Delete Clip (Del)")

        chosen = menu.exec(pos)
        if chosen == act_split:
            self.split_at_playhead()
        elif chosen == act_delete:
            self.delete_selected_clip()

    def _show_track_header_context_menu(self, track: TimelineTrack | None, pos: QPoint) -> None:
        menu = QMenu(self)
        menu.setStyleSheet("background: #1e293b; color: #f8fafc; border: 1px solid #334155;")
        act_add = menu.addAction("+ Add Video Track")
        act_del = None
        if track and len(self.project.video_tracks) > 1:
            act_del = menu.addAction(f"- Delete Track ({track.name})")

        chosen = menu.exec(pos)
        if chosen == act_add:
            self.project.add_video_track()
            self._update_min_size()
            self.projectModified.emit()
            self.update()
        elif act_del and chosen == act_del and track:
            self.project.remove_video_track(track.track_id)
            self._update_min_size()
            self.projectModified.emit()
            self.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            self.delete_selected_clip()
        elif event.key() == Qt.Key.Key_C:
            self.split_at_playhead()
        super().keyPressEvent(event)

    def delete_selected_clip(self) -> None:
        if not self.selected_clip_id:
            return
        found = self.project.find_clip_by_id(self.selected_clip_id)
        if found:
            track, clip = found
            track.remove_clip(clip.clip_id)
            self.selected_clip_id = None
            self.clipSelected.emit(None)
            self.projectModified.emit()
            self.update()

    def split_at_playhead(self) -> None:
        frame = self.project.playhead_frame
        target_clip_id = self.selected_clip_id

        # If no clip explicitly selected, find topmost active clip at playhead
        if not target_clip_id:
            active = self.project.get_active_video_clips_at(frame)
            if active:
                target_clip_id = active[-1][1].clip_id

        if target_clip_id:
            res = self.project.split_clip(target_clip_id, frame)
            if res:
                self.projectModified.emit()
                self.update()

    # Drag & Drop handling from MediaPool
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasFormat("application/x-dlss-timeline-asset"):
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if event.mimeData().hasFormat("application/x-dlss-timeline-asset"):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        data = event.mimeData().data("application/x-dlss-timeline-asset")
        if not data:
            return

        payload = json.loads(data.data().decode("utf-8"))
        drop_x = event.position().x()
        drop_y = event.position().y()

        target_track = self._get_track_at_y(drop_y)
        if not target_track:
            target_track = self.project.video_tracks[0] if self.project.video_tracks else None

        if not target_track:
            return

        drop_frame = self.x_to_frame(drop_x)
        item_type = payload.get("type")

        if item_type == "adjustment_layer":
            clip = TimelineClip.create_adjustment_layer(
                track_id=target_track.track_id,
                timeline_in=drop_frame,
                duration_frames=150,  # 5 seconds
            )
            target_track.add_clip(clip)
            self.selected_clip_id = clip.clip_id
            self.clipSelected.emit(clip)

        elif item_type == "media":
            asset_id = payload.get("asset_id")
            if asset_id:
                self.mediaDropped.emit(asset_id, target_track.track_id, drop_frame)

        self._update_min_size()
        self.projectModified.emit()
        self.update()
        event.acceptProposedAction()
