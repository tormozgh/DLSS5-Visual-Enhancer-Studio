"""Media Pool widget for importing, managing, and dragging video assets and FX layers onto timeline."""

from __future__ import annotations

import json
import os

import cv2
from PyQt6.QtCore import QByteArray, QPoint, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QDrag, QIcon, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ....timeline.frame_cache import VideoFrameCache
from ....timeline.models import MediaAsset


class MediaAssetListItem(QListWidgetItem):
    """List item representing an imported media asset or FX adjustment layer generator."""

    def __init__(
        self,
        asset: MediaAsset | None,
        is_adjustment_layer: bool = False,
        fx_type: str = "dlss5",
    ) -> None:
        super().__init__()
        self.asset = asset
        self.is_adjustment_layer = is_adjustment_layer
        self.fx_type = fx_type

        if is_adjustment_layer:
            if fx_type == "reshade":
                self.setText("ReShade FX Layer\n[Cinematic Color Grading & CAS]")
                self.setToolTip("Drag onto a higher video track to apply ReShade shaders and grading")
            else:
                self.setText("DLSS 5 Adjustment Layer\n[AI Neural Reconstruction Layer]")
                self.setToolTip("Drag onto a higher video track to enhance underlying clips")
        elif asset:
            dur_m = int(asset.duration_sec // 60)
            dur_s = int(asset.duration_sec % 60)
            info = f"{asset.name}\n{asset.width}x{asset.height} | {asset.fps:.1f} fps | {dur_m:02d}:{dur_s:02d}"
            self.setText(info)
            self.setToolTip(f"File: {asset.file_path}\nTotal Frames: {asset.duration_frames}")


class DraggableMediaListWidget(QListWidget):
    """List widget supporting reliable drag initiation with custom MIME data."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._drag_start_pos = QPoint()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_pos = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            dist = (event.position().toPoint() - self._drag_start_pos).manhattanLength()
            if dist >= 4:
                item = self.itemAt(self._drag_start_pos) or self.currentItem()
                if isinstance(item, MediaAssetListItem):
                    drag = QDrag(self)
                    mime = QMimeData()
                    if item.is_adjustment_layer:
                        payload = {"type": "fx_layer", "fx_type": item.fx_type}
                    elif item.asset:
                        payload = {"type": "media", "asset_id": item.asset.asset_id}
                    else:
                        payload = {}
                    mime.setData("application/x-dlss-timeline-asset", QByteArray(json.dumps(payload).encode("utf-8")))
                    drag.setMimeData(mime)
                    drag.setPixmap(item.icon().pixmap(48, 30))
                    drag.exec(Qt.DropAction.CopyAction)
                    return
        super().mouseMoveEvent(event)


class MediaPoolWidget(QFrame):
    """Media Pool widget allowing users to import, preview, and drag clips and FX into timeline."""

    assetDoubleClicked = pyqtSignal(MediaAsset)
    addAdjustmentLayerRequested = pyqtSignal(str)  # fx_type ("dlss5" or "reshade")

    def __init__(self, frame_cache: VideoFrameCache | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("mediaPoolWidget")
        self.frame_cache = frame_cache or VideoFrameCache()
        self.assets: dict[str, MediaAsset] = {}

        self.setStyleSheet(
            "QFrame#mediaPoolWidget {"
            "  background-color: #121316;"
            "  border-right: 1px solid #23252a;"
            "}"
            "QListWidget {"
            "  background-color: #16171b;"
            "  border: 1px solid #26282e;"
            "  border-radius: 4px;"
            "  color: #e2e8f0;"
            "  font-size: 11px;"
            "}"
            "QListWidget::item {"
            "  padding: 6px;"
            "  border-bottom: 1px solid #1f2025;"
            "}"
            "QListWidget::item:selected {"
            "  background-color: #1e293b;"
            "  color: #38bdf8;"
            "}"
            "QListWidget::item:hover {"
            "  background-color: #1a1b20;"
            "}"
        )

        self.setAcceptDrops(True)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Header Title
        lbl_title = QLabel("Project Media Bin")
        lbl_title.setStyleSheet("font-weight: 700; font-size: 12px; color: #f8fafc;")
        layout.addWidget(lbl_title)

        # Actions Toolbar
        btn_bar = QHBoxLayout()
        btn_bar.setSpacing(4)

        self.btn_import = QPushButton("Import Media")
        self.btn_import.setStyleSheet(
            "QPushButton { background: #2563eb; color: white; border-radius: 4px; padding: 5px 8px; font-weight: 600; font-size: 11px; }"
            "QPushButton:hover { background: #1d4ed8; }"
        )
        self.btn_import.clicked.connect(self._on_import_clicked)
        btn_bar.addWidget(self.btn_import)

        self.btn_dlss_layer = QPushButton("+ DLSS 5")
        self.btn_dlss_layer.setToolTip("Add DLSS 5 Neural Adjustment Layer")
        self.btn_dlss_layer.setStyleSheet(
            "QPushButton { background: #7c3aed; color: white; border-radius: 4px; padding: 5px 8px; font-weight: 600; font-size: 11px; }"
            "QPushButton:hover { background: #6d28d9; }"
        )
        self.btn_dlss_layer.clicked.connect(lambda: self.addAdjustmentLayerRequested.emit("dlss5"))
        btn_bar.addWidget(self.btn_dlss_layer)

        self.btn_reshade_layer = QPushButton("+ ReShade")
        self.btn_reshade_layer.setToolTip("Add ReShade Cinematic FX Layer")
        self.btn_reshade_layer.setStyleSheet(
            "QPushButton { background: #0284c7; color: white; border-radius: 4px; padding: 5px 8px; font-weight: 600; font-size: 11px; }"
            "QPushButton:hover { background: #0369a1; }"
        )
        self.btn_reshade_layer.clicked.connect(lambda: self.addAdjustmentLayerRequested.emit("reshade"))
        btn_bar.addWidget(self.btn_reshade_layer)

        layout.addLayout(btn_bar)

        # Assets List
        self.list_widget = DraggableMediaListWidget(self)
        self.list_widget.setIconSize(QSize(64, 40))
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list_widget.setDragEnabled(True)
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.list_widget)

        # Add generator items at top of list
        dlss_item = MediaAssetListItem(None, is_adjustment_layer=True, fx_type="dlss5")
        dlss_item.setIcon(QIcon(self._generate_fx_icon("#7c3aed", "DLSS 5\nLAYER")))
        self.list_widget.addItem(dlss_item)

        reshade_item = MediaAssetListItem(None, is_adjustment_layer=True, fx_type="reshade")
        reshade_item.setIcon(QIcon(self._generate_fx_icon("#0284c7", "RESHADE\nFX")))
        self.list_widget.addItem(reshade_item)

    def _generate_fx_icon(self, hex_color: str, label: str) -> QPixmap:
        pix = QPixmap(64, 40)
        pix.fill(QColor(hex_color))
        painter = QPainter(pix)
        painter.setPen(QColor("#ffffff"))
        font = painter.font()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, label)
        painter.end()
        return pix

    def _on_import_clicked(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Import Media Files",
            "",
            "Video & Image Files (*.mp4 *.mkv *.mov *.avi *.webm *.png *.jpg *.jpeg *.bmp *.webp *.tiff);;All Files (*.*)",
        )
        for f in files:
            self.import_file(f)

    def import_file(self, file_path: str) -> MediaAsset | None:
        try:
            if not file_path or not os.path.exists(file_path):
                return None

            norm_target = os.path.normcase(os.path.abspath(file_path))
            for asset in self.assets.values():
                if os.path.normcase(os.path.abspath(asset.file_path)) == norm_target:
                    return asset

            meta = self.frame_cache.get_media_metadata(file_path)
            if not meta:
                return None

            total_frames, duration_sec, fps, width, height = meta
            asset = MediaAsset.create(
                file_path=file_path,
                duration_frames=total_frames,
                duration_sec=duration_sec,
                fps=fps,
                width=width,
                height=height,
            )
            self.assets[asset.asset_id] = asset

            thumb_bgr = self.frame_cache.generate_thumbnail(file_path, (64, 40))
            item = MediaAssetListItem(asset, is_adjustment_layer=False)
            if thumb_bgr is not None and thumb_bgr.size > 0:
                rgb = cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
                item.setIcon(QIcon(QPixmap.fromImage(qimg)))
            else:
                default_pix = QPixmap(64, 40)
                default_pix.fill(QColor("#1e293b"))
                item.setIcon(QIcon(default_pix))

            self.list_widget.addItem(item)
            return asset
        except Exception:
            return None

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        if isinstance(item, MediaAssetListItem):
            if item.is_adjustment_layer:
                self.addAdjustmentLayerRequested.emit(item.fx_type)
            elif item.asset:
                self.assetDoubleClicked.emit(item.asset)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        urls = event.mimeData().urls()
        for url in urls:
            fpath = url.toLocalFile()
            if fpath:
                self.import_file(fpath)
