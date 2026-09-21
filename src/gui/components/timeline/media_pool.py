"""Media Pool (Project Bin) widget for importing and managing timeline assets."""

from __future__ import annotations

import json
import os
from typing import Callable

from PyQt6.QtCore import QByteArray, QMimeData, QPoint, QSize, Qt, pyqtSignal
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
    """List item representing an imported media asset or adjustment layer generator."""

    def __init__(self, asset: MediaAsset | None, is_adjustment_layer: bool = False) -> None:
        super().__init__()
        self.asset = asset
        self.is_adjustment_layer = is_adjustment_layer

        if is_adjustment_layer:
            self.setText("DLSS 5 Adjustment Layer\n[AI Neural FX Layer]")
            self.setToolTip("Drag onto a higher video track (e.g. V2/V3) to enhance underlying clips")
        elif asset:
            dur_m = int(asset.duration_sec // 60)
            dur_s = int(asset.duration_sec % 60)
            info = f"{asset.name}\n{asset.width}x{asset.height} | {asset.fps:.1f} fps | {dur_m:02d}:{dur_s:02d}"
            self.setText(info)
            self.setToolTip(f"File: {asset.file_path}\nTotal Frames: {asset.duration_frames}")


class MediaPoolWidget(QFrame):
    """Media Pool widget allowing users to import, preview, and drag clips into timeline."""

    assetDoubleClicked = pyqtSignal(MediaAsset)
    addAdjustmentLayerRequested = pyqtSignal()

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
        lbl_title = QLabel("Media Pool & Project Assets")
        lbl_title.setStyleSheet("font-weight: 700; font-size: 12px; color: #f8fafc;")
        layout.addWidget(lbl_title)

        # Actions Toolbar
        btn_bar = QHBoxLayout()
        btn_bar.setSpacing(6)

        self.btn_import = QPushButton("Import Media")
        self.btn_import.setStyleSheet(
            "QPushButton { background: #2563eb; color: white; border-radius: 4px; padding: 5px 10px; font-weight: 600; font-size: 11px; }"
            "QPushButton:hover { background: #1d4ed8; }"
        )
        self.btn_import.clicked.connect(self._on_import_clicked)
        btn_bar.addWidget(self.btn_import)

        self.btn_adj_layer = QPushButton("+ DLSS 5 Adjustment Layer")
        self.btn_adj_layer.setStyleSheet(
            "QPushButton { background: #7c3aed; color: white; border-radius: 4px; padding: 5px 10px; font-weight: 600; font-size: 11px; }"
            "QPushButton:hover { background: #6d28d9; }"
        )
        self.btn_adj_layer.clicked.connect(self._on_create_adjustment_layer_clicked)
        btn_bar.addWidget(self.btn_adj_layer)

        layout.addLayout(btn_bar)

        # Assets List
        self.list_widget = QListWidget()
        self.list_widget.setIconSize(QSize(64, 40))
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list_widget.setDragEnabled(True)
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.list_widget)

        # Add permanent Adjustment Layer generator item at top of list
        adj_item = MediaAssetListItem(None, is_adjustment_layer=True)
        adj_icon_pixmap = self._generate_adjustment_icon()
        adj_item.setIcon(QIcon(adj_icon_pixmap))
        self.list_widget.addItem(adj_item)

    def _generate_adjustment_icon(self) -> QPixmap:
        pix = QPixmap(64, 40)
        pix.fill(QColor("#7c3aed"))
        painter = QPainter(pix)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, "DLSS 5\nLAYER")
        painter.end()
        return pix

    def _on_import_clicked(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Import Media Files",
            "",
            "Video & Image Files (*.mp4 *.mkv *.mov *.avi *.webm *.png *.jpg *.jpeg);;All Files (*.*)",
        )
        for f in files:
            self.import_file(f)

    def import_file(self, file_path: str) -> MediaAsset | None:
        if not os.path.exists(file_path):
            return None

        # Check if already imported
        for asset in self.assets.values():
            if os.path.abspath(asset.file_path) == os.path.abspath(file_path):
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

        # Generate thumbnail
        thumb_bgr = self.frame_cache.generate_thumbnail(file_path, (64, 40))
        item = MediaAssetListItem(asset, is_adjustment_layer=False)
        if thumb_bgr is not None:
            rgb = cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            item.setIcon(QIcon(QPixmap.fromImage(qimg)))
        else:
            pix = QPixmap(64, 40)
            pix.fill(QColor("#2563eb"))
            item.setIcon(QIcon(pix))

        self.list_widget.addItem(item)
        return asset

    def _on_create_adjustment_layer_clicked(self) -> None:
        self.addAdjustmentLayerRequested.emit()

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        if isinstance(item, MediaAssetListItem):
            if item.is_adjustment_layer:
                self.addAdjustmentLayerRequested.emit()
            elif item.asset:
                self.assetDoubleClicked.emit(item.asset)

    # Drag and Drop support
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            item = self.list_widget.itemAt(self.list_widget.mapFrom(self, event.pos()))
            if isinstance(item, MediaAssetListItem):
                drag = QDrag(self)
                mime = QMimeData()
                if item.is_adjustment_layer:
                    payload = {"type": "adjustment_layer"}
                else:
                    payload = {"type": "media", "asset_id": item.asset.asset_id}
                mime.setData("application/x-dlss-timeline-asset", QByteArray(json.dumps(payload).encode("utf-8")))
                drag.setMimeData(mime)
                drag.setPixmap(item.icon().pixmap(48, 30))
                drag.exec(Qt.DropAction.CopyAction)
                return
        super().mousePressEvent(event)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        urls = event.mimeData().urls()
        for url in urls:
            path = url.toLocalFile()
            if path and os.path.exists(path):
                self.import_file(path)
        event.acceptProposedAction()
