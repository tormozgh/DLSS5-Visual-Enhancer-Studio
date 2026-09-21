"""Clip Inspector and DLSS 5 Effect Controls Widget."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ....timeline.models import DLSSConfig, TimelineClip


class ClipInspectorWidget(QFrame):
    """Inspector panel displaying clip metadata and DLSS 5 Neural Adjustment Layer parameters."""

    parametersChanged = pyqtSignal(object)  # TimelineClip

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("clipInspectorWidget")
        self._current_clip: TimelineClip | None = None
        self._is_updating_ui = False

        self.setStyleSheet(
            "QFrame#clipInspectorWidget {"
            "  background-color: #121316;"
            "  border-left: 1px solid #23252a;"
            "}"
            "QGroupBox {"
            "  color: #e2e8f0;"
            "  font-size: 11px;"
            "  font-weight: 700;"
            "  border: 1px solid #262932;"
            "  border-radius: 4px;"
            "  margin-top: 10px;"
            "  padding-top: 10px;"
            "}"
            "QGroupBox::title {"
            "  subcontrol-origin: margin;"
            "  subcontrol-position: top left;"
            "  padding: 0 5px;"
            "  color: #38bdf8;"
            "}"
            "QLabel { color: #94a3b8; font-size: 11px; }"
            "QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {"
            "  background-color: #1a1c22;"
            "  color: #f8fafc;"
            "  border: 1px solid #2d313b;"
            "  border-radius: 4px;"
            "  padding: 4px;"
            "  font-size: 11px;"
            "}"
            "QSlider::groove:horizontal { height: 4px; background: #262932; border-radius: 2px; }"
            "QSlider::sub-page:horizontal { background: #7c3aed; border-radius: 2px; }"
            "QSlider::handle:horizontal { width: 12px; height: 12px; margin: -4px 0; background: #c084fc; border-radius: 6px; }"
        )

        self._init_ui()

    def _init_ui(self) -> None:
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(8, 8, 8, 8)
        self.main_layout.setSpacing(8)

        # Header Title
        self.lbl_title = QLabel("Clip & Effect Inspector")
        self.lbl_title.setStyleSheet("font-weight: 700; font-size: 12px; color: #f8fafc;")
        self.main_layout.addWidget(self.lbl_title)

        # Empty State Message
        self.lbl_empty = QLabel("Select a clip or DLSS 5 Adjustment Layer on the timeline to edit its properties.")
        self.lbl_empty.setWordWrap(True)
        self.lbl_empty.setStyleSheet("color: #64748b; font-style: italic; padding: 20px 0;")
        self.main_layout.addWidget(self.lbl_empty)

        # Container for controls
        self.container = QWidget()
        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(8)

        # 1. General Clip Properties Group
        grp_clip = QGroupBox("General Properties")
        form_clip = QFormLayout(grp_clip)
        form_clip.setContentsMargins(8, 8, 8, 8)
        form_clip.setSpacing(6)

        self.txt_clip_name = QLineEdit()
        self.txt_clip_name.textChanged.connect(self._on_name_changed)
        form_clip.addRow("Clip Name:", self.txt_clip_name)

        self.lbl_clip_type = QLabel("Media")
        form_clip.addRow("Type:", self.lbl_clip_type)

        self.lbl_clip_duration = QLabel("0 frames")
        form_clip.addRow("Duration:", self.lbl_clip_duration)

        # Opacity slider
        self.slider_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_opacity.setRange(0, 100)
        self.slider_opacity.setValue(100)
        self.slider_opacity.valueChanged.connect(self._on_opacity_changed)
        self.lbl_opacity_val = QLabel("100%")
        row_op = QHBoxLayout()
        row_op.addWidget(self.slider_opacity)
        row_op.addWidget(self.lbl_opacity_val)
        form_clip.addRow("Opacity:", row_op)

        container_layout.addWidget(grp_clip)

        # 2. DLSS 5 Neural Enhancement Group
        self.grp_dlss = QGroupBox("DLSS 5 Neural Engine Parameters")
        form_dlss = QFormLayout(self.grp_dlss)
        form_dlss.setContentsMargins(8, 8, 8, 8)
        form_dlss.setSpacing(6)

        self.chk_dlss_enable = QCheckBox("Enable DLSS 5 Enhancement")
        self.chk_dlss_enable.setStyleSheet("color: #38bdf8; font-weight: 600;")
        self.chk_dlss_enable.toggled.connect(self._on_dlss_toggled)
        form_dlss.addRow(self.chk_dlss_enable)

        # Preset
        self.cmb_preset = QComboBox()
        self.cmb_preset.addItems(["Ultra Quality", "Quality", "Balanced", "Performance", "Ultra Performance"])
        self.cmb_preset.currentIndexChanged.connect(self._on_params_changed)
        form_dlss.addRow("Preset:", self.cmb_preset)

        # Scale Factor
        self.spn_scale = QDoubleSpinBox()
        self.spn_scale.setRange(1.0, 4.0)
        self.spn_scale.setSingleStep(0.1)
        self.spn_scale.setValue(2.0)
        self.spn_scale.valueChanged.connect(self._on_params_changed)
        form_dlss.addRow("Scale Factor:", self.spn_scale)

        # Sharpness
        self.slider_sharpness = QSlider(Qt.Orientation.Horizontal)
        self.slider_sharpness.setRange(0, 100)
        self.slider_sharpness.setValue(65)
        self.slider_sharpness.valueChanged.connect(self._on_params_changed)
        self.lbl_sharp_val = QLabel("65%")
        row_sh = QHBoxLayout()
        row_sh.addWidget(self.slider_sharpness)
        row_sh.addWidget(self.lbl_sharp_val)
        form_dlss.addRow("Sharpness:", row_sh)

        # Denoise
        self.slider_denoise = QSlider(Qt.Orientation.Horizontal)
        self.slider_denoise.setRange(0, 100)
        self.slider_denoise.setValue(40)
        self.slider_denoise.valueChanged.connect(self._on_params_changed)
        self.lbl_denoise_val = QLabel("40%")
        row_dn = QHBoxLayout()
        row_dn.addWidget(self.slider_denoise)
        row_dn.addWidget(self.lbl_denoise_val)
        form_dlss.addRow("AI Denoise:", row_dn)

        # Cinematic 3D LUT
        self.chk_tone = QCheckBox("Cinematic Tone & Color Grading")
        self.chk_tone.setChecked(True)
        self.chk_tone.toggled.connect(self._on_params_changed)
        form_dlss.addRow(self.chk_tone)

        self.cmb_lut = QComboBox()
        self.cmb_lut.addItems([
            "Cinematic Teal & Orange",
            "Warm Golden Hour",
            "Bleach Bypass",
            "Technicolor 3-Strip",
            "Cyberpunk Neon",
            "Neutral Broadcast",
        ])
        self.cmb_lut.currentIndexChanged.connect(self._on_params_changed)
        form_dlss.addRow("Color Profile:", self.cmb_lut)

        # HDR Dynamic Range Boost
        self.slider_hdr = QSlider(Qt.Orientation.Horizontal)
        self.slider_hdr.setRange(0, 100)
        self.slider_hdr.setValue(35)
        self.slider_hdr.valueChanged.connect(self._on_params_changed)
        self.lbl_hdr_val = QLabel("0.35")
        row_hdr = QHBoxLayout()
        row_hdr.addWidget(self.slider_hdr)
        row_hdr.addWidget(self.lbl_hdr_val)
        form_dlss.addRow("HDR Boost:", row_hdr)

        container_layout.addWidget(self.grp_dlss)
        container_layout.addStretch()

        self.main_layout.addWidget(self.container)
        self.container.setVisible(False)

    def inspect_clip(self, clip: TimelineClip | None) -> None:
        """Bind selected clip to inspector controls."""
        self._current_clip = clip
        if clip is None:
            self.lbl_empty.setVisible(True)
            self.container.setVisible(False)
            return

        self._is_updating_ui = True
        self.lbl_empty.setVisible(False)
        self.container.setVisible(True)

        self.txt_clip_name.setText(clip.name)
        is_adj = clip.clip_type == "adjustment_layer"
        self.lbl_clip_type.setText("DLSS 5 Adjustment Layer" if is_adj else "Media Video Clip")
        self.lbl_clip_type.setStyleSheet("color: #c084fc; font-weight: bold;" if is_adj else "color: #38bdf8; font-weight: bold;")
        self.lbl_clip_duration.setText(f"{clip.duration_frames} frames ({clip.timeline_in} -> {clip.timeline_out})")

        op_int = int(clip.opacity * 100.0)
        self.slider_opacity.setValue(op_int)
        self.lbl_opacity_val.setText(f"{op_int}%")

        # Configure DLSS settings
        cfg = clip.dlss_params
        self.chk_dlss_enable.setChecked(cfg.enabled)
        preset_idx = self.cmb_preset.findText(cfg.preset)
        if preset_idx >= 0:
            self.cmb_preset.setCurrentIndex(preset_idx)
        self.spn_scale.setValue(cfg.scale_factor)

        self.slider_sharpness.setValue(int(cfg.sharpness))
        self.lbl_sharp_val.setText(f"{int(cfg.sharpness)}%")

        self.slider_denoise.setValue(int(cfg.denoise))
        self.lbl_denoise_val.setText(f"{int(cfg.denoise)}%")

        self.chk_tone.setChecked(cfg.cinematic_tone)
        lut_idx = self.cmb_lut.findText(cfg.reshade_preset)
        if lut_idx >= 0:
            self.cmb_lut.setCurrentIndex(lut_idx)

        hdr_int = int(cfg.hdr_boost * 100.0)
        self.slider_hdr.setValue(hdr_int)
        self.lbl_hdr_val.setText(f"{cfg.hdr_boost:.2f}")

        self._is_updating_ui = False

    def _on_name_changed(self, text: str) -> None:
        if self._current_clip and not self._is_updating_ui:
            self._current_clip.name = text
            self.parametersChanged.emit(self._current_clip)

    def _on_opacity_changed(self, val: int) -> None:
        self.lbl_opacity_val.setText(f"{val}%")
        if self._current_clip and not self._is_updating_ui:
            self._current_clip.opacity = val / 100.0
            self.parametersChanged.emit(self._current_clip)

    def _on_dlss_toggled(self, checked: bool) -> None:
        if self._current_clip and not self._is_updating_ui:
            self._current_clip.dlss_params.enabled = checked
            self.parametersChanged.emit(self._current_clip)

    def _on_params_changed(self) -> None:
        if not self._current_clip or self._is_updating_ui:
            return

        cfg = self._current_clip.dlss_params
        cfg.preset = self.cmb_preset.currentText()
        cfg.scale_factor = self.spn_scale.value()
        cfg.sharpness = float(self.slider_sharpness.value())
        cfg.denoise = float(self.slider_denoise.value())
        cfg.cinematic_tone = self.chk_tone.isChecked()
        cfg.reshade_preset = self.cmb_lut.currentText()
        cfg.hdr_boost = self.slider_hdr.value() / 100.0

        self.lbl_sharp_val.setText(f"{int(cfg.sharpness)}%")
        self.lbl_denoise_val.setText(f"{int(cfg.denoise)}%")
        self.lbl_hdr_val.setText(f"{cfg.hdr_boost:.2f}")

        self.parametersChanged.emit(self._current_clip)
