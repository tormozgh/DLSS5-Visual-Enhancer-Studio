"""Clip Inspector and DLSS 5 / ReShade FX Effect Controls Widget."""

from __future__ import annotations

import json
import os
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ....timeline.models import DLSSConfig, ReShadeConfig, TimelineClip
from ..sliders import LabeledSlider


class ClipInspectorWidget(QFrame):
    """Inspector panel displaying footage properties or extensible FX Layer (DLSS 5 / ReShade) controls."""

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
            "  font-size: 11px;"
            "  font-weight: 600;"
            "}"
            "QLabel { color: #94a3b8; font-size: 11px; }"
            "QComboBox, QLineEdit {"
            "  background-color: #1a1c22;"
            "  color: #f8fafc;"
            "  border: 1px solid #2d313b;"
            "  border-radius: 4px;"
            "  padding: 4px 6px;"
            "  font-size: 11px;"
            "}"
            "QComboBox:focus, QLineEdit:focus {"
            "  border-color: #38bdf8;"
            "}"
            "QSlider::groove:horizontal { height: 4px; background: #262932; border-radius: 2px; }"
            "QSlider::sub-page:horizontal { background: #3b82f6; border-radius: 2px; }"
            "QSlider::handle:horizontal { width: 12px; height: 12px; margin: -4px 0; background: #93c5fd; border-radius: 6px; }"
            "QPushButton {"
            "  background: #1a1d24;"
            "  color: #e2e8f0;"
            "  border: 1px solid #2d313b;"
            "  border-radius: 4px;"
            "  padding: 5px 8px;"
            "  font-size: 11px;"
            "  font-weight: 600;"
            "}"
            "QPushButton:hover {"
            "  background: #262a35;"
            "  border-color: #38bdf8;"
            "}"
            "QPushButton:pressed {"
            "  background: #13151a;"
            "}"
            "QCheckBox { color: #cbd5e1; font-size: 11px; spacing: 6px; }"
            "QCheckBox::indicator { width: 14px; height: 14px; border-radius: 3px; border: 1px solid #334155; background: #1a1c22; }"
            "QCheckBox::indicator:checked { background: #2563eb; border-color: #38bdf8; }"
        )

        self._init_ui()

    def _init_ui(self) -> None:
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(8, 8, 8, 8)
        self.main_layout.setSpacing(6)

        # Header Title
        self.lbl_title = QLabel("Clip & Effect Inspector")
        self.lbl_title.setStyleSheet("font-weight: 700; font-size: 12px; color: #f8fafc;")
        self.main_layout.addWidget(self.lbl_title)

        # Empty State Message
        self.lbl_empty = QLabel("Select a clip or FX Layer on the timeline to edit its properties.")
        self.lbl_empty.setWordWrap(True)
        self.lbl_empty.setStyleSheet("color: #64748b; font-style: italic; padding: 20px 0;")
        self.main_layout.addWidget(self.lbl_empty)

        # Scroll Area for the extensive Inspector controls
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        self.container = QWidget()
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(0, 0, 4, 0)
        self.container_layout.setSpacing(10)

        # =========================================================================
        # SECTION A: Media Footage Properties (Visible ONLY for Footage Clips)
        # =========================================================================
        self.grp_media_props = QGroupBox("Footage Properties")
        media_layout = QVBoxLayout(self.grp_media_props)
        media_layout.setContentsMargins(8, 8, 8, 8)
        media_layout.setSpacing(6)

        row_m_name = QHBoxLayout()
        row_m_name.addWidget(QLabel("Clip Name:"))
        self.txt_media_name = QLineEdit()
        self.txt_media_name.textChanged.connect(self._on_media_name_changed)
        row_m_name.addWidget(self.txt_media_name, 1)
        media_layout.addLayout(row_m_name)

        row_m_file = QHBoxLayout()
        row_m_file.addWidget(QLabel("Source File:"))
        self.lbl_media_file = QLabel("None")
        self.lbl_media_file.setStyleSheet("color: #cbd5e1; font-family: monospace; font-size: 10px;")
        row_m_file.addWidget(self.lbl_media_file, 1)
        media_layout.addLayout(row_m_file)

        row_m_res = QHBoxLayout()
        row_m_res.addWidget(QLabel("Native Resolution:"))
        self.lbl_media_res = QLabel("1920x1080 @ 30.0 fps")
        self.lbl_media_res.setStyleSheet("color: #38bdf8; font-weight: 600;")
        row_m_res.addWidget(self.lbl_media_res, 1)
        media_layout.addLayout(row_m_res)

        row_m_dur = QHBoxLayout()
        row_m_dur.addWidget(QLabel("Duration & Timing:"))
        self.lbl_media_duration = QLabel("0 frames")
        row_m_dur.addWidget(self.lbl_media_duration, 1)
        media_layout.addLayout(row_m_dur)

        # Scaling / Fitting mode dropdown
        row_scale_mode = QHBoxLayout()
        lbl_sm = QLabel("Scale Mode:")
        self.combo_scale_mode = QComboBox()
        self.combo_scale_mode.addItem("Fit to Frame (Letterbox)", "fit")
        self.combo_scale_mode.addItem("Fill Frame (Crop)", "fill")
        self.combo_scale_mode.addItem("Stretch to Fill", "stretch")
        self.combo_scale_mode.currentIndexChanged.connect(self._on_scale_mode_changed)
        row_scale_mode.addWidget(lbl_sm)
        row_scale_mode.addWidget(self.combo_scale_mode, 1)
        media_layout.addLayout(row_scale_mode)

        # Footage Opacity
        row_m_op = QHBoxLayout()
        row_m_op.addWidget(QLabel("Opacity:"))
        self.slider_media_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_media_opacity.setRange(0, 100)
        self.slider_media_opacity.setValue(100)
        self.slider_media_opacity.valueChanged.connect(self._on_media_opacity_changed)
        row_m_op.addWidget(self.slider_media_opacity, 1)
        self.lbl_media_opacity_val = QLabel("100%")
        self.lbl_media_opacity_val.setFixedWidth(36)
        row_m_op.addWidget(self.lbl_media_opacity_val)
        media_layout.addLayout(row_m_op)

        self.container_layout.addWidget(self.grp_media_props)

        # =========================================================================
        # SECTION B: FX Layer Properties & Effect Selector (Visible for FX Layers)
        # =========================================================================
        self.grp_fx_props = QGroupBox("FX Adjustment Layer")
        fx_layout = QVBoxLayout(self.grp_fx_props)
        fx_layout.setContentsMargins(8, 8, 8, 8)
        fx_layout.setSpacing(6)

        row_fx_name = QHBoxLayout()
        row_fx_name.addWidget(QLabel("Layer Name:"))
        self.txt_fx_name = QLineEdit()
        self.txt_fx_name.textChanged.connect(self._on_fx_name_changed)
        row_fx_name.addWidget(self.txt_fx_name, 1)
        fx_layout.addLayout(row_fx_name)

        # Effect Type Selector (DLSS 5 vs ReShade)
        row_fx_type = QHBoxLayout()
        lbl_fx_sel = QLabel("Effect Type:")
        lbl_fx_sel.setStyleSheet("font-weight: bold; color: #38bdf8;")
        self.combo_fx_type = QComboBox()
        self.combo_fx_type.addItem("DLSS 5 Neural Rendering", "dlss5")
        self.combo_fx_type.addItem("ReShade FX & Cinematic Grading", "reshade")
        self.combo_fx_type.currentIndexChanged.connect(self._on_fx_type_changed)
        row_fx_type.addWidget(lbl_fx_sel)
        row_fx_type.addWidget(self.combo_fx_type, 1)
        fx_layout.addLayout(row_fx_type)

        # FX Opacity
        row_fx_op = QHBoxLayout()
        row_fx_op.addWidget(QLabel("Layer Opacity:"))
        self.slider_fx_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_fx_opacity.setRange(0, 100)
        self.slider_fx_opacity.setValue(100)
        self.slider_fx_opacity.valueChanged.connect(self._on_fx_opacity_changed)
        row_fx_op.addWidget(self.slider_fx_opacity, 1)
        self.lbl_fx_opacity_val = QLabel("100%")
        self.lbl_fx_opacity_val.setFixedWidth(36)
        row_fx_op.addWidget(self.lbl_fx_opacity_val)
        fx_layout.addLayout(row_fx_op)

        self.container_layout.addWidget(self.grp_fx_props)

        # =========================================================================
        # SECTION C: DLSS 5 Neural Rendering Controls (Matching Image 1)
        # =========================================================================
        # Card 1: Presets Settings
        self.grp_presets = QGroupBox("Presets Settings")
        presets_layout = QVBoxLayout(self.grp_presets)
        presets_layout.setContentsMargins(8, 8, 8, 8)
        presets_layout.setSpacing(6)

        row_quick = QHBoxLayout()
        row_quick.setSpacing(6)
        self.btn_preset_default = QPushButton("Default")
        self.btn_preset_default.clicked.connect(self._preset_default)
        self.btn_preset_detail = QPushButton("Detail-Only")
        self.btn_preset_detail.clicked.connect(self._preset_detail_only)
        self.btn_reset_all = QPushButton("Reset All")
        self.btn_reset_all.clicked.connect(self._reset_all)
        row_quick.addWidget(self.btn_preset_default)
        row_quick.addWidget(self.btn_preset_detail)
        row_quick.addWidget(self.btn_reset_all)
        presets_layout.addLayout(row_quick)

        row_save_load = QHBoxLayout()
        row_save_load.setSpacing(6)
        self.btn_save_settings = QPushButton("Save Settings...")
        self.btn_save_settings.clicked.connect(self._on_save_settings)
        self.btn_load_settings = QPushButton("Load Settings...")
        self.btn_load_settings.clicked.connect(self._on_load_settings)
        row_save_load.addWidget(self.btn_save_settings)
        row_save_load.addWidget(self.btn_load_settings)
        presets_layout.addLayout(row_save_load)

        self.container_layout.addWidget(self.grp_presets)

        # Card 2: DLSS 5 Core
        self.grp_dlss_core = QGroupBox("DLSS 5 Neural Rendering")
        dlss_layout = QVBoxLayout(self.grp_dlss_core)
        dlss_layout.setContentsMargins(8, 8, 8, 8)
        dlss_layout.setSpacing(6)

        row_style = QHBoxLayout()
        lbl_style = QLabel("NR Style:")
        lbl_style.setStyleSheet("font-weight: 600; color: #f8fafc;")
        self.combo_style = QComboBox()
        self.combo_style.addItems(["Default", "Natural", "Cinematic"])
        self.combo_style.currentTextChanged.connect(self._on_style_changed)
        row_style.addWidget(lbl_style)
        row_style.addWidget(self.combo_style, 1)
        dlss_layout.addLayout(row_style)

        row_scale = QHBoxLayout()
        lbl_scale = QLabel("Scale:")
        lbl_scale.setStyleSheet("font-weight: 600; color: #f8fafc;")
        self.combo_scale = QComboBox()
        self.combo_scale.addItem("Source (100%)", 1.0)
        self.combo_scale.addItem("75%", 0.75)
        self.combo_scale.addItem("50%", 0.50)
        self.combo_scale.addItem("25%", 0.25)
        self.combo_scale.currentIndexChanged.connect(self._on_scale_changed)
        row_scale.addWidget(lbl_scale)
        row_scale.addWidget(self.combo_scale, 1)
        dlss_layout.addLayout(row_scale)

        self.slider_intensity = LabeledSlider("NR Intensity", 0.0, 2.0, 1.0, step=0.05, decimals=2)
        self.slider_intensity.valueChanged.connect(self._on_dlss_slider_changed)
        dlss_layout.addWidget(self.slider_intensity)

        self.slider_passes = LabeledSlider("NR Passes", 1.0, 4.0, 1.0, step=1.0, decimals=0)
        self.slider_passes.valueChanged.connect(self._on_dlss_slider_changed)
        dlss_layout.addWidget(self.slider_passes)

        self.container_layout.addWidget(self.grp_dlss_core)

        # Card 3: Tone Structure
        self.grp_tone = QGroupBox("Tone  Structure")
        tone_layout = QVBoxLayout(self.grp_tone)
        tone_layout.setContentsMargins(8, 8, 8, 8)
        tone_layout.setSpacing(6)

        self.slider_tone = LabeledSlider("Local Tone Strength", 0.0, 2.0, 1.0, step=0.05, decimals=2)
        self.slider_tone.valueChanged.connect(self._on_dlss_slider_changed)
        tone_layout.addWidget(self.slider_tone)

        self.slider_structure = LabeledSlider("Local Structure Strength", 0.0, 2.0, 1.0, step=0.05, decimals=2)
        self.slider_structure.valueChanged.connect(self._on_dlss_slider_changed)
        tone_layout.addWidget(self.slider_structure)

        self.slider_skin = LabeledSlider("Skin Structure Strength", -1.0, 1.0, -1.0, step=0.05, decimals=2)
        self.slider_skin.valueChanged.connect(self._on_dlss_slider_changed)
        tone_layout.addWidget(self.slider_skin)

        self.container_layout.addWidget(self.grp_tone)

        # Card 4: Neural Composition
        self.grp_comp = QGroupBox("Neural Composition")
        comp_layout = QVBoxLayout(self.grp_comp)
        comp_layout.setContentsMargins(8, 8, 8, 8)
        comp_layout.setSpacing(6)

        self.slider_color_strength = LabeledSlider("NR Color Strength", 0.0, 1.0, 1.0, step=0.05, decimals=2)
        self.slider_color_strength.valueChanged.connect(self._on_dlss_slider_changed)
        comp_layout.addWidget(self.slider_color_strength)

        self.slider_tone_preservation = LabeledSlider("Tone Preservation", 0.0, 1.0, 0.0, step=0.05, decimals=2)
        self.slider_tone_preservation.valueChanged.connect(self._on_dlss_slider_changed)
        comp_layout.addWidget(self.slider_tone_preservation)

        self.slider_face_protection = LabeledSlider("Face/Skin Protection", 0.0, 1.0, 0.0, step=0.05, decimals=2)
        self.slider_face_protection.valueChanged.connect(self._on_dlss_slider_changed)
        comp_layout.addWidget(self.slider_face_protection)

        self.slider_grain = LabeledSlider("Grain Preservation", 0.0, 1.0, 0.0, step=0.05, decimals=2)
        self.slider_grain.valueChanged.connect(self._on_dlss_slider_changed)
        comp_layout.addWidget(self.slider_grain)

        self.slider_feather = LabeledSlider("Mask Feather", 0.0, 128.0, 0.0, step=1.0, decimals=0, suffix=" px")
        self.slider_feather.valueChanged.connect(self._on_dlss_slider_changed)
        comp_layout.addWidget(self.slider_feather)

        self.chk_auto_mask = QCheckBox("Automatic Mask")
        self.chk_auto_mask.toggled.connect(self._on_dlss_slider_changed)
        comp_layout.addWidget(self.chk_auto_mask)

        self.container_layout.addWidget(self.grp_comp)

        # =========================================================================
        # SECTION D: ReShade FX & Cinematic Grading Controls
        # =========================================================================
        self.grp_reshade = QGroupBox("ReShade FX & Cinematic Grading")
        reshade_layout = QVBoxLayout(self.grp_reshade)
        reshade_layout.setContentsMargins(8, 8, 8, 8)
        reshade_layout.setSpacing(6)

        # Quick ReShade Presets
        row_rs_quick = QHBoxLayout()
        self.btn_rs_default = QPushButton("Default")
        self.btn_rs_default.clicked.connect(self._preset_reshade_default)
        self.btn_rs_warm = QPushButton("Warm")
        self.btn_rs_warm.clicked.connect(lambda: self._apply_reshade_preset("Warm Golden Hour", 0.8, 0.1, 1.1, 1.15, 0.2))
        self.btn_rs_cyber = QPushButton("Cyberpunk")
        self.btn_rs_cyber.clicked.connect(lambda: self._apply_reshade_preset("Cyberpunk Neon", 1.0, 0.2, 1.25, 1.35, -0.3))
        self.btn_rs_bleach = QPushButton("Bleach")
        self.btn_rs_bleach.clicked.connect(lambda: self._apply_reshade_preset("Bleach Bypass", 0.9, 0.0, 1.3, 0.6, 0.0))
        row_rs_quick.addWidget(self.btn_rs_default)
        row_rs_quick.addWidget(self.btn_rs_warm)
        row_rs_quick.addWidget(self.btn_rs_cyber)
        row_rs_quick.addWidget(self.btn_rs_bleach)
        reshade_layout.addLayout(row_rs_quick)

        # 3D LUT Selector
        row_lut = QHBoxLayout()
        lbl_lut = QLabel("3D Color LUT:")
        lbl_lut.setStyleSheet("font-weight: 600; color: #f8fafc;")
        self.combo_lut = QComboBox()
        self.combo_lut.addItems([
            "Cinematic Teal & Orange",
            "Warm Golden Hour",
            "Bleach Bypass",
            "Technicolor 3-Strip",
            "Cyberpunk Neon",
            "Neutral Broadcast",
        ])
        self.combo_lut.currentTextChanged.connect(self._on_reshade_param_changed)
        row_lut.addWidget(lbl_lut)
        row_lut.addWidget(self.combo_lut, 1)
        reshade_layout.addLayout(row_lut)

        self.slider_lut_strength = LabeledSlider("LUT Strength", 0.0, 1.0, 0.85, step=0.05)
        self.slider_lut_strength.valueChanged.connect(self._on_reshade_param_changed)
        reshade_layout.addWidget(self.slider_lut_strength)

        # Exposure & Contrast
        self.slider_exposure = LabeledSlider("Exposure", -2.0, 2.0, 0.0, step=0.05, suffix=" EV")
        self.slider_exposure.valueChanged.connect(self._on_reshade_param_changed)
        reshade_layout.addWidget(self.slider_exposure)

        self.slider_contrast = LabeledSlider("Contrast", 0.5, 2.0, 1.05, step=0.05)
        self.slider_contrast.valueChanged.connect(self._on_reshade_param_changed)
        reshade_layout.addWidget(self.slider_contrast)

        self.slider_saturation = LabeledSlider("Saturation", 0.0, 2.0, 1.10, step=0.05)
        self.slider_saturation.valueChanged.connect(self._on_reshade_param_changed)
        reshade_layout.addWidget(self.slider_saturation)

        self.slider_color_temp = LabeledSlider("Color Temperature", -1.0, 1.0, 0.0, step=0.05)
        self.slider_color_temp.valueChanged.connect(self._on_reshade_param_changed)
        reshade_layout.addWidget(self.slider_color_temp)

        # CAS & Film Grain
        self.slider_cas = LabeledSlider("CAS Sharpness", 0.0, 1.0, 0.40, step=0.05)
        self.slider_cas.valueChanged.connect(self._on_reshade_param_changed)
        reshade_layout.addWidget(self.slider_cas)

        self.slider_reshade_grain = LabeledSlider("Film Grain", 0.0, 1.0, 0.18, step=0.02)
        self.slider_reshade_grain.valueChanged.connect(self._on_reshade_param_changed)
        reshade_layout.addWidget(self.slider_reshade_grain)

        self.slider_bloom = LabeledSlider("Bloom / Glow", 0.0, 1.0, 0.0, step=0.05)
        self.slider_bloom.valueChanged.connect(self._on_reshade_param_changed)
        reshade_layout.addWidget(self.slider_bloom)

        self.container_layout.addWidget(self.grp_reshade)

        self.container_layout.addStretch()
        self.scroll_area.setWidget(self.container)
        self.main_layout.addWidget(self.scroll_area)
        self.scroll_area.setVisible(False)

    def inspect_clip(self, clip: TimelineClip | None) -> None:
        """Bind selected clip to inspector controls, routing to Footage vs FX Layer view."""
        self._current_clip = clip
        if clip is None:
            self.lbl_empty.setVisible(True)
            self.scroll_area.setVisible(False)
            return

        self._is_updating_ui = True
        self.lbl_empty.setVisible(False)
        self.scroll_area.setVisible(True)

        is_footage = clip.clip_type == "media"

        if is_footage:
            # Display ONLY Footage Properties
            self.grp_media_props.setVisible(True)
            self.grp_fx_props.setVisible(False)
            self.grp_presets.setVisible(False)
            self.grp_dlss_core.setVisible(False)
            self.grp_tone.setVisible(False)
            self.grp_comp.setVisible(False)
            self.grp_reshade.setVisible(False)

            self.txt_media_name.setText(clip.name)
            self.lbl_media_file.setText(os.path.basename(clip.media_path or "Unknown"))
            self.lbl_media_res.setText(f"{clip.asset_width}x{clip.asset_height} @ {clip.asset_fps:.2f} fps")
            self.lbl_media_duration.setText(f"{clip.duration_frames} frames ({clip.timeline_in} -> {clip.timeline_out})")

            sm_idx = self.combo_scale_mode.findData(clip.scale_mode)
            if sm_idx >= 0:
                self.combo_scale_mode.setCurrentIndex(sm_idx)

            op_int = int(clip.opacity * 100.0)
            self.slider_media_opacity.setValue(op_int)
            self.lbl_media_opacity_val.setText(f"{op_int}%")

        else:
            # Display FX Layer Properties with Effect Type Selector
            self.grp_media_props.setVisible(False)
            self.grp_fx_props.setVisible(True)

            self.txt_fx_name.setText(clip.name)
            op_int = int(clip.opacity * 100.0)
            self.slider_fx_opacity.setValue(op_int)
            self.lbl_fx_opacity_val.setText(f"{op_int}%")

            fx_type = getattr(clip, "fx_type", "dlss5")
            idx = self.combo_fx_type.findData(fx_type)
            if idx >= 0:
                self.combo_fx_type.setCurrentIndex(idx)

            self._update_fx_panel_visibility(fx_type)

            # Load DLSS 5 parameters
            cfg = clip.dlss_params
            style_idx = self.combo_style.findText(cfg.nr_style)
            if style_idx >= 0:
                self.combo_style.setCurrentIndex(style_idx)

            scale_val = cfg.scale
            for i in range(self.combo_scale.count()):
                val = float(self.combo_scale.itemData(i) or 1.0)
                if abs(val - scale_val) < 0.01:
                    self.combo_scale.setCurrentIndex(i)
                    break

            self.slider_intensity.setValue(cfg.nr_intensity)
            self.slider_passes.setValue(float(cfg.nr_passes))
            self.slider_tone.setValue(cfg.local_tone_strength)
            self.slider_structure.setValue(cfg.local_structure_strength)
            self.slider_skin.setValue(cfg.skin_structure_strength)
            self.slider_color_strength.setValue(cfg.nr_color_strength)
            self.slider_tone_preservation.setValue(cfg.tone_preservation)
            self.slider_face_protection.setValue(cfg.face_skin_protection)
            self.slider_grain.setValue(cfg.grain_preservation)
            self.slider_feather.setValue(float(cfg.mask_feather))
            self.chk_auto_mask.setChecked(cfg.automatic_mask)

            # Load ReShade parameters
            rcfg = getattr(clip, "reshade_params", None) or ReShadeConfig()
            lut_idx = self.combo_lut.findText(rcfg.lut_name)
            if lut_idx >= 0:
                self.combo_lut.setCurrentIndex(lut_idx)

            self.slider_lut_strength.setValue(rcfg.lut_strength)
            self.slider_exposure.setValue(rcfg.exposure)
            self.slider_contrast.setValue(rcfg.contrast)
            self.slider_saturation.setValue(rcfg.saturation)
            self.slider_color_temp.setValue(rcfg.color_temperature)
            self.slider_cas.setValue(rcfg.cas_sharpness)
            self.slider_reshade_grain.setValue(rcfg.grain_intensity)
            self.slider_bloom.setValue(rcfg.bloom_intensity)

        self._is_updating_ui = False

    def _update_fx_panel_visibility(self, fx_type: str) -> None:
        """Switch visible parameter cards based on active effect type."""
        is_dlss = fx_type == "dlss5"
        is_reshade = fx_type == "reshade"

        self.grp_presets.setVisible(is_dlss)
        self.grp_dlss_core.setVisible(is_dlss)
        self.grp_tone.setVisible(is_dlss)
        self.grp_comp.setVisible(is_dlss)
        self.grp_reshade.setVisible(is_reshade)

    # Footage Callbacks
    def _on_media_name_changed(self, text: str) -> None:
        if self._current_clip and not self._is_updating_ui:
            self._current_clip.name = text
            self.parametersChanged.emit(self._current_clip)

    def _on_media_opacity_changed(self, val: int) -> None:
        self.lbl_media_opacity_val.setText(f"{val}%")
        if self._current_clip and not self._is_updating_ui:
            self._current_clip.opacity = val / 100.0
            self.parametersChanged.emit(self._current_clip)

    def _on_scale_mode_changed(self, index: int) -> None:
        if self._current_clip and not self._is_updating_ui:
            self._current_clip.scale_mode = self.combo_scale_mode.itemData(index) or "fit"
            self.parametersChanged.emit(self._current_clip)

    # FX Layer Callbacks
    def _on_fx_name_changed(self, text: str) -> None:
        if self._current_clip and not self._is_updating_ui:
            self._current_clip.name = text
            self.parametersChanged.emit(self._current_clip)

    def _on_fx_opacity_changed(self, val: int) -> None:
        self.lbl_fx_opacity_val.setText(f"{val}%")
        if self._current_clip and not self._is_updating_ui:
            self._current_clip.opacity = val / 100.0
            self.parametersChanged.emit(self._current_clip)

    def _on_fx_type_changed(self, index: int) -> None:
        if not self._current_clip or self._is_updating_ui:
            return
        fx_type = self.combo_fx_type.itemData(index) or "dlss5"
        self._current_clip.fx_type = fx_type
        if fx_type == "reshade":
            self._current_clip.color = "#0ea5e9"
            if self._current_clip.name == "DLSS 5 Neural Layer":
                self._current_clip.name = "ReShade FX Layer"
                self.txt_fx_name.setText("ReShade FX Layer")
        else:
            self._current_clip.color = "#9333ea"
            if self._current_clip.name == "ReShade FX Layer":
                self._current_clip.name = "DLSS 5 Neural Layer"
                self.txt_fx_name.setText("DLSS 5 Neural Layer")

        self._update_fx_panel_visibility(fx_type)
        self.parametersChanged.emit(self._current_clip)

    # DLSS 5 Parameter Callbacks
    def _on_style_changed(self, text: str) -> None:
        if self._current_clip and not self._is_updating_ui:
            self._current_clip.dlss_params.nr_style = text
            self.parametersChanged.emit(self._current_clip)

    def _on_scale_changed(self, index: int) -> None:
        if self._current_clip and not self._is_updating_ui:
            scale_val = float(self.combo_scale.itemData(index) or 1.0)
            self._current_clip.dlss_params.scale = scale_val
            self.parametersChanged.emit(self._current_clip)

    def _on_dlss_slider_changed(self) -> None:
        if not self._current_clip or self._is_updating_ui:
            return

        cfg = self._current_clip.dlss_params
        cfg.nr_intensity = self.slider_intensity.value()
        cfg.nr_passes = int(round(self.slider_passes.value()))
        cfg.local_tone_strength = self.slider_tone.value()
        cfg.local_structure_strength = self.slider_structure.value()
        cfg.skin_structure_strength = self.slider_skin.value()
        cfg.nr_color_strength = self.slider_color_strength.value()
        cfg.tone_preservation = self.slider_tone_preservation.value()
        cfg.face_skin_protection = self.slider_face_protection.value()
        cfg.grain_preservation = self.slider_grain.value()
        cfg.mask_feather = int(round(self.slider_feather.value()))
        cfg.automatic_mask = self.chk_auto_mask.isChecked()

        self.parametersChanged.emit(self._current_clip)

    def _preset_default(self) -> None:
        self._is_updating_ui = True
        self.combo_style.setCurrentText("Default")
        self.combo_scale.setCurrentIndex(0)
        self.slider_intensity.setValue(1.0)
        self.slider_passes.setValue(1.0)
        self.slider_tone.setValue(1.0)
        self.slider_structure.setValue(1.0)
        self.slider_skin.setValue(-1.0)
        self.slider_color_strength.setValue(1.0)
        self.slider_tone_preservation.setValue(0.0)
        self.slider_face_protection.setValue(0.0)
        self.slider_grain.setValue(0.0)
        self.slider_feather.setValue(0.0)
        self.chk_auto_mask.setChecked(False)
        self._is_updating_ui = False
        self._on_dlss_slider_changed()

    def _preset_detail_only(self) -> None:
        self._is_updating_ui = True
        self.slider_color_strength.setValue(0.0)
        self.slider_tone_preservation.setValue(1.0)
        self._is_updating_ui = False
        self._on_dlss_slider_changed()

    def _reset_all(self) -> None:
        self._preset_default()

    def _on_save_settings(self) -> None:
        if not self._current_clip:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Neural Rendering Settings",
            os.path.expanduser("~/DLSS5_Preset.json"),
            "JSON Presets (*.json);;All Files (*.*)",
        )
        if not path:
            return
        try:
            data = self._current_clip.dlss_params.to_dict()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            QMessageBox.information(self, "Preset Saved", f"Settings successfully saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save preset: {e}")

    def _on_load_settings(self) -> None:
        if not self._current_clip:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Neural Rendering Settings",
            os.path.expanduser("~/"),
            "JSON Presets (*.json);;All Files (*.*)",
        )
        if not path or not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._current_clip.dlss_params = DLSSConfig.from_dict(data)
            self.inspect_clip(self._current_clip)
            self.parametersChanged.emit(self._current_clip)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load preset: {e}")

    # ReShade Parameter Callbacks
    def _on_reshade_param_changed(self) -> None:
        if not self._current_clip or self._is_updating_ui:
            return

        rcfg = self._current_clip.reshade_params
        rcfg.lut_name = self.combo_lut.currentText()
        rcfg.lut_strength = self.slider_lut_strength.value()
        rcfg.exposure = self.slider_exposure.value()
        rcfg.contrast = self.slider_contrast.value()
        rcfg.saturation = self.slider_saturation.value()
        rcfg.color_temperature = self.slider_color_temp.value()
        rcfg.cas_sharpness = self.slider_cas.value()
        rcfg.grain_intensity = self.slider_reshade_grain.value()
        rcfg.bloom_intensity = self.slider_bloom.value()

        self.parametersChanged.emit(self._current_clip)

    def _preset_reshade_default(self) -> None:
        self._apply_reshade_preset("Cinematic Teal & Orange", 0.85, 0.0, 1.05, 1.10, 0.0)

    def _apply_reshade_preset(
        self,
        lut_name: str,
        lut_strength: float,
        exposure: float,
        contrast: float,
        saturation: float,
        color_temp: float,
    ) -> None:
        self._is_updating_ui = True
        idx = self.combo_lut.findText(lut_name)
        if idx >= 0:
            self.combo_lut.setCurrentIndex(idx)
        self.slider_lut_strength.setValue(lut_strength)
        self.slider_exposure.setValue(exposure)
        self.slider_contrast.setValue(contrast)
        self.slider_saturation.setValue(saturation)
        self.slider_color_temp.setValue(color_temp)
        self._is_updating_ui = False
        self._on_reshade_param_changed()
