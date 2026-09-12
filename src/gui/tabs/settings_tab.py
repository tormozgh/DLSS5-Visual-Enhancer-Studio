"""Settings Tab: Application configuration, GPU selection, and path preferences."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ...core.paths import CONFIG_PATH, LOGS, OUTPUTS
from ...settings.models import (
    CODEC_CHOICES,
    CONTAINER_CHOICES,
    IMAGE_FORMAT_CHOICES,
    QUALITY_CHOICES,
    UISettings,
)
from ...settings.storage import load_settings, save_settings


class SettingsTab(QWidget):
    """Application preferences and persistent hardware configuration."""

    settingsSaved = pyqtSignal()

    def __init__(
        self,
        settings: UISettings,
        gpu_choices: list[tuple[str, str]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._gpu_choices = gpu_choices
        self._init_ui()

    def _init_ui(self) -> None:
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # 1. Hardware & GPU
        gpu_box = QGroupBox("Hardware & GPU Configuration")
        gpu_layout = QVBoxLayout(gpu_box)

        # AI GPU
        ai_layout = QHBoxLayout()
        ai_label = QLabel("AI Processing GPU:")
        ai_label.setStyleSheet("font-weight: 600;")
        self.combo_ai_gpu = QComboBox()
        for label, val in self._gpu_choices:
            self.combo_ai_gpu.addItem(label, val)
        ai_layout.addWidget(ai_label)
        ai_layout.addWidget(self.combo_ai_gpu, 1)
        gpu_layout.addLayout(ai_layout)

        # Engine Mode
        engine_layout = QHBoxLayout()
        engine_label = QLabel("Processing Engine Path:")
        engine_label.setStyleSheet("font-weight: 600;")
        self.combo_engine_path = QComboBox()
        self.combo_engine_path.addItem("GPU VRAM Direct (CUDA/D3D12 Shared - Recommended)", True)
        self.combo_engine_path.addItem("System RAM Staging", False)
        engine_layout.addWidget(engine_label)
        engine_layout.addWidget(self.combo_engine_path, 1)
        gpu_layout.addLayout(engine_layout)

        layout.addWidget(gpu_box)

        # 2. Output Formats
        format_box = QGroupBox("Default Output Formats")
        format_layout = QVBoxLayout(format_box)

        # Image Format
        img_fmt_layout = QHBoxLayout()
        img_fmt_label = QLabel("Default Image Format:")
        img_fmt_label.setStyleSheet("font-weight: 600;")
        self.combo_image_format = QComboBox()
        for fmt in IMAGE_FORMAT_CHOICES:
            self.combo_image_format.addItem(fmt)
        self.combo_image_format.setCurrentText(self._settings.image_format)
        img_fmt_layout.addWidget(img_fmt_label)
        img_fmt_layout.addWidget(self.combo_image_format, 1)
        format_layout.addLayout(img_fmt_layout)

        # Video Codec
        codec_layout = QHBoxLayout()
        codec_label = QLabel("Default Video Codec:")
        codec_label.setStyleSheet("font-weight: 600;")
        self.combo_codec = QComboBox()
        for c in CODEC_CHOICES:
            self.combo_codec.addItem(c)
        self.combo_codec.setCurrentText(self._settings.codec)
        codec_layout.addWidget(codec_label)
        codec_layout.addWidget(self.combo_codec, 1)
        format_layout.addLayout(codec_layout)

        layout.addWidget(format_box)

        # 3. Directory Paths
        dir_box = QGroupBox("Directories & Output Storage")
        dir_layout = QVBoxLayout(dir_box)
        dir_layout.setSpacing(10)

        out_row = QHBoxLayout()
        out_label = QLabel("Output Directory:")
        out_label.setStyleSheet("font-weight: 600;")
        out_label.setFixedWidth(130)

        current_out = str(self._settings.get_output_dir())
        self.txt_output_dir = QLineEdit(current_out)
        self.txt_output_dir.setPlaceholderText("Select custom output folder...")

        self.btn_browse_output = QPushButton("Browse...")
        self.btn_browse_output.setProperty("class", "mini-btn")
        self.btn_browse_output.clicked.connect(self._on_browse_output_dir)

        self.btn_reset_output = QPushButton("Default")
        self.btn_reset_output.setProperty("class", "mini-btn")
        self.btn_reset_output.setToolTip("Reset to project default (outputs/)")
        self.btn_reset_output.clicked.connect(self._on_reset_output_dir)

        out_row.addWidget(out_label)
        out_row.addWidget(self.txt_output_dir, 1)
        out_row.addWidget(self.btn_browse_output)
        out_row.addWidget(self.btn_reset_output)
        dir_layout.addLayout(out_row)

        log_lbl = QLabel(f"Logs Directory:  {LOGS}")
        log_lbl.setStyleSheet("color: #8e94a0; font-family: monospace; font-size: 11px;")
        cfg_lbl = QLabel(f"Config File:     {CONFIG_PATH}")
        cfg_lbl.setStyleSheet("color: #8e94a0; font-family: monospace; font-size: 11px;")

        dir_layout.addWidget(log_lbl)
        dir_layout.addWidget(cfg_lbl)
        layout.addWidget(dir_box)

        # Actions Box
        action_box = QGroupBox("Configuration Profile Management")
        action_layout = QVBoxLayout(action_box)
        action_layout.setSpacing(10)

        row_cfg = QHBoxLayout()
        row_cfg.setSpacing(8)

        btn_save = QPushButton("Save Settings to Config")
        btn_save.setProperty("class", "primary")
        btn_save.setToolTip("Persist current configuration to config.ini for startup")
        btn_save.clicked.connect(self._save_settings)
        row_cfg.addWidget(btn_save)

        btn_load = QPushButton("Load Settings from Config")
        btn_load.setToolTip("Reload configuration values from config.ini")
        btn_load.clicked.connect(self._load_settings)
        row_cfg.addWidget(btn_load)

        action_layout.addLayout(row_cfg)

        row_file = QHBoxLayout()
        row_file.setSpacing(8)

        btn_export = QPushButton("Export Profile...")
        btn_export.setProperty("class", "mini-btn")
        btn_export.setToolTip("Export complete settings profile to a JSON file")
        btn_export.clicked.connect(self._export_profile)
        row_file.addWidget(btn_export)

        btn_import = QPushButton("Import Profile...")
        btn_import.setProperty("class", "mini-btn")
        btn_import.setToolTip("Import complete settings profile from a JSON file")
        btn_import.clicked.connect(self._import_profile)
        row_file.addWidget(btn_import)

        action_layout.addLayout(row_file)
        layout.addWidget(action_box)

        layout.addStretch()
        scroll_area.setWidget(container)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll_area)

    def _on_browse_output_dir(self) -> None:
        from PyQt6.QtWidgets import QFileDialog
        from pathlib import Path
        current = self.txt_output_dir.text().strip() or str(OUTPUTS)
        chosen = QFileDialog.getExistingDirectory(self, "Choose Output Directory", current)
        if chosen:
            self.txt_output_dir.setText(str(Path(chosen).resolve()))

    def _on_reset_output_dir(self) -> None:
        self.txt_output_dir.setText(str(OUTPUTS))

    def _apply_settings_to_ui(self, settings: UISettings) -> None:
        for idx in range(self.combo_ai_gpu.count()):
            if str(self.combo_ai_gpu.itemData(idx)) == str(settings.ai_gpu_uuid):
                self.combo_ai_gpu.setCurrentIndex(idx)
                break

        for idx in range(self.combo_engine_path.count()):
            if bool(self.combo_engine_path.itemData(idx)) == bool(settings.nr_gpu_mode):
                self.combo_engine_path.setCurrentIndex(idx)
                break

        self.combo_image_format.setCurrentText(settings.image_format)
        self.combo_codec.setCurrentText(settings.codec)
        self.txt_output_dir.setText(str(settings.get_output_dir()))

    def _save_settings(self) -> None:
        try:
            from dataclasses import replace
            from pathlib import Path

            custom_out = self.txt_output_dir.text().strip()
            if custom_out:
                p = Path(custom_out)
                p.mkdir(parents=True, exist_ok=True)
                if p.resolve() == OUTPUTS.resolve():
                    custom_out = ""

            new_settings = replace(
                self._settings,
                ai_gpu_uuid=str(self.combo_ai_gpu.currentData() or "auto"),
                nr_gpu_mode=bool(self.combo_engine_path.currentData()),
                image_format=self.combo_image_format.currentText(),
                codec=self.combo_codec.currentText(),
                custom_output_dir=custom_out,
            )
            save_settings(CONFIG_PATH, new_settings)
            self._settings = new_settings
            self.settingsSaved.emit()
            QMessageBox.information(self, "Settings Saved", "Configuration saved successfully to config.ini.")
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", f"Failed to save settings: {exc}")

    def _load_settings(self) -> None:
        try:
            loaded = load_settings(CONFIG_PATH)
            self._settings = loaded
            self._apply_settings_to_ui(loaded)
            self.settingsSaved.emit()
            QMessageBox.information(self, "Settings Loaded", "Configuration loaded successfully from config.ini.")
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", f"Failed to load settings: {exc}")

    def _export_profile(self) -> None:
        try:
            import json
            from pathlib import Path
            from PyQt6.QtWidgets import QFileDialog
            from ...settings.presets import preset_document

            presets_dir = Path("presets")
            presets_dir.mkdir(parents=True, exist_ok=True)
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Settings Profile",
                str(presets_dir / "settings_profile.json"),
                "JSON Preset (*.json);;All Files (*.*)",
            )
            if not file_path:
                return

            if not file_path.lower().endswith(".json"):
                file_path += ".json"

            target = Path(file_path)
            doc = preset_document(target.stem, self._settings)
            target.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            QMessageBox.information(self, "Profile Exported", f"Settings profile successfully exported to:\n{target.name}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", f"Failed to export profile: {exc}")

    def _import_profile(self) -> None:
        try:
            from pathlib import Path
            from PyQt6.QtWidgets import QFileDialog
            from ...settings.presets import import_settings_preset

            presets_dir = Path("presets")
            presets_dir.mkdir(parents=True, exist_ok=True)
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Import Settings Profile",
                str(presets_dir),
                "JSON Preset (*.json);;All Files (*.*)",
            )
            if not file_path:
                return

            name, new_settings = import_settings_preset(file_path, self._settings)
            self._settings = new_settings
            self._apply_settings_to_ui(new_settings)
            save_settings(CONFIG_PATH, new_settings)
            self.settingsSaved.emit()
            QMessageBox.information(self, "Profile Imported", f"Settings profile '{name}' imported and applied successfully.")
        except Exception as exc:
            QMessageBox.critical(self, "Import Error", f"Failed to import profile: {exc}")
