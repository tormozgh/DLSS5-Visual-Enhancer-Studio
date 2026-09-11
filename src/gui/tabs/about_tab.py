"""About Tab: Project credits, authorship, hardware status, licenses, and architectural information in a 2-column layout."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class AboutTab(QWidget):
    """Information about DLSS 5 Visual Enhancer, authorship credits, detected hardware, and legal licenses."""

    def __init__(self, gpu_name: str, driver: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._gpu_name = gpu_name
        self._driver = driver
        self._init_ui()

    def _init_ui(self) -> None:
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        main_vbox = QVBoxLayout(container)
        main_vbox.setContentsMargins(24, 24, 24, 24)
        main_vbox.setSpacing(16)

        # Header Title
        title = QLabel("DLSS 5 Visual Enhancer Studio")
        title.setStyleSheet("font-size: 22px; font-weight: 800; color: #f0f1f4;")
        main_vbox.addWidget(title)

        subtitle = QLabel("Next-Generation Neural Rendering & Real-Time AI Upscaling Suite")
        subtitle.setStyleSheet("font-size: 13px; color: #8e94a0; font-weight: 500; margin-bottom: 8px;")
        main_vbox.addWidget(subtitle)

        # 2-Column Main Layout
        columns_layout = QHBoxLayout()
        columns_layout.setSpacing(16)

        # ==========================================
        # LEFT COLUMN: Credits & Licenses
        # ==========================================
        left_col = QVBoxLayout()
        left_col.setSpacing(16)

        # 1. Credits Card
        credits_card = QFrame()
        credits_card.setProperty("class", "studio-card")
        c_layout = QVBoxLayout(credits_card)
        c_layout.setContentsMargins(14, 14, 14, 14)
        c_layout.setSpacing(10)

        credits_title = QLabel("Credits")
        credits_title.setStyleSheet("color: #d0d4dc; font-weight: 700; font-size: 13px;")
        c_layout.addWidget(credits_title)

        core_credit = QLabel(
            "<b>Original Project:</b> Developed by <b>Merserk</b><br>"
            "GitHub Repository: <a href='https://github.com/Merserk/dlss5-visual-enhancer' style='color: #7289da; text-decoration: none;'>https://github.com/Merserk/dlss5-visual-enhancer</a>"
        )
        core_credit.setStyleSheet("color: #e0e3ea; line-height: 1.6; font-size: 12px;")
        core_credit.setOpenExternalLinks(True)
        c_layout.addWidget(core_credit)

        ui_credit = QLabel(
            "<b>Windows Native Studio Desktop UI:</b> Developed by <b>tormozgh</b><br>"
            "GitHub Profile: <a href='https://github.com/tormozgh' style='color: #7289da; text-decoration: none;'>https://github.com/tormozgh</a>"
        )
        ui_credit.setStyleSheet("color: #e0e3ea; line-height: 1.6; font-size: 12px;")
        ui_credit.setOpenExternalLinks(True)
        c_layout.addWidget(ui_credit)

        left_col.addWidget(credits_card)

        # 2. License and Third-Party Notices Card
        lic_card = QFrame()
        lic_card.setProperty("class", "studio-card")
        lic_layout = QVBoxLayout(lic_card)
        lic_layout.setContentsMargins(14, 14, 14, 14)
        lic_layout.setSpacing(10)

        lic_title = QLabel("License and Third-Party Notices")
        lic_title.setStyleSheet("color: #d0d4dc; font-weight: 700; font-size: 13px;")
        lic_layout.addWidget(lic_title)

        lic_main = QLabel(
            "Original application code is licensed under the <b>MIT License</b>, copyright &copy; 2026 Merserk. "
            "That license covers only original project code; it does not relicense or grant rights to any third-party software, "
            "model, binary, trademark, media, or other asset."
        )
        lic_main.setStyleSheet("color: #c0c4ce; line-height: 1.5; font-size: 11px;")
        lic_main.setWordWrap(True)
        lic_layout.addWidget(lic_main)

        lic_details = QLabel(
            "<b>- NVIDIA DLSS/NGX and RTX Video:</b> NVIDIA and its suppliers retain their rights in genuine NVIDIA SDK and runtime "
            "files used for DLSS Neural Rendering, DLSS Frame Generation, and RTX Video features. Use and distribution are governed "
            "by the applicable NVIDIA license terms, including the <a href='https://github.com/NVIDIA/DLSS/blob/main/LICENSE.txt' style='color: #7289da; text-decoration: none;'>NVIDIA RTX SDK License</a>. "
            "Their presence in a portable package does not imply a standalone redistribution right, and this project must not be represented as NVIDIA-sponsored or endorsed.<br><br>"
            "<b>- FFmpeg:</b> FFmpeg and the bundled FFmpeg build retain their own copyright and license terms. Anyone redistributing the included binaries "
            "must preserve the applicable notices and satisfy the license and corresponding-source obligations of that build. See <a href='https://github.com/FFmpeg/FFmpeg/blob/master/LICENSE.md' style='color: #7289da; text-decoration: none;'>FFmpeg licensing</a>.<br><br>"
            "<b>- MPV and yt-dlp:</b> The bundled portable MPV player and yt-dlp resolver retain their own copyright and license terms; preserve the notices shipped with each distribution.<br><br>"
            "<b>- Python, PyQt6 and Packages:</b> Python is provided under the <a href='https://docs.python.org/3.13/license.html' style='color: #7289da; text-decoration: none;'>PSF License</a>. "
            "The portable Python runtime and packages including PyQt6, Pillow, pillow-heif, rawpy, resvg-py, PyAV, OpenCV, NumPy, their transitive dependencies, "
            "and bundled codecs retain their own copyright and license terms.<br><br>"
            "<b>Trademarks:</b> NVIDIA, GeForce RTX, NGX, DLSS, and RTX Video are trademarks and/or registered trademarks of NVIDIA Corporation. "
            "FFmpeg, MPV, yt-dlp, Python, PyQt6, and other names belong to their respective owners."
        )
        lic_details.setStyleSheet("color: #9aa0ac; line-height: 1.5; font-size: 11px;")
        lic_details.setWordWrap(True)
        lic_details.setOpenExternalLinks(True)
        lic_layout.addWidget(lic_details)

        left_col.addWidget(lic_card)
        left_col.addStretch()

        columns_layout.addLayout(left_col, 1)

        # ==========================================
        # RIGHT COLUMN: Hardware & Features
        # ==========================================
        right_col = QVBoxLayout()
        right_col.setSpacing(16)

        # 3. Hardware Accelerator Card
        hw_card = QFrame()
        hw_card.setProperty("class", "studio-card")
        hw_layout = QVBoxLayout(hw_card)
        hw_layout.setContentsMargins(14, 14, 14, 14)
        hw_layout.setSpacing(8)

        hw_title = QLabel("Hardware Accelerator")
        hw_title.setStyleSheet("color: #d0d4dc; font-weight: 700; font-size: 13px;")
        hw_desc = QLabel(
            f"Primary AI GPU: {self._gpu_name}\n"
            f"NVIDIA Driver: {self._driver}\n"
            "Direct3D 12 Feature 18 Neural Pipeline with RTX Tensor Cores"
        )
        hw_desc.setStyleSheet("color: #e0e3ea; line-height: 1.5; font-size: 12px;")
        hw_layout.addWidget(hw_title)
        hw_layout.addWidget(hw_desc)
        right_col.addWidget(hw_card)

        # 4. Features Summary Card
        feat_card = QFrame()
        feat_card.setProperty("class", "studio-card")
        feat_layout = QVBoxLayout(feat_card)
        feat_layout.setContentsMargins(14, 14, 14, 14)
        feat_layout.setSpacing(8)

        feat_title = QLabel("Core Engine Features")
        feat_title.setStyleSheet("color: #d0d4dc; font-weight: 700; font-size: 13px;")
        feat_text = QLabel(
            "- Neuroframe Engine: In-process Direct3D 12 / NGX neural rendering bridge.\n"
            "- Professional Timeline Scrubber: Real-time frame seek, synchronized playback, and SMPTE timecode.\n"
            "- Dual Canvas Comparison: Draggable Split-Slider and Side-by-Side synchronized viewports.\n"
            "- RTX Video Super Resolution (VSR) & RTX Video HDR hardware processing.\n"
            "- DLSS Frame Generation (DLSSG) multi-frame video interpolation."
        )
        feat_text.setStyleSheet("color: #e0e3ea; line-height: 1.6; font-size: 12px;")
        feat_layout.addWidget(feat_title)
        feat_layout.addWidget(feat_text)
        right_col.addWidget(feat_card)

        # 5. Runtime Components Note Card
        notice_card = QFrame()
        notice_card.setProperty("class", "studio-card")
        notice_layout = QVBoxLayout(notice_card)
        notice_layout.setContentsMargins(14, 14, 14, 14)
        notice_layout.setSpacing(8)

        notice_title = QLabel("Runtime Components Note")
        notice_title.setStyleSheet("color: #b0b5c0; font-weight: 700; font-size: 12px;")
        notice_desc = QLabel(
            "Official portable release packages bundle 'neuroframe_engine.dll', 'nvngx_dlssnr.dll', and 'ffmpeg.exe' in bin/.\n"
            "If using a source Git clone, copy the runtime bin/ directory from official release archives to enable native GPU execution."
        )
        notice_desc.setStyleSheet("color: #9ca0ab; font-size: 11px; line-height: 1.5;")
        notice_desc.setWordWrap(True)
        notice_layout.addWidget(notice_title)
        notice_layout.addWidget(notice_desc)
        right_col.addWidget(notice_card)
        right_col.addStretch()

        columns_layout.addLayout(right_col, 1)

        main_vbox.addLayout(columns_layout)

        # Footer
        footer = QLabel("Independent community project. MIT License.")
        footer.setStyleSheet("color: #606470; font-size: 11px; margin-top: 8px;")
        main_vbox.addWidget(footer)

        scroll_area.setWidget(container)
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(scroll_area)
