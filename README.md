# DLSS 5 Visual Enhancer Studio
<img width="1916" height="1097" alt="dlss5ves" src="https://github.com/user-attachments/assets/01cb99b2-67a3-4e81-ab53-805d82f9fe4b" />

[![Platform](https://img.shields.io/badge/Platform-Windows%2011-0078D4?style=flat-square&logo=windows11&logoColor=white)](https://github.com/tormozgh/dlss5-visual-enhancer/releases)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-RTX-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://github.com/tormozgh/dlss5-visual-enhancer)
[![DLSS](https://img.shields.io/badge/DLSS-5%20Neural-76B900?style=flat-square)](https://github.com/tormozgh/dlss5-visual-enhancer)
[![UI](https://img.shields.io/badge/UI-PyQt6%20Native%20Studio-5c6ac4?style=flat-square)](https://github.com/tormozgh/dlss5-visual-enhancer)
[![License](https://img.shields.io/badge/License-MIT-gray?style=flat-square)](LICENSE)

A standalone, high-performance Windows Desktop application (built with PyQt6 and Direct3D 12) for real-time **NVIDIA DLSS 5 Neural Rendering**, **RTX Video Super Resolution (VSR)**, **RTX Video HDR**, and **DLSS Frame Generation (DLSSG)**.

This native studio suite replaces browser-based interfaces with an ultra-responsive, zero-latency desktop workflow engineered for creative artists, video editors, and AI visual enthusiasts.


## Why the Native Studio Edition?

The original web UI relied on local browser rendering and round-trip server requests, causing delays when tweaking neural parameters or comparing frames. This Windows Native Studio edition runs directly on the desktop hardware:

- **Real-Time Interactive Preview:** Instant slider adjustments with thread-safe queued GPU preview rendering (~18–25 FPS live preview on NVIDIA RTX GPUs).
- **Professional Video Scrubber Timeline:** Frame-accurate scrubbing, play/pause with temporal DLSS continuity, frame-stepping (`|<`, `<`, `>`, `>|`), looping, and SMPTE-standard timecode display (`HH:MM:SS:FF`).
- **Interactive Dual Canvas Comparison:** Interactive draggable Split-View divider slider and synchronized Side-by-Side viewports with zoom, pan, 1:1 pixel view, and fit-to-canvas modes.
- **Production Export Pipelines:** Full background video and image rendering for Neural Rendering, Upscale, and Frame Interpolation with live progress bars, cancellation support, and output file management.
- **Customizable Output Directory:** Select custom storage paths for rendered media directly in Settings.


---

## Key Features

### 1. Neural Rendering (DLSS 5)
- Powered by the self-contained **Neuroframe Engine** (in-process Direct3D 12 Feature 18 & NVIDIA NGX runtime).
- Real-time adjustment of **NR Style** (Default, Natural, Cinematic), **NR Intensity**, **NR Passes** (1 to 4 multi-pass accumulation), **Local Tone**, **Local Structure**, **Skin Structure**, and **Color Strength**.
- Advanced composition controls: **Tone Preservation**, **Face/Skin Protection**, **Grain Preservation**, and **Shimmer Suppression**.
- Temporal stabilization during video playback to ensure flicker-free, continuous neural enhancement.
- Instant single-image export (`PNG`, `JPEG`, `WebP`, `TIFF`) and multi-pass video conversion (`MP4`, `MKV`, `MOV` with NVENC hardware acceleration).

### 2. Upscale (RTX Video Super Resolution & HDR)
- Dedicated workflow for hardware-accelerated upscaling via NVIDIA RTX Tensor Cores.
- **VSR Quality Levels:** Level 1 (Fast) through Level 4 (Ultra Quality).
- **Scale Factors:** 1.5×, 2.0× (Standard 4K/1440p Target), 3.0×, and 4.0× scaling with live target resolution calculation.
- **RTX Video HDR:** Hardware SDR-to-10-bit HDR conversion with adjustable contrast, saturation, and peak luminance up to 2000 nits.
- Direct image and full-length video upscale export.

### 3. Frame Interpolation (DLSS Frame Generation)
- Neural multi-frame interpolation powered by NVIDIA Optical Flow Accelerator (OFA).
- Boosts video framerates smoothly up to 240+ FPS (60, 120, 144, 240 FPS targets).
- Engines: **Auto**, **Native DLSSG**, and **Cascade Multi-Pass**.
- Preflight analysis displaying native temporal grids, cascade stages, and estimated output frame counts.
- 3-second quick clip preview and full interpolated video export.

### 4. Settings & Storage
- AI Processing GPU and Video Processing GPU device selection.
- GPU VRAM Direct (CUDA/D3D12 Shared) vs. System RAM Staging execution path.
- Customizable Output Directory selection with persistent `config.ini` storage.

---

## Installation & Running

### Option 1: Standalone Portable Package (Recommended)

1. Go to the [Releases](https://github.com/tormozgh/dlss5-visual-enhancer/releases) page.
2. Download the latest `DLSS5-Visual-Enhancer-Studio-v1.0.0.zip` release archive.
3. Extract the ZIP to your desired location (e.g. `D:\DLSS5-Studio`).
4. Double-click **`start.bat`**. The native Windows Studio UI will launch immediately.

### Option 2: Running from Source

1. Clone this repository:
   ```bash
   git clone https://github.com/tormozgh/dlss5-visual-enhancer.git
   cd dlss5-visual-enhancer
   ```
2. Download the bundled runtime `bin/` directory from the official release package and place it in the project root.
3. Run `start.bat` or launch with Python:
   ```bash
   bin\python-3.13.15-embed-amd64\python.exe main_gui.py
   ```

---

## Hardware Requirements

- **Operating System:** 64-bit Windows 11 with Direct3D 12 support.
- **GPU:** NVIDIA GeForce RTX Series GPU with a modern NVIDIA Game Ready or Studio Driver.
  - **DLSS 5 Neural Rendering:** GeForce RTX 20, 30, 40, and 50 Series (Turing, Ampere, Ada Lovelace, Blackwell).
  - **Frame Generation (DLSSG):** GeForce RTX 40 and 50 Series GPUs (Hardware-accelerated GPU scheduling / HAGS recommended).
  - **RTX Video Super Resolution & HDR:** RTX 30, 40, and 50 Series GPUs.

---

## Authorship & Credits

- **Original Core Engine:** Developed by [Merserk](https://github.com/Merserk/dlss5-visual-enhancer)
- **Desktop Studio:** Developed by [tormozgh](https://github.com/tormozgh)

---

## License & Third-Party Notices

Original application code is licensed under the **MIT License**, copyright &copy; 2026 Merserk.

- **NVIDIA DLSS/NGX and RTX Video:** NVIDIA and its suppliers retain their rights in genuine NVIDIA SDK and runtime files used for DLSS Neural Rendering, DLSS Frame Generation, and RTX Video features. Use and distribution are governed by the applicable NVIDIA license terms, including the [NVIDIA RTX SDK License](https://github.com/NVIDIA/DLSS/blob/main/LICENSE.txt).
- **FFmpeg:** Retains its own copyright and license terms. See [FFmpeg licensing](https://github.com/FFmpeg/FFmpeg/blob/master/LICENSE.md).
- **MPV and yt-dlp:** Retain their own copyright and license terms.
- **Python, PyQt6 and Packages:** Python is provided under the [PSF License](https://docs.python.org/3.13/license.html). PyQt6, Pillow, OpenCV, NumPy, and their dependencies retain their own respective licenses.
- **Trademarks:** NVIDIA, GeForce RTX, NGX, DLSS, and RTX Video are trademarks and/or registered trademarks of NVIDIA Corporation.
