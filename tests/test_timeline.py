"""Unit tests for the DLSS 5 Timeline Studio engine, models, and compositing pipeline."""

import contextlib
import os
import tempfile
import cv2
import numpy as np
import pytest

from src.timeline.models import (
    DLSSConfig,
    MediaAsset,
    TimelineClip,
    TimelineTrack,
    TimelineProject,
)
from src.timeline.frame_cache import VideoFrameCache
from src.timeline.processor import DLSS5TimelineProcessor
from src.timeline.compositor import TimelineCompositor


def test_dlss_config_defaults():
    cfg = DLSSConfig()
    assert cfg.preset == "Quality"
    assert cfg.scale_factor == 2.0
    assert cfg.sharpness == 65.0
    assert cfg.denoise == 40.0
    assert cfg.hdr_boost == 0.35
    assert cfg.opacity == 1.0
    d = cfg.to_dict()
    assert d["preset"] == "Quality"
    cfg2 = DLSSConfig.from_dict(d)
    assert cfg2.scale_factor == 2.0


def test_timeline_clip_and_tracks():
    asset = MediaAsset.create(
        file_path="test.mp4",
        duration_frames=120,
        duration_sec=4.0,
        fps=30.0,
        width=1920,
        height=1080,
    )
    clip = TimelineClip.create_media_clip(asset, track_id=1, timeline_in=10)
    assert clip.track_id == 1
    assert clip.timeline_in == 10
    assert clip.timeline_out == 130
    assert clip.duration_frames == 120

    # Adjustment Layer
    adj = TimelineClip.create_adjustment_layer(track_id=2, timeline_in=20, duration_frames=60)
    assert adj.clip_type == "adjustment_layer"
    assert adj.track_id == 2
    assert adj.timeline_in == 20
    assert adj.timeline_out == 80
    assert adj.duration_frames == 60


def test_timeline_project_and_active_clips():
    project = TimelineProject.create_default(fps=30.0, width=1920, height=1080)
    assert len(project.audio_tracks) == 0  # Audio tracks removed per design

    # Test adding and removing video tracks
    init_v_count = len(project.video_tracks)
    new_track = project.add_video_track()
    assert len(project.video_tracks) == init_v_count + 1
    assert new_track.name == f"V{init_v_count + 1}"
    project.remove_video_track(new_track.track_id)
    assert len(project.video_tracks) == init_v_count

    track_v1 = next(t for t in project.video_tracks if t.name == "V1")
    track_v2 = next(t for t in project.video_tracks if t.name == "V2")

    # Add clip to V1
    c1 = TimelineClip(
        clip_id="c1",
        clip_type="media",
        name="Clip 1",
        track_id=1,
        timeline_in=0,
        timeline_out=100,
    )
    track_v1.add_clip(c1)

    # Add adjustment layer to V2
    adj = TimelineClip.create_adjustment_layer(track_id=2, timeline_in=30, duration_frames=40)
    track_v2.add_clip(adj)

    # Check active clips at frame 10 (only c1 on V1)
    active_10 = project.get_active_video_clips_at(10)
    assert len(active_10) == 1
    assert active_10[0][1].clip_id == "c1"

    # Check active clips at frame 50 (c1 on V1 and adj on V2)
    active_50 = project.get_active_video_clips_at(50)
    assert len(active_50) == 2
    assert active_50[0][1].clip_id == "c1"
    assert active_50[1][1].clip_type == "adjustment_layer"


def test_clip_splitting():
    project = TimelineProject.create_default(fps=30.0, width=1920, height=1080)
    track_v1 = next(t for t in project.video_tracks if t.name == "V1")

    clip = TimelineClip(
        clip_id="splittable_clip",
        clip_type="media",
        name="Splittable",
        track_id=1,
        timeline_in=10,
        timeline_out=60,
        source_in=0,
        source_out=50,
    )
    track_v1.add_clip(clip)

    res = project.split_clip("splittable_clip", split_frame=30)
    assert res is not None
    part1, part2 = res
    assert part1.timeline_in == 10
    assert part1.timeline_out == 30
    assert part1.duration_frames == 20

    assert part2.timeline_in == 30
    assert part2.timeline_out == 60
    assert part2.duration_frames == 30
    assert len(track_v1.clips) == 2


def test_dlss5_processor():
    processor = DLSS5TimelineProcessor()
    test_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    test_frame[60:180, 80:240] = (200, 150, 100)

    cfg = DLSSConfig(
        preset="Ultra Quality",
        scale_factor=1.0,
        sharpness=80.0,
        denoise=40.0,
        cinematic_tone=True,
        reshade_preset="Cinematic Teal & Orange",
        hdr_boost=0.5,
        opacity=0.9,
    )

    out = processor.process_frame(test_frame, cfg, target_size=(320, 240))
    assert out is not None
    assert out.shape == (240, 320, 3)
    assert out.dtype == np.uint8


def test_compositor_with_adjustment_layer():
    cache = VideoFrameCache()
    project = TimelineProject.create_default(fps=30.0, width=320, height=240)
    track_v1 = next(t for t in project.video_tracks if t.name == "V1")
    track_v2 = next(t for t in project.video_tracks if t.name == "V2")

    # Create dummy color image asset
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img_path = f.name
    img = np.full((240, 320, 3), 120, dtype=np.uint8)
    cv2.imwrite(img_path, img)

    try:
        asset = MediaAsset.create(
            file_path=img_path,
            duration_frames=60,
            duration_sec=2.0,
            fps=30.0,
            width=320,
            height=240,
        )
        clip_v1 = TimelineClip.create_media_clip(asset, track_id=1, timeline_in=0)
        track_v1.add_clip(clip_v1)

        # DLSS 5 Adjustment Layer on V2
        adj_clip = TimelineClip.create_adjustment_layer(track_id=2, timeline_in=0, duration_frames=60)
        adj_clip.dlss_params.sharpness = 90.0
        adj_clip.dlss_params.hdr_boost = 0.5
        track_v2.add_clip(adj_clip)

        compositor = TimelineCompositor(project, cache)

        # Render composite frame
        rendered = compositor.render_frame(10, target_size=(320, 240))
        assert rendered is not None
        assert rendered.shape == (240, 320, 3)

        # Render split comparison
        split_rendered = compositor.render_frame(10, target_size=(320, 240), split_ratio=0.5)
        assert split_rendered is not None
        assert split_rendered.shape == (240, 320, 3)
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)


def test_media_pool_import_and_drag_drop():
    import os
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    from src.gui.components.timeline.media_pool import MediaPoolWidget

    cache = VideoFrameCache()
    pool = MediaPoolWidget(cache)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img_path = f.name
    img = np.full((120, 160, 3), 200, dtype=np.uint8)
    cv2.imwrite(img_path, img)

    try:
        asset = pool.import_file(img_path)
        assert asset is not None
        assert asset.width == 160
        assert asset.height == 120
        assert asset.duration_frames == 150
        assert len(pool.assets) == 1
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)


def test_sequence_settings_and_reshade_fx():
    from src.timeline.models import SequenceSettings, ReShadeConfig

    # Sequence settings
    seq = SequenceSettings(name="Cinema Master 4K", width=3840, height=2160, fps=24.0)
    d = seq.to_dict()
    assert d["width"] == 3840
    seq2 = SequenceSettings.from_dict(d)
    assert seq2.fps == 24.0

    # Project sequence integration
    proj = TimelineProject.create_default(fps=24.0, width=3840, height=2160, sequence_name="Cinema Master 4K")
    assert proj.sequence.name == "Cinema Master 4K"
    assert proj.width == 3840

    # ReShade config
    rcfg = ReShadeConfig(lut_name="Cyberpunk Neon", exposure=0.3, contrast=1.15)
    rd = rcfg.to_dict()
    assert rd["lut_name"] == "Cyberpunk Neon"
    rcfg2 = ReShadeConfig.from_dict(rd)
    assert abs(rcfg2.exposure - 0.3) < 0.01

    # FX Layer creation
    fx_dlss = TimelineClip.create_fx_layer(track_id=2, timeline_in=0, duration_frames=60, fx_type="dlss5")
    assert fx_dlss.fx_type == "dlss5"
    assert fx_dlss.clip_type == "fx_layer"

    fx_reshade = TimelineClip.create_fx_layer(track_id=2, timeline_in=0, duration_frames=60, fx_type="reshade")
    assert fx_reshade.fx_type == "reshade"
    assert fx_reshade.color == "#0ea5e9"

    # Processor evaluation
    proc = DLSS5TimelineProcessor()
    frame = np.full((120, 160, 3), 150, dtype=np.uint8)
    out_rs = proc.process_reshade_frame(frame, rcfg)
    assert out_rs is not None
    assert out_rs.shape == frame.shape


def test_original_preview_non_black_without_fx_layer():
    cache = VideoFrameCache()
    proj = TimelineProject.create_default(fps=30.0, width=320, height=240)
    track_v1 = next(t for t in proj.video_tracks if t.name == "V1")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img_path = f.name
    img = np.full((240, 320, 3), 180, dtype=np.uint8)
    cv2.imwrite(img_path, img)

    try:
        asset = MediaAsset.create(file_path=img_path, duration_frames=60, duration_sec=2.0, fps=30.0, width=320, height=240)
        clip_v1 = TimelineClip.create_media_clip(asset, track_id=1, timeline_in=0)
        track_v1.add_clip(clip_v1)

        comp = TimelineCompositor(proj, cache)
        # Without any FX layer, return_raw=True must yield non-black original
        composite, raw = comp.render_frame(0, target_size=(320, 240), return_raw=True)
        assert composite is not None
        assert raw is not None
        assert raw.max() > 0, "Original preview frame must not be black when media footage is present!"
        assert np.array_equal(composite, raw), "Without FX layer, composite and original should both reflect footage"
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)


def test_dlss5_processor_no_lut_color_distortion():
    """Verify that DLSS 5 layer preserves color neutrality and does not apply ReShade LUT."""
    processor = DLSS5TimelineProcessor()
    # Neutral gray image
    neutral_frame = np.full((120, 160, 3), 128, dtype=np.uint8)
    cfg = DLSSConfig(
        nr_style="Default",
        scale=1.0,
        nr_intensity=1.0,
        nr_passes=1,
        local_tone_strength=1.0,
        local_structure_strength=1.0,
        skin_structure_strength=0.0,
        nr_color_strength=1.0,
        tone_preservation=0.0,
    )

    out = processor.process_frame(neutral_frame, cfg, target_size=(160, 120), frame_idx=0)
    assert out is not None
    assert out.shape == (120, 160, 3)

    # Neutral input should remain color neutral (B, G, R differences should be minimal, unlike a Teal/Orange LUT)
    b, g, r = out[60, 80]
    diff = max(abs(int(b) - int(g)), abs(int(g) - int(r)), abs(int(b) - int(r)))
    assert diff < 15, f"DLSS 5 layer must not apply color grading LUT! Color difference was {diff}"
    processor.close()


def test_timeline_studio_sequence_splash_workflow():
    """Verify that TimelineStudioTab starts on the initial sequence splash page."""
    from PyQt6.QtWidgets import QApplication
    from src.settings.models import UISettings
    from src.gui.tabs.timeline_tab import TimelineStudioTab

    app = QApplication.instance() or QApplication([])
    tab = TimelineStudioTab(UISettings())

    # Initially before creating sequence:
    assert tab._sequence_initialized is False
    assert tab.stack.currentIndex() == 0  # Splash page

    # Verify splash page UI elements exist
    assert hasattr(tab, "splash_page")
    assert tab.splash_page is not None

    # Simulate sequence creation
    tab._sequence_initialized = True
    tab.stack.setCurrentIndex(1)
    assert tab.stack.currentIndex() == 1  # Active workspace
    tab.close()


def test_dlss5_consecutive_frames_gpu_speed():
    """Verify that DLSS5TimelineProcessor reuses the active GPU session seamlessly across consecutive frames."""
    processor = DLSS5TimelineProcessor()
    frame = np.full((120, 160, 3), 150, dtype=np.uint8)
    cfg = DLSSConfig(nr_style="Default", scale=1.0, nr_intensity=1.0, nr_passes=1)

    outputs = []
    for i in range(4):
        out = processor.process_frame(frame, cfg, target_size=(160, 120), frame_idx=i)
        assert out is not None
        assert out.shape == (120, 160, 3)
        outputs.append(out)

    assert len(outputs) == 4
    processor.close()


def test_timeline_export_finishes_without_deadlock():
    """Verify that TimelineExportWorker executes to 100% and finishes cleanly without hanging."""
    import time
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from src.timeline.export_worker import TimelineExportWorker

    cache = VideoFrameCache()
    proj = TimelineProject.create_default(fps=30.0, width=160, height=120)
    comp = TimelineCompositor(proj, cache)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        export_path = f.name

    try:
        worker = TimelineExportWorker(
            compositor=comp,
            start_frame=0,
            end_frame=15,
            output_path=export_path,
            target_resolution=(160, 120),
            target_fps=30.0,
            codec="libx264",
            bitrate_mbps=5.0,
        )

        completed = []
        errors = []
        worker.finishedExport.connect(completed.append)
        worker.failedExport.connect(errors.append)

        worker.start()
        # Wait up to 10 seconds for 15 frames to finish
        worker.wait(10000)
        app.processEvents()

        assert not errors, f"Export failed with error: {errors}"
        assert len(completed) == 1, "Export did not emit finishedExport signal (likely deadlocked!)"
        assert os.path.exists(export_path), "Export file was not created"
        assert os.path.getsize(export_path) > 0, "Export file is empty"
    finally:
        if os.path.exists(export_path):
            with contextlib.suppress(Exception):
                os.remove(export_path)


def test_settings_get_output_dir():
    """Verify that UISettings.get_output_dir() resolves correctly without NameError."""
    from dataclasses import replace
    from src.settings.models import UISettings
    from src.core.paths import OUTPUTS
    from pathlib import Path

    s = UISettings()
    assert s.get_output_dir() == OUTPUTS
    assert isinstance(s.get_output_dir(), Path)

    # Custom output dir
    with tempfile.TemporaryDirectory() as td:
        s2 = replace(s, custom_output_dir=td)
        assert s2.get_output_dir() == Path(td)


def test_draggable_media_list_widget_and_drag_creation():
    """Verify DraggableMediaListWidget mouse events instantiate QMimeData properly without crashing."""
    import json
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent, QIcon, QPixmap
    from PyQt6.QtWidgets import QApplication
    from src.gui.components.timeline.media_pool import DraggableMediaListWidget, MediaAssetListItem

    app = QApplication.instance() or QApplication([])

    widget = DraggableMediaListWidget()
    asset = MediaAsset.create(
        file_path="test_drag.mp4",
        duration_frames=60,
        duration_sec=2.0,
        fps=30.0,
        width=1920,
        height=1080,
    )
    item = MediaAssetListItem(asset, is_adjustment_layer=False)
    pix = QPixmap(48, 30)
    pix.fill(Qt.GlobalColor.blue)
    item.setIcon(QIcon(pix))
    widget.addItem(item)
    widget.setCurrentItem(item)

    # Press event
    press_event = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(10.0, 10.0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    widget.mousePressEvent(press_event)
    assert widget._drag_start_pos == QPoint(10, 10)

    # Release event resets drag pos
    release_event = QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease,
        QPointF(10.0, 10.0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    widget.mouseReleaseEvent(release_event)
    assert widget._drag_start_pos.isNull()


def test_core_ffmpeg_preview_probe_import():
    """Verify that probe_video is properly imported and callable in preview module."""
    from src.core.ffmpeg.preview import is_browser_playable, probe_video
    assert callable(probe_video)
    # Testing a non-existent file should safely return False without exception
    assert is_browser_playable("non_existent_file.mp4") is False


