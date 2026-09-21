"""Unit tests for the DLSS 5 Timeline Studio engine, models, and compositing pipeline."""

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
    assert len(project.video_tracks) == 3
    assert len(project.audio_tracks) == 2

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
