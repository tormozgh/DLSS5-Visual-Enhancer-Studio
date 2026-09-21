"""Timeline Data Models for NLE Video Editing and DLSS 5 Adjustment Layers."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DLSSConfig:
    """Configuration parameters for DLSS 5 Neural Enhancement per clip or adjustment layer."""

    preset: str = "Quality"  # "Ultra Quality", "Quality", "Balanced", "Performance", "Ultra Performance"
    scale_factor: float = 2.0  # 1.0x to 4.0x
    sharpness: float = 65.0  # 0.0 to 100.0
    denoise: float = 40.0  # 0.0 to 100.0
    cinematic_tone: bool = True
    hdr_boost: float = 0.35  # 0.0 to 1.0
    model_name: str = "DLSS 5 Neural Reconstruction"
    reshade_preset: str = "Cinematic Realism"
    opacity: float = 1.0  # 0.0 to 1.0
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "scale_factor": self.scale_factor,
            "sharpness": self.sharpness,
            "denoise": self.denoise,
            "cinematic_tone": self.cinematic_tone,
            "hdr_boost": self.hdr_boost,
            "model_name": self.model_name,
            "reshade_preset": self.reshade_preset,
            "opacity": self.opacity,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DLSSConfig:
        return cls(
            preset=data.get("preset", "Quality"),
            scale_factor=float(data.get("scale_factor", 2.0)),
            sharpness=float(data.get("sharpness", 65.0)),
            denoise=float(data.get("denoise", 40.0)),
            cinematic_tone=bool(data.get("cinematic_tone", True)),
            hdr_boost=float(data.get("hdr_boost", 0.35)),
            model_name=data.get("model_name", "DLSS 5 Neural Reconstruction"),
            reshade_preset=data.get("reshade_preset", "Cinematic Realism"),
            opacity=float(data.get("opacity", 1.0)),
            enabled=bool(data.get("enabled", True)),
        )


@dataclass
class MediaAsset:
    """Imported media asset inside the project bin/media pool."""

    asset_id: str
    file_path: str
    name: str
    duration_frames: int
    duration_sec: float
    fps: float
    width: int
    height: int
    has_audio: bool = True
    thumbnail_path: str | None = None

    @classmethod
    def create(
        cls,
        file_path: str,
        duration_frames: int,
        duration_sec: float,
        fps: float,
        width: int,
        height: int,
        has_audio: bool = True,
        thumbnail_path: str | None = None,
    ) -> MediaAsset:
        name = os.path.basename(file_path)
        return cls(
            asset_id=str(uuid.uuid4()),
            file_path=file_path,
            name=name,
            duration_frames=max(1, duration_frames),
            duration_sec=max(0.1, duration_sec),
            fps=max(1.0, fps),
            width=width,
            height=height,
            has_audio=has_audio,
            thumbnail_path=thumbnail_path,
        )


@dataclass
class TimelineClip:
    """A clip residing on a track in the multi-track timeline."""

    clip_id: str
    clip_type: str  # "media" | "adjustment_layer"
    name: str
    track_id: int
    timeline_in: int  # Starting frame on timeline (inclusive)
    timeline_out: int  # Ending frame on timeline (exclusive)
    source_in: int = 0  # Starting frame in source file
    source_out: int = 0  # Ending frame in source file
    asset_id: str | None = None
    media_path: str | None = None
    dlss_params: DLSSConfig = field(default_factory=DLSSConfig)
    opacity: float = 1.0
    color: str = "#2563eb"

    @property
    def duration_frames(self) -> int:
        return max(0, self.timeline_out - self.timeline_in)

    @classmethod
    def create_media_clip(
        cls,
        asset: MediaAsset,
        track_id: int,
        timeline_in: int,
        source_in: int = 0,
        source_duration: int | None = None,
    ) -> TimelineClip:
        duration = source_duration or asset.duration_frames
        timeline_out = timeline_in + duration
        source_out = source_in + duration
        return cls(
            clip_id=str(uuid.uuid4()),
            clip_type="media",
            name=asset.name,
            track_id=track_id,
            timeline_in=timeline_in,
            timeline_out=timeline_out,
            source_in=source_in,
            source_out=source_out,
            asset_id=asset.asset_id,
            media_path=asset.file_path,
            dlss_params=DLSSConfig(enabled=False),
            opacity=1.0,
            color="#2563eb",
        )

    @classmethod
    def create_adjustment_layer(
        cls,
        track_id: int,
        timeline_in: int,
        duration_frames: int = 150,  # 5 seconds at 30 fps
        name: str = "DLSS 5 Adjustment Layer",
    ) -> TimelineClip:
        timeline_out = timeline_in + max(1, duration_frames)
        return cls(
            clip_id=str(uuid.uuid4()),
            clip_type="adjustment_layer",
            name=name,
            track_id=track_id,
            timeline_in=timeline_in,
            timeline_out=timeline_out,
            source_in=0,
            source_out=max(1, duration_frames),
            asset_id=None,
            media_path=None,
            dlss_params=DLSSConfig(enabled=True),
            opacity=1.0,
            color="#9333ea",  # Distinctive purple/amber for adjustment layer
        )


@dataclass
class TimelineTrack:
    """A single timeline track (V1, V2, A1, etc.) hosting clips."""

    track_id: int
    name: str
    track_type: str = "video"  # "video" | "audio"
    is_muted: bool = False
    is_locked: bool = False
    clips: list[TimelineClip] = field(default_factory=list)

    def add_clip(self, clip: TimelineClip) -> None:
        clip.track_id = self.track_id
        self.clips.append(clip)
        self.clips.sort(key=lambda c: c.timeline_in)

    def remove_clip(self, clip_id: str) -> bool:
        initial_len = len(self.clips)
        self.clips = [c for c in self.clips if c.clip_id != clip_id]
        return len(self.clips) < initial_len

    def get_clip_at(self, frame: int) -> TimelineClip | None:
        for clip in self.clips:
            if clip.timeline_in <= frame < clip.timeline_out:
                return clip
        return None


@dataclass
class TimelineProject:
    """Top-level project representing the complete timeline state."""

    name: str = "DLSS 5 Project"
    fps: float = 30.0
    width: int = 1920
    height: int = 1080
    video_tracks: list[TimelineTrack] = field(default_factory=list)
    audio_tracks: list[TimelineTrack] = field(default_factory=list)
    work_area_in: int = 0
    work_area_out: int = 0  # 0 indicates end of timeline
    playhead_frame: int = 0

    @classmethod
    def create_default(cls, fps: float = 30.0, width: int = 1920, height: int = 1080) -> TimelineProject:
        proj = cls(fps=fps, width=width, height=height)
        # Default tracks: V3, V2, V1 and A1, A2
        proj.video_tracks = [
            TimelineTrack(track_id=3, name="V3", track_type="video"),
            TimelineTrack(track_id=2, name="V2", track_type="video"),
            TimelineTrack(track_id=1, name="V1", track_type="video"),
        ]
        proj.audio_tracks = [
            TimelineTrack(track_id=1, name="A1", track_type="audio"),
            TimelineTrack(track_id=2, name="A2", track_type="audio"),
        ]
        return proj

    def get_total_frames(self) -> int:
        max_frame = 0
        for track in self.video_tracks + self.audio_tracks:
            for clip in track.clips:
                if clip.timeline_out > max_frame:
                    max_frame = clip.timeline_out
        return max(max_frame, 300)  # At least 10 seconds empty timeline

    def get_total_duration_sec(self) -> float:
        return self.get_total_frames() / max(1.0, self.fps)

    def find_clip_by_id(self, clip_id: str) -> tuple[TimelineTrack, TimelineClip] | None:
        for track in self.video_tracks + self.audio_tracks:
            for clip in track.clips:
                if clip.clip_id == clip_id:
                    return track, clip
        return None

    def get_active_video_clips_at(self, frame: int) -> list[tuple[TimelineTrack, TimelineClip]]:
        """Return all active video clips at the given frame sorted from bottom (V1) to top."""
        active = []
        # Sort video tracks by track_id ascending (V1 -> V2 -> V3)
        sorted_tracks = sorted(self.video_tracks, key=lambda t: t.track_id)
        for track in sorted_tracks:
            if track.is_muted:
                continue
            clip = track.get_clip_at(frame)
            if clip is not None:
                active.append((track, clip))
        return active

    def split_clip(self, clip_id: str, split_frame: int) -> tuple[TimelineClip, TimelineClip] | None:
        """Split a clip at the given timeline frame into two independent clips."""
        found = self.find_clip_by_id(clip_id)
        if not found:
            return None
        track, clip = found

        if not (clip.timeline_in < split_frame < clip.timeline_out):
            return None

        split_offset = split_frame - clip.timeline_in
        original_out = clip.timeline_out
        original_source_out = clip.source_out

        # Clip 1 takes left half
        clip.timeline_out = split_frame
        clip.source_out = clip.source_in + split_offset

        # Clip 2 takes right half
        clip2 = TimelineClip(
            clip_id=str(uuid.uuid4()),
            clip_type=clip.clip_type,
            name=f"{clip.name} (Split)",
            track_id=clip.track_id,
            timeline_in=split_frame,
            timeline_out=original_out,
            source_in=clip.source_in + split_offset,
            source_out=original_source_out,
            asset_id=clip.asset_id,
            media_path=clip.media_path,
            dlss_params=DLSSConfig.from_dict(clip.dlss_params.to_dict()),
            opacity=clip.opacity,
            color=clip.color,
        )
        track.add_clip(clip2)
        return clip, clip2
