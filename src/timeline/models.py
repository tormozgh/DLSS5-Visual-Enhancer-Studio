"""Timeline Data Models for NLE Video Editing and DLSS 5 / ReShade FX Adjustment Layers."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DLSSConfig:
    """Configuration parameters for DLSS 5 Neural Enhancement per clip or adjustment layer."""

    # Neural Rendering Core (matching Image 1)
    nr_style: str = "Default"  # "Default", "Natural", "Cinematic"
    scale: float = 1.0  # 1.0, 0.75, 0.50, 0.25
    nr_intensity: float = 1.0  # 0.00 to 2.00
    nr_passes: int = 1  # 1 to 4

    # Tone & Structure
    local_tone_strength: float = 1.0  # 0.00 to 2.00
    local_structure_strength: float = 1.0  # 0.00 to 2.00
    skin_structure_strength: float = -1.0  # -1.00 to 1.00

    # Neural Composition
    nr_color_strength: float = 1.0  # 0.00 to 1.00
    tone_preservation: float = 0.0  # 0.00 to 1.00
    face_skin_protection: float = 0.0  # 0.00 to 1.00
    grain_preservation: float = 0.0  # 0.00 to 1.00
    mask_feather: int = 0  # 0 to 128 px
    automatic_mask: bool = False

    # Backwards compatibility & Post-processing
    preset: str = "Quality"
    scale_factor: float = 2.0
    sharpness: float = 65.0
    denoise: float = 40.0
    cinematic_tone: bool = True
    hdr_boost: float = 0.35
    model_name: str = "DLSS 5 Neural Reconstruction"
    reshade_preset: str = "Cinematic Realism"
    opacity: float = 1.0
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "nr_style": self.nr_style,
            "scale": self.scale,
            "nr_intensity": self.nr_intensity,
            "nr_passes": self.nr_passes,
            "local_tone_strength": self.local_tone_strength,
            "local_structure_strength": self.local_structure_strength,
            "skin_structure_strength": self.skin_structure_strength,
            "nr_color_strength": self.nr_color_strength,
            "tone_preservation": self.tone_preservation,
            "face_skin_protection": self.face_skin_protection,
            "grain_preservation": self.grain_preservation,
            "mask_feather": self.mask_feather,
            "automatic_mask": self.automatic_mask,
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
            nr_style=data.get("nr_style", "Default"),
            scale=float(data.get("scale", 1.0)),
            nr_intensity=float(data.get("nr_intensity", 1.0)),
            nr_passes=int(data.get("nr_passes", 1)),
            local_tone_strength=float(data.get("local_tone_strength", 1.0)),
            local_structure_strength=float(data.get("local_structure_strength", 1.0)),
            skin_structure_strength=float(data.get("skin_structure_strength", -1.0)),
            nr_color_strength=float(data.get("nr_color_strength", 1.0)),
            tone_preservation=float(data.get("tone_preservation", 0.0)),
            face_skin_protection=float(data.get("face_skin_protection", 0.0)),
            grain_preservation=float(data.get("grain_preservation", 0.0)),
            mask_feather=int(data.get("mask_feather", 0)),
            automatic_mask=bool(data.get("automatic_mask", False)),
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
class ReShadeConfig:
    """Configuration for ReShade post-processing FX on timeline clips/layers."""

    enabled: bool = True
    lut_enabled: bool = True
    lut_name: str = "Cinematic Teal & Orange"
    lut_strength: float = 0.85
    tonemap_enabled: bool = True
    exposure: float = 0.0  # -2.0 to +2.0 EV
    contrast: float = 1.05  # 0.5 to 2.0
    saturation: float = 1.10  # 0.0 to 2.0
    color_temperature: float = 0.0  # -1.0 to +1.0
    cas_enabled: bool = True
    cas_sharpness: float = 0.40  # 0.0 to 1.0
    grain_enabled: bool = True
    grain_intensity: float = 0.18  # 0.0 to 1.0
    grain_size: float = 1.5  # 1.0 to 3.0
    grain_colored: bool = False
    hdr_boost: float = 0.35  # 0.0 to 1.0
    bloom_intensity: float = 0.0  # 0.0 to 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "lut_enabled": self.lut_enabled,
            "lut_name": self.lut_name,
            "lut_strength": self.lut_strength,
            "tonemap_enabled": self.tonemap_enabled,
            "exposure": self.exposure,
            "contrast": self.contrast,
            "saturation": self.saturation,
            "color_temperature": self.color_temperature,
            "cas_enabled": self.cas_enabled,
            "cas_sharpness": self.cas_sharpness,
            "grain_enabled": self.grain_enabled,
            "grain_intensity": self.grain_intensity,
            "grain_size": self.grain_size,
            "grain_colored": self.grain_colored,
            "hdr_boost": self.hdr_boost,
            "bloom_intensity": self.bloom_intensity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReShadeConfig:
        return cls(
            enabled=bool(data.get("enabled", True)),
            lut_enabled=bool(data.get("lut_enabled", True)),
            lut_name=str(data.get("lut_name", "Cinematic Teal & Orange")),
            lut_strength=float(data.get("lut_strength", 0.85)),
            tonemap_enabled=bool(data.get("tonemap_enabled", True)),
            exposure=float(data.get("exposure", 0.0)),
            contrast=float(data.get("contrast", 1.05)),
            saturation=float(data.get("saturation", 1.10)),
            color_temperature=float(data.get("color_temperature", 0.0)),
            cas_enabled=bool(data.get("cas_enabled", True)),
            cas_sharpness=float(data.get("cas_sharpness", 0.40)),
            grain_enabled=bool(data.get("grain_enabled", True)),
            grain_intensity=float(data.get("grain_intensity", 0.18)),
            grain_size=float(data.get("grain_size", 1.5)),
            grain_colored=bool(data.get("grain_colored", False)),
            hdr_boost=float(data.get("hdr_boost", 0.35)),
            bloom_intensity=float(data.get("bloom_intensity", 0.0)),
        )


@dataclass
class SequenceSettings:
    """Settings defining a video timeline sequence (timebase, frame size, aspect)."""

    name: str = "Sequence 01"
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    preset_name: str = "1080p Full HD (1920x1080 30fps)"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "preset_name": self.preset_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SequenceSettings:
        return cls(
            name=str(data.get("name", "Sequence 01")),
            width=int(data.get("width", 1920)),
            height=int(data.get("height", 1080)),
            fps=float(data.get("fps", 30.0)),
            preset_name=str(data.get("preset_name", "1080p Full HD (1920x1080 30fps)")),
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
    clip_type: str  # "media" | "fx_layer" | "adjustment_layer"
    name: str
    track_id: int
    timeline_in: int  # Starting frame on timeline (inclusive)
    timeline_out: int  # Ending frame on timeline (exclusive)
    source_in: int = 0  # Starting frame in source file
    source_out: int = 0  # Ending frame in source file
    asset_id: str | None = None
    media_path: str | None = None
    fx_type: str = "dlss5"  # "dlss5" | "reshade"
    dlss_params: DLSSConfig = field(default_factory=DLSSConfig)
    reshade_params: ReShadeConfig = field(default_factory=ReShadeConfig)
    opacity: float = 1.0
    scale_mode: str = "fit"  # "fit" | "fill" | "stretch"
    asset_fps: float = 30.0
    asset_width: int = 1920
    asset_height: int = 1080
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
        timeline_fps: float = 30.0,
    ) -> TimelineClip:
        # Calculate duration in timeline frame units conforming to timeline sequence fps
        timeline_duration = source_duration if source_duration is not None else int(round(asset.duration_sec * timeline_fps))
        timeline_out = timeline_in + max(1, timeline_duration)
        source_out = source_in + asset.duration_frames
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
            fx_type="none",
            dlss_params=DLSSConfig(enabled=False),
            reshade_params=ReShadeConfig(enabled=False),
            opacity=1.0,
            scale_mode="fit",
            asset_fps=asset.fps,
            asset_width=asset.width,
            asset_height=asset.height,
            color="#2563eb",
        )

    @classmethod
    def create_fx_layer(
        cls,
        track_id: int,
        timeline_in: int,
        duration_frames: int = 150,  # 5 seconds at 30 fps
        fx_type: str = "dlss5",
        name: str | None = None,
    ) -> TimelineClip:
        timeline_out = timeline_in + max(1, duration_frames)
        if fx_type == "reshade":
            layer_name = name or "ReShade FX Layer"
            color = "#0ea5e9"  # Cyan for ReShade
        else:
            layer_name = name or "DLSS 5 Neural Layer"
            color = "#9333ea"  # Purple for DLSS 5

        return cls(
            clip_id=str(uuid.uuid4()),
            clip_type="fx_layer",
            name=layer_name,
            track_id=track_id,
            timeline_in=timeline_in,
            timeline_out=timeline_out,
            source_in=0,
            source_out=max(1, duration_frames),
            asset_id=None,
            media_path=None,
            fx_type=fx_type,
            dlss_params=DLSSConfig(enabled=True),
            reshade_params=ReShadeConfig(enabled=True),
            opacity=1.0,
            color=color,
        )

    @classmethod
    def create_adjustment_layer(
        cls,
        track_id: int,
        timeline_in: int,
        duration_frames: int = 150,
        name: str = "DLSS 5 Adjustment Layer",
    ) -> TimelineClip:
        """Backwards compatibility alias for DLSS 5 FX layer."""
        clip = cls.create_fx_layer(
            track_id=track_id,
            timeline_in=timeline_in,
            duration_frames=duration_frames,
            fx_type="dlss5",
            name=name,
        )
        clip.clip_type = "adjustment_layer"
        return clip


@dataclass
class TimelineTrack:
    """A single timeline track (V1, V2, etc.) hosting clips."""

    track_id: int
    name: str
    track_type: str = "video"  # "video"
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
    """Top-level project representing the complete timeline state and sequence."""

    name: str = "Sequence 01"
    fps: float = 30.0
    width: int = 1920
    height: int = 1080
    sequence: SequenceSettings = field(default_factory=SequenceSettings)
    video_tracks: list[TimelineTrack] = field(default_factory=list)
    audio_tracks: list[TimelineTrack] = field(default_factory=list)
    work_area_in: int = 0
    work_area_out: int = 0  # 0 indicates end of timeline
    playhead_frame: int = 0

    @classmethod
    def create_default(
        cls,
        fps: float = 30.0,
        width: int = 1920,
        height: int = 1080,
        sequence_name: str = "Sequence 01",
    ) -> TimelineProject:
        seq = SequenceSettings(name=sequence_name, width=width, height=height, fps=fps)
        proj = cls(name=sequence_name, fps=fps, width=width, height=height, sequence=seq)
        # Default tracks: V3, V2, V1
        proj.video_tracks = [
            TimelineTrack(track_id=3, name="V3", track_type="video"),
            TimelineTrack(track_id=2, name="V2", track_type="video"),
            TimelineTrack(track_id=1, name="V1", track_type="video"),
        ]
        proj.audio_tracks = []
        return proj

    def update_sequence(self, settings: SequenceSettings) -> None:
        """Update active sequence configuration (resolution, framerate, name)."""
        self.sequence = settings
        self.name = settings.name
        self.fps = settings.fps
        self.width = settings.width
        self.height = settings.height

    def add_video_track(self, name: str | None = None) -> TimelineTrack:
        """Add a new video track on top of the timeline."""
        existing_ids = [t.track_id for t in self.video_tracks]
        next_id = max(existing_ids) + 1 if existing_ids else 1
        track_name = name or f"V{next_id}"
        new_track = TimelineTrack(track_id=next_id, name=track_name, track_type="video")
        self.video_tracks.append(new_track)
        return new_track

    def remove_video_track(self, track_id: int | None = None) -> bool:
        """Remove a video track (highest or specified by track_id). Keeps at least 1 track."""
        if len(self.video_tracks) <= 1:
            return False
        if track_id is None:
            target = max(self.video_tracks, key=lambda t: t.track_id)
        else:
            target = next((t for t in self.video_tracks if t.track_id == track_id), None)
            if not target:
                return False
        self.video_tracks.remove(target)
        return True

    def get_total_frames(self) -> int:
        max_frame = 0
        for track in self.video_tracks:
            for clip in track.clips:
                if clip.timeline_out > max_frame:
                    max_frame = clip.timeline_out
        return max(max_frame, 300)  # At least 10 seconds empty timeline

    def get_total_duration_sec(self) -> float:
        return self.get_total_frames() / max(1.0, self.fps)

    def find_clip_by_id(self, clip_id: str) -> tuple[TimelineTrack, TimelineClip] | None:
        for track in self.video_tracks:
            for clip in track.clips:
                if clip.clip_id == clip_id:
                    return track, clip
        return None

    def get_active_video_clips_at(self, frame: int) -> list[tuple[TimelineTrack, TimelineClip]]:
        """Return all active video clips at the given frame sorted from bottom (V1) to top."""
        active = []
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
            fx_type=clip.fx_type,
            dlss_params=DLSSConfig.from_dict(clip.dlss_params.to_dict()),
            reshade_params=ReShadeConfig.from_dict(clip.reshade_params.to_dict()),
            opacity=clip.opacity,
            scale_mode=clip.scale_mode,
            asset_fps=clip.asset_fps,
            asset_width=clip.asset_width,
            asset_height=clip.asset_height,
            color=clip.color,
        )
        track.add_clip(clip2)
        return clip, clip2
