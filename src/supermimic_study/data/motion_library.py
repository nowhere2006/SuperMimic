"""Validated, concatenated access to a directory of motion clips."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


MOTION_FIELDS = {
    "joint_pos": (29,),
    "joint_vel": (29,),
    "body_pos_w": (30, 3),
    "body_quat_w": (30, 4),
    "body_lin_vel_w": (30, 3),
    "body_ang_vel_w": (30, 3),
}
_SHA256_LENGTH = 64


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH:
        raise ValueError(f"{label} must be a 64-character SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be a 64-character SHA-256 hex digest") from exc
    return value.lower()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _string_names(value: Any, expected_count: int, label: str) -> tuple[str, ...]:
    if (not isinstance(value, list) or len(value) != expected_count
            or any(not isinstance(name, str) or not name for name in value)):
        raise ValueError(f"{label} must contain {expected_count} non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} must not contain duplicates")
    return tuple(value)


class MotionLibrary:
    """Load clips from a manifest and expose validated per-clip frame indexing.

    All six fields are concatenated along time. ``frame_ids`` passed to
    :meth:`get_frames` are local to the corresponding ``clip_ids`` entry;
    crossing a clip boundary is rejected rather than silently clamped.
    """

    def __init__(
        self,
        *,
        tensors: dict[str, torch.Tensor],
        clip_names: tuple[str, ...],
        joint_names: tuple[str, ...],
        body_names: tuple[str, ...],
        fps: int,
        frame_counts: tuple[int, ...],
        offsets: tuple[int, ...],
        device: torch.device,
    ) -> None:
        self._tensors = tensors
        self.clip_names = clip_names
        self.joint_names = joint_names
        self.body_names = body_names
        self.fps = fps
        self.frame_counts = frame_counts
        self.offsets = offsets
        self.total_frames = sum(frame_counts)
        self.num_clips = len(clip_names)
        self.device = device
        self.storage_bytes = sum(t.numel() * t.element_size() for t in tensors.values())

    @classmethod
    def from_directory(
        cls, directory: str | Path, device: str | torch.device = "cpu"
    ) -> "MotionLibrary":
        """Validate ``manifest.json`` and load its clips onto ``device``."""
        root = Path(directory).expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"motion manifest not found: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read motion manifest: {manifest_path}") from exc
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be a JSON object")
        if manifest.get("schema_version") != 1:
            raise ValueError("manifest schema_version must be 1")
        if manifest.get("fps") != 50:
            raise ValueError("manifest fps must be 50")
        joint_names = _string_names(manifest.get("joint_names"), 29, "joint_names")
        body_names = _string_names(manifest.get("body_names"), 30, "body_names")
        clips = manifest.get("clips")
        if not isinstance(clips, list) or not clips:
            raise ValueError("manifest clips must be a non-empty list")

        target_device = torch.device(device)
        names: list[str] = []
        counts: list[int] = []
        loaded: dict[str, list[torch.Tensor]] = {key: [] for key in MOTION_FIELDS}
        for index, clip in enumerate(clips):
            label = f"clips[{index}]"
            if not isinstance(clip, dict):
                raise ValueError(f"{label} must be an object")
            name = clip.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError(f"{label}.name must be a non-empty string")
            if name in names:
                raise ValueError(f"duplicate clip name: {name}")
            names.append(name)
            path_value = clip.get("path")
            if not isinstance(path_value, str) or not path_value:
                raise ValueError(f"{label}.path must be a non-empty relative path")
            relative = Path(path_value)
            if relative.is_absolute():
                raise ValueError(f"{label}.path must be relative to the motion directory")
            path = (root / relative).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"{label}.path escapes the motion directory")
            if not path.is_file():
                raise ValueError(f"motion file not found for {name}: {path}")
            _require_sha256(clip.get("source_sha256"), f"{label}.source_sha256")
            expected_npz_hash = clip.get("npz_sha256")
            if expected_npz_hash is not None:
                expected_npz_hash = _require_sha256(expected_npz_hash, f"{label}.npz_sha256")
                if _sha256(path) != expected_npz_hash:
                    raise ValueError(f"NPZ SHA-256 mismatch for clip {name}")
            frames = clip.get("frames")
            if isinstance(frames, bool) or not isinstance(frames, int) or frames <= 0:
                raise ValueError(f"{label}.frames must be a positive integer")
            duration = clip.get("duration_s")
            if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
                raise ValueError(f"{label}.duration_s must be a finite non-negative number")
            if not math.isclose(duration, (frames - 1) / 50, rel_tol=0, abs_tol=1e-6):
                raise ValueError(f"{label}.duration_s does not match frame count and fps")

            try:
                with np.load(path, allow_pickle=False) as archive:
                    if "fps" in archive and (archive["fps"].shape != (1,) or archive["fps"][0] != 50):
                        raise ValueError(f"NPZ fps must be [50] for clip {name}")
                    for field, metadata, expected in (
                        ("joint_names", "joint_names", joint_names),
                        ("body_names", "body_names", body_names),
                    ):
                        if metadata in archive:
                            actual = tuple(str(v) for v in archive[metadata].tolist())
                            if actual != expected:
                                raise ValueError(f"NPZ {metadata} mismatch for clip {name}")
                    for field, trailing_shape in MOTION_FIELDS.items():
                        if field not in archive:
                            raise ValueError(f"NPZ missing {field} for clip {name}")
                        array = archive[field]
                        if array.dtype != np.dtype("float32"):
                            raise ValueError(f"{field} must be float32 for clip {name}")
                        if array.shape != (frames, *trailing_shape):
                            raise ValueError(
                                f"{field} shape for clip {name} must be {(frames, *trailing_shape)}, got {array.shape}"
                            )
                        if not np.isfinite(array).all():
                            raise ValueError(f"{field} contains non-finite values for clip {name}")
                        loaded[field].append(torch.from_numpy(array.copy()).to(target_device))
            except (OSError, EOFError, ValueError) as exc:
                if isinstance(exc, ValueError) and str(exc).startswith(("NPZ ", "joint_names ", "body_names ", "joint_", "body_")):
                    raise
                raise ValueError(f"cannot load NPZ for clip {name}: {exc}") from exc
            counts.append(frames)

        offsets: list[int] = []
        cursor = 0
        for count in counts:
            offsets.append(cursor)
            cursor += count
        tensors = {field: torch.cat(parts, dim=0) for field, parts in loaded.items()}
        return cls(
            tensors=tensors,
            clip_names=tuple(names),
            joint_names=joint_names,
            body_names=body_names,
            fps=50,
            frame_counts=tuple(counts),
            offsets=tuple(offsets),
            device=target_device,
        )

    def get_frames(self, clip_ids: Any, frame_ids: Any) -> dict[str, torch.Tensor]:
        """Return copied frame batches; ``frame_ids`` are clip-local indices."""
        clips = self._indices(clip_ids, "clip_ids")
        frames = self._indices(frame_ids, "frame_ids")
        if len(clips) != len(frames):
            raise ValueError("clip_ids and frame_ids must have the same length")
        if clips.numel() and ((clips < 0).any() or (clips >= self.num_clips).any()):
            raise IndexError("clip_ids contains an out-of-range clip index")
        if frames.numel():
            limits = torch.tensor(self.frame_counts, dtype=torch.int64)[clips]
            if (frames < 0).any() or (frames >= limits).any():
                raise IndexError("frame_ids contains an out-of-range local frame index")
            offsets = torch.tensor(self.offsets, dtype=torch.int64)[clips]
            absolute = offsets + frames
            indices = absolute.to(self.device)
        else:
            indices = torch.empty((0,), dtype=torch.int64, device=self.device)
        return {field: tensor.index_select(0, indices).clone() for field, tensor in self._tensors.items()}

    @staticmethod
    def _indices(value: Any, label: str) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().to(device="cpu")
        else:
            try:
                tensor = torch.as_tensor(value, device="cpu")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{label} must be a one-dimensional integer sequence") from exc
        if tensor.ndim != 1:
            raise ValueError(f"{label} must be one-dimensional")
        if tensor.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError(f"{label} must contain integer indices")
        return tensor.to(torch.int64)
