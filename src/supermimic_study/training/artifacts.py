"""Small, failure-safe persistence helpers for training runs.

This module deliberately does not inspect or snapshot the training loop. Callers
pass the transition and probe payloads they want to retain.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any, Mapping
import uuid

import torch


def _cpu_copy(value: Any) -> Any:
    """Recursively detach and copy tensors to CPU without retaining autograd."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    return value


class RunArtifacts:
    """Persist JSONL logs, per-environment transitions, probes and checkpoints."""

    def __init__(self, root, checkpoint_every=10, keep_checkpoints=3,
                 record_transitions=True):
        if checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive")
        if keep_checkpoints < 1:
            raise ValueError("keep_checkpoints must be at least one")
        self.root = Path(root)
        self.checkpoint_every = int(checkpoint_every)
        self.keep_checkpoints = int(keep_checkpoints)
        self.record_transitions = bool(record_transitions)
        self.transitions_dir = self.root / "transitions"
        self.probes_dir = self.root / "probes"
        self.checkpoints_dir = self.root / "checkpoints"
        self.root.mkdir(parents=True, exist_ok=True)
        if self.record_transitions:
            self.transitions_dir.mkdir(parents=True, exist_ok=True)
        self.probes_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        chunks = sorted(self.transitions_dir.glob("chunk-*.pt")) if self.record_transitions else []
        self._chunk_index = max((int(path.stem.split("-")[-1]) for path in chunks), default=0)
        self._pending_transitions: list[dict[str, Any]] = []
        self._vector_steps = 0
        self._pending_vector_steps = 0

    def _append_json(self, filename: str, record: Mapping[str, Any]) -> None:
        # Validate before opening so non-serializable or NaN data cannot append
        # a partial/broken JSONL record.
        line = json.dumps(dict(record), allow_nan=False, ensure_ascii=False, separators=(",", ":"))
        with (self.root / filename).open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()

    def event(self, name, **fields):
        self._append_json("events.jsonl", {"name": str(name), **fields})

    def metrics(self, values: dict):
        if not isinstance(values, dict):
            raise TypeError("metrics expects a dict")
        self._append_json("metrics.jsonl", values)

    def transitions(self, batch: Mapping[str, Any]) -> None:
        """Cache one vector step as independent CPU copies, one row per env."""
        if not self.record_transitions:
            return
        if not isinstance(batch, Mapping) or not batch:
            raise ValueError("transition batch must be a non-empty mapping")
        values = {key: torch.as_tensor(value) for key, value in batch.items()}
        sizes = {int(value.shape[0]) for value in values.values() if value.ndim > 0}
        if any(value.ndim == 0 for value in values.values()) or len(sizes) != 1:
            raise ValueError("every transition field must have the same leading environment dimension")
        num_envs = sizes.pop()
        if num_envs <= 0:
            raise ValueError("transition batch must contain at least one environment")
        copied = {key: _cpu_copy(value) for key, value in values.items()}
        self._pending_transitions.extend(
            {key: value[env].clone() for key, value in copied.items()}
            for env in range(num_envs)
        )
        self._vector_steps += 1
        self._pending_vector_steps += 1
        if self._pending_vector_steps >= 32:
            self.flush_transitions()

    def flush_transitions(self):
        if not self.record_transitions or not self._pending_transitions:
            return None
        next_index = self._chunk_index + 1
        path = self.transitions_dir / f"chunk-{next_index:06d}.pt"
        payload = {"vector_steps": self._pending_vector_steps,
                   "transitions": self._pending_transitions}
        self._atomic_torch_save(payload, path)
        self._chunk_index = next_index
        self._pending_transitions = []
        self._pending_vector_steps = 0
        return path

    def probe(self, iteration, values: dict):
        path = self.probes_dir / f"iteration-{int(iteration):06d}.pt"
        self._atomic_torch_save(_cpu_copy(values), path)
        return path

    @staticmethod
    def _atomic_torch_save(payload, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            torch.save(_cpu_copy(payload), temporary)
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def checkpoint(self, payload, iteration, tag=None):
        """Save a checkpoint atomically and apply retention to periodic saves.

        Callers decide when periodic checkpoints are due. Tagged saves (such
        as initial/final/emergency) are always saved and are not pruned.
        Existing iteration filenames are never replaced, preserving last-good
        files if a caller accidentally reuses an iteration.
        """
        iteration = int(iteration)
        started = time.monotonic()
        if tag is None:
            path = self.checkpoints_dir / f"iteration-{iteration:06d}.pt"
        elif tag in {"emergency", "interrupted"}:
            path = self.checkpoints_dir / f"{tag}-iteration-{iteration:06d}-{uuid.uuid4().hex}.pt"
        else:
            path = self.checkpoints_dir / f"{tag}-iteration-{iteration:06d}.pt"
        if path.exists():
            raise FileExistsError(f"checkpoint already exists: {path}")
        wrapped = _cpu_copy(payload)
        if tag is not None:
            if not isinstance(wrapped, dict):
                wrapped = {"payload": wrapped}
            wrapped["artifact_tag"] = str(tag)
        self._atomic_torch_save(wrapped, path)
        elapsed = time.monotonic() - started
        size = path.stat().st_size

        if tag is None or tag in {"initial", "final"}:
            self._atomic_json({"path": str(path.relative_to(self.root)),
                               "iteration": iteration, "tag": tag},
                              self.checkpoints_dir / "latest.json")
        if tag is None:
            periodic = sorted(
                self.checkpoints_dir.glob("iteration-*.pt"),
                key=self._iteration_from_path,
            )
            while len(periodic) > self.keep_checkpoints:
                old = periodic.pop(0)
                if old != path:
                    old.unlink()
        # Keep detailed write evidence in the event log while preserving the
        # compact path return expected by the training caller.
        self.event("checkpoint_saved", iteration=iteration, tag=tag,
                   path=str(path.relative_to(self.root)), save_bytes=size,
                   save_seconds=elapsed)
        return path

    @staticmethod
    def _iteration_from_path(path: Path) -> int:
        return int(path.stem.removeprefix("iteration-"))

    @staticmethod
    def _atomic_json(value, path: Path):
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            data = json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"))
            with temporary.open("w", encoding="utf-8") as stream:
                stream.write(data + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
