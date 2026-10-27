"""Per-environment transition replay for online world-model fitting.

Each ``add`` call records one vectorized simulator step.  The caller must pass
the transition's *pre-reset* next state and the observation actually consumed
by the policy at that step.  Replay keeps environment-local sequence numbers
and episode ids so sampled windows cannot bridge a reset or an overwritten
ring slot.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


_WIDTHS = {
    "state": 71,
    "action": 29,
    "next_state": 71,
    "context": 33,
    "observation": 930,
    "previous_action": 29,
    "policy_action": 29,
    "executed_action": 29,
    "previous_policy_action": 29,
    "previous_executed_action": 29,
}
_SCALARS = ("reward", "terminated", "truncated", "clip_id", "frame_id", "next_frame_id")
_REQUIRED = (*_WIDTHS, *_SCALARS)


class ReplayBuffer:
    """A bounded ring with independent chronology for every vectorized env.

    ``capacity`` is per environment.  Continuous tensors are detached and
    copied on insertion, which prevents simulator-owned buffers or autograd
    graphs from being retained by replay.
    """

    def __init__(self, num_envs: int, capacity: int = 1000,
                 device: str | torch.device = "cpu", seed: int = 0) -> None:
        if num_envs <= 0 or capacity <= 0:
            raise ValueError("num_envs and capacity must be positive")
        self.num_envs = int(num_envs)
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self._rings: list[list[dict[str, Any] | None]] = [
            [None] * self.capacity for _ in range(self.num_envs)
        ]
        self._total = [0] * self.num_envs
        self._episode = [0] * self.num_envs
        self._rng = torch.Generator(device="cpu").manual_seed(seed)
        self._optional_keys: set[str] | None = None

    @staticmethod
    def _as_env_batch(name: str, value: Any, n: int) -> torch.Tensor:
        try:
            tensor = torch.as_tensor(value)
        except Exception as exc:
            raise ValueError(f"{name} must be convertible to a tensor") from exc
        if tensor.ndim == 0 or tensor.shape[0] != n:
            raise ValueError(f"{name} must have leading dimension num_envs={n}")
        if name in _WIDTHS:
            width = _WIDTHS[name]
            if tensor.ndim != 2 or tensor.shape[1] != width:
                raise ValueError(f"{name} must have shape [num_envs, {width}]")
            if not tensor.is_floating_point():
                raise ValueError(f"{name} must be floating point")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains non-finite values")
        else:
            if tensor.numel() != n:
                raise ValueError(f"{name} must contain one scalar per environment")
            tensor = tensor.reshape(n)
            if name in ("reward",) and (not tensor.is_floating_point() or not torch.isfinite(tensor).all()):
                raise ValueError("reward must contain finite floating-point values")
            if name in ("terminated", "truncated") and tensor.dtype != torch.bool:
                raise ValueError(f"{name} must be boolean")
            if name in ("clip_id", "frame_id", "next_frame_id"):
                if tensor.is_floating_point() and (not torch.isfinite(tensor).all() or not torch.equal(tensor, tensor.round())):
                    raise ValueError(f"{name} must contain finite integer ids")
        return tensor

    def add(self, batch: Mapping[str, Any]) -> None:
        """Append exactly one transition for every environment.

        Required floating fields are state(71), action(29), next_state(71),
        context(33), observation(930), previous_action(29), reward.  The
        remaining required fields are boolean end flags and scalar clip/frame
        ids.  Optional fields (for example ``clean_observation``) are copied
        and retained without imposing a schema on diagnostics.
        """
        missing = sorted(set(_REQUIRED) - set(batch))
        if missing:
            raise ValueError(f"replay batch missing required keys: {', '.join(missing)}")
        checked = {key: self._as_env_batch(key, batch[key], self.num_envs) for key in _REQUIRED}
        if not torch.equal(checked["action"],checked["executed_action"]):
            raise ValueError("action alias must equal executed_action")
        if not torch.equal(checked["previous_action"],checked["previous_policy_action"]):
            raise ValueError("previous_action alias must equal previous_policy_action")
        # Consecutive source-frame ids are required for ordinary transitions.
        # A clipped terminal may clamp its successor id to the final frame.
        for env in range(self.num_envs):
            frame = int(checked["frame_id"][env].item())
            next_frame = int(checked["next_frame_id"][env].item())
            terminal_clip_end = bool(checked["truncated"][env].item())
            if next_frame != frame + 1 and not (terminal_clip_end and next_frame == frame):
                raise ValueError(
                    f"next_frame_id must equal frame_id + 1 (or clamp a truncated clip tail); env {env} has {frame}->{next_frame}"
                )

        optional: dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if key in checked:
                continue
            tensor = torch.as_tensor(value)
            if tensor.ndim == 0 or tensor.shape[0] != self.num_envs:
                raise ValueError(f"optional field {key} must have leading dimension num_envs={self.num_envs}")
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise ValueError(f"optional field {key} contains non-finite values")
            if key == "clean_observation" and (tensor.ndim != 2 or tensor.shape[1] != 93):
                raise ValueError("clean_observation must have shape [num_envs, 93]")
            optional[key] = tensor
        optional_keys = set(optional)
        if self._optional_keys is not None and optional_keys != self._optional_keys:
            raise ValueError("optional replay fields must stay consistent across add calls")

        # Validate all inputs before mutating any ring, keeping a vectorized add atomic.
        for env in range(self.num_envs):
            entry: dict[str, Any] = {
                key: value[env].detach().to(self.device).clone()
                for key, value in checked.items()
            }
            entry.update({key: value[env].detach().to(self.device).clone() for key, value in optional.items()})
            entry["_seq"] = self._total[env]
            entry["_episode"] = self._episode[env]
            self._rings[env][self._total[env] % self.capacity] = entry
            self._total[env] += 1
            if bool(checked["terminated"][env].item() or checked["truncated"][env].item()):
                self._episode[env] += 1
        if self._optional_keys is None:
            self._optional_keys = optional_keys

    def _chronological(self, env: int) -> list[dict[str, Any]]:
        total = self._total[env]
        first = max(0, total - self.capacity)
        result: list[dict[str, Any]] = []
        for seq in range(first, total):
            item = self._rings[env][seq % self.capacity]
            if item is not None and item["_seq"] == seq:
                result.append(item)
        return result

    def _windows(self, length: int) -> list[tuple[int, list[dict[str, Any]]]]:
        candidates: list[tuple[int, list[dict[str, Any]]]] = []
        for env in range(self.num_envs):
            items = self._chronological(env)
            for start in range(0, len(items) - length + 1):
                window = items[start:start + length]
                if len(window) != length:
                    continue
                if any(window[i + 1]["_seq"] != window[i]["_seq"] + 1 for i in range(length - 1)):
                    continue
                if any(window[i + 1]["_episode"] != window[i]["_episode"] for i in range(length - 1)):
                    continue
                if any(not torch.equal(window[i]["clip_id"], window[i + 1]["clip_id"])
                       or not torch.equal(window[i]["context"], window[i + 1]["context"])
                       for i in range(length - 1)):
                    continue
                # The end transition may be terminal/truncated. Nothing after
                # a done transition may be included in the same window.
                if any(bool(x["terminated"].item() or x["truncated"].item()) for x in window[:-1]):
                    continue
                if any(int(window[i]["next_frame_id"].item()) != int(window[i + 1]["frame_id"].item())
                       for i in range(length - 1)):
                    continue
                if any(not torch.allclose(window[i]["next_state"], window[i + 1]["state"], atol=1e-5, rtol=1e-5)
                       for i in range(length - 1)):
                    continue
                if any(not torch.equal(window[i][key],window[i+1][previous_key])
                       for i in range(length-1)
                       for key,previous_key in (("policy_action","previous_policy_action"),
                                                ("executed_action","previous_executed_action"))):
                    continue
                candidates.append((env, window))
        return candidates

    def _draw(self, count: int, upper: int) -> torch.Tensor:
        if count <= 0:
            raise ValueError("batch_size must be positive")
        if upper <= 0:
            raise ValueError("replay contains no eligible samples")
        indices = torch.randint(upper, (count,), generator=self._rng)
        return indices

    def _stack(self, entries: list[dict[str, Any]], key: str) -> torch.Tensor:
        return torch.stack([entry[key] for entry in entries]).to(self.device)

    def sample_windows(self, batch_size: int, length: int) -> dict[str, torch.Tensor]:
        """Sample contiguous transition windows shaped ``[B, length, ...]``."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if length <= 0:
            raise ValueError("length must be positive")
        candidates = self._windows(length)
        if not candidates:
            raise ValueError(
                f"not enough contiguous replay for window length {length}; "
                "windows cannot cross episode boundaries or overwritten gaps"
            )
        chosen = self._draw(batch_size, len(candidates)).tolist()
        selected = [candidates[i][1] for i in chosen]
        keys = [key for key in selected[0][0] if not key.startswith("_")]
        result = {key: torch.stack([torch.stack([step[key] for step in window]) for window in selected])
                  for key in keys}
        result["episode_id"] = torch.tensor(
            [[window[0]["_episode"]] * length for window in selected], device=self.device, dtype=torch.long
        )
        result["env_id"] = torch.tensor([candidates[i][0] for i in chosen], device=self.device, dtype=torch.long)
        return result

    def sample_starts(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample valid imagination starts with current policy history.

        ``history`` is ``[B, 10, 93]``.  The policy-ready flattened vector is
        term-major: each term contributes its ten-frame block before the next
        term. Reconstruct by splitting those blocks and joining at each time
        index. The original 930-vector is also returned unchanged.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        candidates: list[tuple[int, dict[str, Any]]] = []
        for env in range(self.num_envs):
            for entry in self._chronological(env):
                candidates.append((env, entry))
        if not candidates:
            raise ValueError("replay contains no transitions for imagination starts")
        chosen = self._draw(batch_size, len(candidates)).tolist()
        pairs = [candidates[i] for i in chosen]
        entries = [x[1] for x in pairs]
        observation = self._stack(entries, "observation")
        term_widths = (29, 29, 3, 3, 29)
        term_lengths = tuple(width * 10 for width in term_widths)
        if sum(term_lengths) != observation.shape[-1]:
            raise RuntimeError("configured observation terms do not sum to the 930-dimensional observation")
        blocks = observation.split(term_lengths, dim=-1)
        history = torch.cat(
            [block.reshape(batch_size, 10, width) for block, width in zip(blocks, term_widths, strict=True)],
            dim=-1,
        )
        result = {
            "state": self._stack(entries, "state"),
            "observation": observation,
            "history": history,
            "previous_action": self._stack(entries, "previous_action"),
            "previous_policy_action": self._stack(entries,"previous_policy_action"),
            "previous_executed_action": self._stack(entries,"previous_executed_action"),
            "context": self._stack(entries, "context"),
            "clip_id": self._stack(entries, "clip_id"),
            "frame_id": self._stack(entries, "frame_id"),
            "env_id": torch.tensor([x[0] for x in pairs], device=self.device, dtype=torch.long),
            "episode_id": torch.tensor([x[1]["_episode"] for x in pairs], device=self.device, dtype=torch.long),
        }
        if "clean_observation" in entries[0]:
            result["clean_observation"] = self._stack(entries, "clean_observation")
        return result
