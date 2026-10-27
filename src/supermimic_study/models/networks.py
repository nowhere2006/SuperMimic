"""Policy, critic, and state-dynamics network initializations.

The layer widths follow Appendix Table A4 of the supplied SuperMimic paper.
SiLU activations and the fixed input scales below are engineering
initialization conventions; they are not specified by that table. The world
model here is only the learned dynamics core. Reward, observation, and
termination heads are intentionally outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .state import StateCodec


@dataclass(frozen=True)
class NetworkConfig:
    """Network dimensions; defaults reproduce the paper's Table A4 widths."""

    motion_encoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    policy_decoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    critic_hidden: tuple[int, ...] = (256, 128)
    dynamics_hidden: tuple[int, ...] = (2048, 2048, 2048)
    history: int = 10
    future_frames: int = 10
    latent_dim: int = 32
    critic_root_error: bool = True

    def __post_init__(self) -> None:
        for name in (
            "motion_encoder_hidden",
            "policy_decoder_hidden",
            "critic_hidden",
            "dynamics_hidden",
        ):
            widths = getattr(self, name)
            if not widths or any(not isinstance(width, int) or width <= 0 for width in widths):
                raise ValueError(f"{name} must contain positive integer widths")
        if self.history <= 0 or self.future_frames <= 0 or self.latent_dim <= 0:
            raise ValueError("history, future_frames, and latent_dim must be positive")
        if not isinstance(self.critic_root_error,bool):
            raise ValueError("critic_root_error must be a bool")


STATE_DIM = 71
ACTION_DIM = 29
CONTEXT_DIM = 33
STATE_DELTA_DIM = 70
OBSERVATION_FRAME_DIM = 93
GOAL_FRAME_DIM = 68


def _require_last_dim(name: str, tensor: torch.Tensor, expected: int) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim == 0 or tensor.shape[-1] != expected:
        actual = "scalar" if tensor.ndim == 0 else tensor.shape[-1]
        raise ValueError(f"{name} must have last dimension {expected}; got {actual}")


def _mlp(input_dim: int, hidden: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_dim
    for next_width in hidden:
        layers.extend((nn.Linear(width, next_width), nn.SiLU()))
        width = next_width
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


def _obs_scale(history: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # Term-major: q, qdot, angular velocity, projected gravity, previous action.
    scale = torch.cat(
        (
            torch.ones(29 * history),
            torch.full((29 * history,), 5.0),
            torch.full((3 * history,), 2.0),
            torch.ones(3 * history),
            torch.ones(29 * history),
        )
    )
    return scale.to(device=device, dtype=dtype)


def _goal_scale(future_frames: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # Per frame: relative q, qdot, projected gravity, root angular velocity,
    # heading-frame linear velocity, height. The flattened goal is frame-major.
    per_frame = torch.cat(
        (
            torch.ones(29),
            torch.full((29,), 5.0),
            torch.ones(3),
            torch.full((3,), 2.0),
            torch.ones(3),
            torch.ones(1),
        )
    )
    return per_frame.to(device=device, dtype=dtype).repeat(future_frames)


class Policy(nn.Module):
    """Encode a future motion goal, then decode it with observation history.

    Inputs use arbitrary leading batch dimensions. Observation is a flattened
    term-major history ``(..., history * 93)``; goal is a flattened frame-major
    future sequence ``(..., future_frames * 68)``. The returned tensor is a
    29D action vector and remains connected to autograd.
    """

    def __init__(self, config: NetworkConfig | None = None) -> None:
        super().__init__()
        self.config = config or NetworkConfig()
        obs_dim = self.config.history * OBSERVATION_FRAME_DIM
        goal_dim = self.config.future_frames * GOAL_FRAME_DIM
        self.motion_encoder = _mlp(goal_dim, self.config.motion_encoder_hidden, self.config.latent_dim)
        self.decoder = _mlp(
            self.config.latent_dim + obs_dim,
            self.config.policy_decoder_hidden,
            ACTION_DIM,
        )
        nn.init.orthogonal_(self.decoder[-1].weight, gain=0.01)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, observation: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        _require_last_dim("observation", observation, self.config.history * OBSERVATION_FRAME_DIM)
        _require_last_dim("goal", goal, self.config.future_frames * GOAL_FRAME_DIM)
        scaled_observation = observation / _obs_scale(
            self.config.history, device=observation.device, dtype=observation.dtype
        )
        latent = self.encode_goal(goal)
        return self.decoder(torch.cat((latent, scaled_observation), dim=-1))

    def encode_goal(self, goal: torch.Tensor) -> torch.Tensor:
        """Return the 32D (by default) tanh motion latent for a future goal."""
        _require_last_dim("goal", goal, self.config.future_frames * GOAL_FRAME_DIM)
        scaled_goal = goal / _goal_scale(
            self.config.future_frames, device=goal.device, dtype=goal.dtype
        )
        return torch.tanh(self.motion_encoder(scaled_goal))


class Critic(nn.Module):
    """Value from raw state, goal, context and training-only relative root pose."""

    def __init__(self, config: NetworkConfig | None = None) -> None:
        super().__init__()
        self.config = config or NetworkConfig()
        self.codec = StateCodec()
        self.value_network = _mlp(
            STATE_DIM + self.config.future_frames * GOAL_FRAME_DIM + CONTEXT_DIM + (4 if self.config.critic_root_error else 0),
            self.config.critic_hidden,
            1,
        )

    def forward(
        self, state: torch.Tensor, goal: torch.Tensor, context: torch.Tensor,
        root_error: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _require_last_dim("state", state, STATE_DIM)
        _require_last_dim("goal", goal, self.config.future_frames * GOAL_FRAME_DIM)
        _require_last_dim("context", context, CONTEXT_DIM)
        scaled_goal = goal / _goal_scale(
            self.config.future_frames, device=goal.device, dtype=goal.dtype
        )
        normalized_context = self.codec.normalize_context(context)
        inputs = [state,scaled_goal,normalized_context]
        if self.config.critic_root_error:
            if root_error is None:
                raise ValueError("critic requires same-time root_error features")
            _require_last_dim("root_error",root_error,4)
            if root_error.shape[:-1] != state.shape[:-1]:
                raise ValueError("root_error batch dimensions must match state")
            # Metres and radians; identity scales 1 m / 1 rad. These are critic
            # input features, distinct from world-model loss calibration.
            inputs.append(root_error)
        return self.value_network(torch.cat(inputs, dim=-1))


class DynamicsModel(nn.Module):
    """Predict a canonical 70D state delta and integrate it into simulator state.

    Input state (71D), action (29D), and context (33D) are passed as separate
    tensors. State canonicalization and delta integration belong to StateCodec.
    """

    def __init__(self, config: NetworkConfig | None = None) -> None:
        super().__init__()
        self.config = config or NetworkConfig()
        self.codec = StateCodec()
        self.delta_network = _mlp(STATE_DIM + ACTION_DIM + CONTEXT_DIM, self.config.dynamics_hidden, STATE_DELTA_DIM)
        nn.init.zeros_(self.delta_network[-1].weight)
        nn.init.zeros_(self.delta_network[-1].bias)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        _require_last_dim("state", state, STATE_DIM)
        _require_last_dim("action", action, ACTION_DIM)
        _require_last_dim("context", context, CONTEXT_DIM)
        canonical_state = self.codec.canonical(state)
        normalized_context = self.codec.normalize_context(context)
        predicted_delta = self.delta_network(torch.cat((canonical_state, action, normalized_context), dim=-1))
        return self.codec.integrate(state, predicted_delta)
