"""World-model wrapper with analytic observation/reward and learned termination."""
from __future__ import annotations

import torch
from torch import nn

from .heads import observation_from_state, tracking_reward
from .kinematics import DifferentiableG1Kinematics
from .networks import DynamicsModel, NetworkConfig, _mlp
from .state import quat_normalize, quat_rotate_inverse


class WorldModel(nn.Module):
    """Composable one-step model used by fitting and imagined rollouts.

    Dynamics predicts a 71D next state. Observation and reward are analytic.
    Termination is a learned logit trained with BCE; callers should use its
    sigmoid probability for survival weighting, never hard-threshold in BPTT.
    """

    def __init__(self, dynamics: DynamicsModel | None = None,
                 kinematics: DifferentiableG1Kinematics | None = None,
                 default_q: torch.Tensor | None = None,
                 joint_limits: torch.Tensor | None = None,
                 config: NetworkConfig | None = None):
        super().__init__()
        self.config = config or (dynamics.config if dynamics is not None else NetworkConfig())
        self.dynamics = dynamics or DynamicsModel(self.config)
        self.kinematics = kinematics
        self.register_buffer("default_q", torch.zeros(29) if default_q is None else torch.as_tensor(default_q).float().clone())
        if joint_limits is None:
            joint_limits = torch.tensor([[-float("inf"), float("inf")]]).repeat(29, 1)
        self.register_buffer("joint_limits", torch.as_tensor(joint_limits).float().clone())
        # Learned Bernoulli head sees canonical physical state and four
        # reference-root features (height and pelvis-frame gravity direction).
        self.termination_network = _mlp(71 + 4, self.config.critic_hidden, 1)
        # An engineering prior avoids an initial 0.5 survival probability at
        # every imagined step; this bias is not specified by the paper.
        nn.init.constant_(self.termination_network[-1].bias, -2.0)

    def predict_next(self, state: torch.Tensor, executed_action: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.dynamics(state, executed_action, context)

    def predict_observation(self, state: torch.Tensor, last_policy_action: torch.Tensor,
                            context: torch.Tensor, default_q: torch.Tensor | None = None) -> torch.Tensor:
        return observation_from_state(state, last_policy_action, context,
                                      self.default_q if default_q is None else default_q)

    def predict_reward(self, next_state: torch.Tensor, reference: dict[str, torch.Tensor],
                       policy_action: torch.Tensor, previous_policy_action: torch.Tensor,
                       *, dt: float = 0.02, body_names=None) -> torch.Tensor:
        if self.kinematics is None:
            raise RuntimeError("predict_reward requires a DifferentiableG1Kinematics module")
        return tracking_reward(self.kinematics(next_state), reference, policy_action, previous_policy_action,
                               next_state[..., 7:36], self.joint_limits, dt, body_names or self.kinematics.body_names)

    def _reference_root_features(self, reference: dict[str, torch.Tensor], body_names=None) -> torch.Tensor:
        if body_names is not None:
            names = tuple(body_names)
        elif self.kinematics is not None:
            names = self.kinematics.body_names
        else:
            names = ("pelvis",)
        if "pelvis" not in names:
            raise ValueError("reference body_names must include pelvis")
        idx = names.index("pelvis")
        pos = reference["body_pos_w"][..., idx, :]
        quat = quat_normalize(reference["body_quat_w"][..., idx, :])
        gravity = torch.zeros_like(pos); gravity[..., 2] = -1
        gravity_local = quat_rotate_inverse(quat, gravity)
        return torch.cat((pos[..., 2:3], gravity_local), dim=-1)

    def predict_termination_logit(self, next_state: torch.Tensor,
                                  reference: dict[str, torch.Tensor], body_names=None) -> torch.Tensor:
        root = self._reference_root_features(reference, body_names)
        features = torch.cat((self.dynamics.codec.canonical(next_state), root), dim=-1)
        return self.termination_network(features).squeeze(-1)

    def termination_probability(self, next_state: torch.Tensor,
                                reference: dict[str, torch.Tensor], body_names=None) -> torch.Tensor:
        return torch.sigmoid(self.predict_termination_logit(next_state, reference, body_names))

    def forward(self, state: torch.Tensor, executed_action: torch.Tensor, context: torch.Tensor,
                previous_policy_action: torch.Tensor | None = None,
                reference: dict[str, torch.Tensor] | None = None,
                *, policy_action: torch.Tensor | None = None,
                dt: float = 0.02, body_names=None) -> dict[str, torch.Tensor]:
        """Dynamics consumes executed action; task heads require explicit clean actions.

        In imagination the executed action equals the clean policy action because
        no exploration noise is added, but callers still state both roles.
        """
        if previous_policy_action is not None and policy_action is None:
            raise ValueError("policy_action must be explicit when requesting task heads")
        next_state = self.predict_next(state, executed_action, context)
        result = {"next_state": next_state}
        names = body_names or (self.kinematics.body_names if self.kinematics is not None else None)
        if policy_action is not None:
            result["observation"] = self.predict_observation(next_state, policy_action, context)
        if reference is not None:
            if previous_policy_action is None or policy_action is None:
                raise ValueError("clean policy_action and previous_policy_action are required to calculate reward")
            result["reward"] = self.predict_reward(next_state, reference, policy_action, previous_policy_action,
                                                   dt=dt, body_names=names)
            result["termination_logit"] = self.predict_termination_logit(next_state, reference, names)
            result["termination_probability"] = torch.sigmoid(result["termination_logit"])
        return result
