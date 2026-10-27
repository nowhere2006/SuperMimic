"""Analytic observation, tracking reward, and hard termination definitions."""
from __future__ import annotations

from collections.abc import Sequence
import torch
from torch import nn
from torch.nn import functional as F

from .state import quat_normalize, quat_multiply, quat_conjugate, quat_rotate, quat_rotate_inverse, heading_quat

TRACKING_BODY_NAMES = (
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
)
ANCHOR_BODY_NAME = "torso_link"
ROOT_BODY_NAME = "pelvis"


def observation_from_state(state: torch.Tensor, last_policy_action: torch.Tensor,
                           context: torch.Tensor, default_q: torch.Tensor) -> torch.Tensor:
    """93D proprioception; the last-action slot stores clean policy output."""
    if state.shape[-1] != 71 or last_policy_action.shape[-1] != 29 or context.shape[-1] != 33:
        raise ValueError("expected state/action/context widths 71/29/33")
    q0 = torch.as_tensor(default_q, dtype=state.dtype, device=state.device)
    q = state[..., 7:36] + context[..., 1:30] - q0
    omega = state[..., 39:42]  # pelvis body-frame angular velocity (IMU convention)
    down = torch.zeros_like(state[..., :3]); down[..., 2] = -1
    gravity = quat_rotate_inverse(quat_normalize(state[..., 3:7]), down)
    return torch.cat((q, state[..., 42:71], omega, gravity, last_policy_action), dim=-1)


def _rotation_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Stable SO(3) geodesic angle, finite at identity (including its gradient)."""
    rel = quat_normalize(quat_multiply(quat_normalize(a), quat_conjugate(quat_normalize(b))))
    return 2 * torch.atan2(torch.linalg.vector_norm(rel[..., 1:], dim=-1), rel[..., 0].abs().clamp_min(1e-12))


def _select(data: dict[str, torch.Tensor], body_names: Sequence[str] | None,
            requested: Sequence[str], key: str) -> torch.Tensor:
    value = data[key]
    names = tuple(body_names) if body_names is not None else tuple(requested)
    if value.shape[-2] == len(requested) and names == tuple(requested):
        return value
    missing = [n for n in requested if n not in names]
    if missing:
        raise ValueError(f"body data missing required names: {missing}")
    indices = torch.tensor([names.index(n) for n in requested], device=value.device)
    return value.index_select(-2, indices)


def tracking_reward(current_body: dict[str, torch.Tensor], reference: dict[str, torch.Tensor],
                    policy_action: torch.Tensor, previous_policy_action: torch.Tensor,
                    joint_pos: torch.Tensor, limits: torch.Tensor,
                    dt: float = 0.02, body_names: Sequence[str] | None = None) -> torch.Tensor:
    """BeyondMimic tracking reward used by the paper, without collision term.

    Tracking uses the official mjlab G1 subset and torso_link anchor. The reward
    sums six exponentials plus action-rate and joint-limit penalties. The paper
    keeps official scales/stds, sets joint-limit weight to -2, and drops collision.
    """
    if limits.shape[-1] != 2 or joint_pos.shape[-1] != 29:
        raise ValueError("joint_pos must be 29D and limits must end in [lower, upper]")
    names = tuple(body_names) if body_names is not None else TRACKING_BODY_NAMES
    cur_p = _select(current_body, names, TRACKING_BODY_NAMES, "body_pos_w")
    ref_p = _select(reference, names, TRACKING_BODY_NAMES, "body_pos_w")
    cur_q = _select(current_body, names, TRACKING_BODY_NAMES, "body_quat_w")
    ref_q = _select(reference, names, TRACKING_BODY_NAMES, "body_quat_w")
    cur_v = _select(current_body, names, TRACKING_BODY_NAMES, "body_lin_vel_w")
    ref_v = _select(reference, names, TRACKING_BODY_NAMES, "body_lin_vel_w")
    cur_w = _select(current_body, names, TRACKING_BODY_NAMES, "body_ang_vel_w")
    ref_w = _select(reference, names, TRACKING_BODY_NAMES, "body_ang_vel_w")
    anchor = TRACKING_BODY_NAMES.index(ANCHOR_BODY_NAME)
    # Global anchor terms.
    root_pos = torch.exp(-((cur_p[..., anchor, :] - ref_p[..., anchor, :]) ** 2).sum(-1) / 0.3**2)
    root_ori = torch.exp(-_rotation_error(cur_q[..., anchor, :], ref_q[..., anchor, :]).square() / 0.4**2)
    # Match mjlab's relative pose convention: yaw-align the reference anchor,
    # align horizontal position and reference anchor height, then compare links.
    yaw = heading_quat(quat_multiply(cur_q[..., anchor, :], quat_conjugate(ref_q[..., anchor, :])))
    align_pos = cur_p[..., anchor, :].clone()
    align_pos[..., 2] = ref_p[..., anchor, 2]
    rel_p = align_pos[..., None, :] + quat_rotate(yaw[..., None, :], ref_p - ref_p[..., anchor:anchor+1, :])
    rel_q = quat_multiply(yaw[..., None, :], ref_q)
    body_pos = torch.exp(-((rel_p-cur_p).square().sum(-1).mean(-1))/0.3**2)
    body_ori = torch.exp(-_rotation_error(rel_q, cur_q).square().mean(-1)/0.4**2)
    body_lin = torch.exp(-((cur_v-ref_v).square().sum(-1).mean(-1))/1.0**2)
    body_ang = torch.exp(-((cur_w-ref_w).square().sum(-1).mean(-1))/3.14**2)
    action_penalty = -0.1 * (policy_action-previous_policy_action).square().sum(-1)
    lim = torch.as_tensor(limits, dtype=joint_pos.dtype, device=joint_pos.device)
    lower, upper = lim[..., 0], lim[..., 1]
    limit_penalty = -2.0 * (F.relu(joint_pos-upper)+F.relu(lower-joint_pos)).sum(-1)
    # Manager-based RL convention scales each reward term by its control period.
    return dt * (0.5*root_pos + 0.5*root_ori + body_pos + body_ori + body_lin + body_ang + action_penalty + limit_penalty)


def hard_termination(current_body: dict[str, torch.Tensor], reference: dict[str, torch.Tensor],
                     body_names: Sequence[str] | None = None) -> torch.Tensor:
    """Paper rule: pelvis height error > .25 m or gravity-z difference > .8."""
    names = tuple(body_names) if body_names is not None else (ROOT_BODY_NAME,)
    if ROOT_BODY_NAME not in names:
        raise ValueError("body_names must include pelvis for root termination")
    idx = names.index(ROOT_BODY_NAME)
    p, rp = current_body["body_pos_w"][..., idx, :], reference["body_pos_w"][..., idx, :]
    q, rq = current_body["body_quat_w"][..., idx, :], reference["body_quat_w"][..., idx, :]
    down = torch.zeros_like(p); down[..., 2] = -1
    gz = quat_rotate_inverse(q, down)[..., 2]
    rgz = quat_rotate_inverse(rq, down)[..., 2]
    return ((p[..., 2]-rp[..., 2]).abs() > 0.25) | ((gz-rgz).abs() > 0.8)
