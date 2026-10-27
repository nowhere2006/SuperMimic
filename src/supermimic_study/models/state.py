"""Explicit, replaceable state conventions for initialization and dynamics.

The paper does not enumerate its state vector or integration parameterization.
These are engineering conventions informed by the reference implementation,
not recovered author code: 71-d MuJoCo-layout state, 71 canonical features,
and 70 velocity/pose corrections at 50 Hz.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

STATE_DIM = 71
ACTION_DIM = 29
CONTEXT_DIM = 33
DELTA_DIM = 70


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return F.normalize(q, dim=-1, eps=1e-8)


def quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw), dim=-1)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    cross = 2 * torch.linalg.cross(q[..., 1:], v, dim=-1)
    return v + q[..., :1] * cross + torch.linalg.cross(q[..., 1:], cross, dim=-1)


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_rotate(quat_conjugate(q), v)


def heading_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat_normalize(q).unbind(-1)
    half_yaw = 0.5 * torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    zeros = torch.zeros_like(half_yaw)
    return torch.stack((half_yaw.cos(), zeros, zeros, half_yaw.sin()), dim=-1)


def rotation_features(q: torch.Tensor) -> torch.Tensor:
    """First two rotation-matrix columns, flattened row-major (6 values)."""
    w, x, y, z = quat_normalize(q).unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*y+w*z),
                        1-2*(x*x+z*z), 2*(x*z-w*y), 2*(y*z+w*x)), dim=-1)


def quat_from_rotation_vector(vector: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    # sinc gives the finite small-angle limit sin(angle/2)/angle = 1/2.
    xyz = vector * (0.5 * torch.sinc(angle / (2 * math.pi)))
    return torch.cat(((angle / 2).cos(), xyz), dim=-1)


class StateCodec(nn.Module):
    """s=[p_env(3), q_wxyz(4), joints(29), v_world(3), omega_body(3), qdot(29)]."""
    def __init__(self) -> None:
        super().__init__()
        self.dt = 0.02
        def repeat(values: tuple, sizes: tuple) -> torch.Tensor:
            return torch.cat([torch.full((n,), float(v)) for v, n in zip(values, sizes, strict=True)])
        self.register_buffer("feature_scale", repeat((1, 1, 1, 2, 1, 5), (1, 6, 3, 3, 29, 29)))
        self.register_buffer("delta_scale", repeat((0.3, 0.5, 2, 0.01, 0.02, 0.02), (3, 3, 29, 3, 3, 29)))
        self.register_buffer("context_center", torch.tensor([0.75] + [0.0]*32))
        self.register_buffer("context_scale", torch.tensor([0.45] + [0.01]*29 + [0.05]*3))

    @staticmethod
    def _check(value: torch.Tensor, width: int, label: str) -> None:
        if value.ndim < 2 or value.shape[-1] != width:
            raise ValueError(f"{label} must have a batch dimension and final width {width}")

    def canonical(self, state: torch.Tensor) -> torch.Tensor:
        self._check(state, STATE_DIM, "state")
        q = quat_normalize(state[..., 3:7])
        heading = heading_quat(q)
        local_orientation = quat_multiply(quat_conjugate(heading), q)
        return torch.cat((state[..., 2:3], rotation_features(local_orientation),
                          quat_rotate_inverse(heading, state[..., 36:39]),
                          state[..., 39:42], state[..., 7:36], state[..., 42:71]), dim=-1) / self.feature_scale

    def normalize_context(self, context: torch.Tensor) -> torch.Tensor:
        self._check(context, CONTEXT_DIM, "context")
        return (context - self.context_center) / self.context_scale

    def integrate(self, state: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        self._check(state, STATE_DIM, "state")
        self._check(delta, DELTA_DIM, "delta")
        if state.shape[:-1] != delta.shape[:-1]:
            raise ValueError("state and delta batch dimensions must agree")
        d = delta * self.delta_scale
        q = quat_normalize(state[..., 3:7])
        heading = heading_quat(q)
        v_new = state[..., 36:39] + quat_rotate(heading, d[..., :3])
        omega_new = state[..., 39:42] + d[..., 3:6]
        qdot_new = state[..., 42:71] + d[..., 6:35]
        p_new = state[..., :3] + self.dt*v_new + quat_rotate(heading, d[..., 35:38])
        q_new = quat_normalize(quat_multiply(q, quat_from_rotation_vector(self.dt*omega_new+d[..., 38:41])))
        joint_new = state[..., 7:36] + self.dt*qdot_new + d[..., 41:70]
        return torch.cat((p_new, q_new, joint_new, v_new, omega_new, qdot_new), dim=-1)
