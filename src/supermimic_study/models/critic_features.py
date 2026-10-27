"""Differentiable training-only root tracking features for the value estimator."""
from __future__ import annotations

import torch

from .state import quat_normalize, quat_rotate_inverse


def _yaw(q):
    w, x, y, z = quat_normalize(q).unbind(-1)
    forward_x = 1-2*(y.square()+z.square())
    forward_y = 2*(w*z+x*y)
    # Yaw is undefined when the forward axis is vertical. Choose zero there;
    # provide safe atan2 inputs so the unused branch cannot produce NaN grads.
    valid = forward_x.square()+forward_y.square() > 1e-12
    return torch.atan2(torch.where(valid,forward_y,torch.zeros_like(forward_y)),
                       torch.where(valid,forward_x,torch.ones_like(forward_x)))


def root_tracking_features(state, reference_position, reference_quaternion):
    """[position error in reference heading frame (m), wrapped yaw error (rad)].

    State and reference must be at the same time and use the same env-local
    origin/world axes. Gradients with respect to predicted state stay live.
    """
    if (state.ndim < 2 or state.shape[-1] != 71 or
            reference_position.shape != (*state.shape[:-1],3) or
            reference_quaternion.shape != (*state.shape[:-1],4)):
        raise ValueError("root features require matching (...,71), (...,3), (...,4) tensors")
    reference_yaw = _yaw(reference_quaternion)
    zeros = torch.zeros_like(reference_yaw)
    heading = torch.stack(((reference_yaw/2).cos(),zeros,zeros,(reference_yaw/2).sin()),-1)
    position = quat_rotate_inverse(heading,state[...,:3]-reference_position)
    difference = _yaw(state[...,3:7])-reference_yaw
    yaw = torch.atan2(difference.sin(),difference.cos())
    return torch.cat((position,yaw.unsqueeze(-1)),-1)
