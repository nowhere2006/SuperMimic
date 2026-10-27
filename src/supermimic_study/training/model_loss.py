"""Physical-state losses; SuperTrack-inspired initial contribution balancing.

SuperTrack uses grouped L1 losses with hand-adjusted initial contributions.
Our six groups and one-window automatic calibration are explicit adaptations.
"""
from __future__ import annotations

import torch
from torch import nn

from supermimic_study.models.state import quat_conjugate, quat_multiply, quat_normalize

GROUP_NAMES = ("root_position", "root_rotation", "root_linear_velocity",
               "root_angular_velocity", "joint_position", "joint_velocity")
ERROR_VERSION = "physical70-shortest-rotvec-v1"


def physical_errors(prediction: torch.Tensor, target: torch.Tensor):
    """Six component errors, preserving batch/time axes; labels are simulator states."""
    if prediction.shape != target.shape or prediction.ndim < 2 or prediction.shape[-1] != 71:
        raise ValueError("prediction/target must have matching (..., 71) state shapes")
    relative = quat_normalize(quat_multiply(quat_conjugate(quat_normalize(target[..., 3:7])),
                                           quat_normalize(prediction[..., 3:7])))
    # Canonical shortest rotation. At exactly pi, lexicographic axis sign makes
    # quaternion antipodes agree; the SO(3) log is still discontinuous at this cut.
    w, xyz = relative[..., :1], relative[..., 1:]
    pivot = xyz.gather(-1, xyz.abs().argmax(-1, keepdim=True))
    flip = (w < 0) | ((w == 0) & (pivot < 0))
    relative = torch.where(flip, -relative, relative)
    w, xyz = relative[..., :1], relative[..., 1:]
    sine = torch.linalg.vector_norm(xyz, dim=-1, keepdim=True)
    regular = 2 * torch.atan2(sine, w) / sine.clamp_min(1e-6)
    factor = torch.where(sine < 1e-6, 2 + sine.square()/3, regular)
    rotation = xyz * factor
    return (prediction[..., :3]-target[..., :3], rotation,
            prediction[..., 36:39]-target[..., 36:39],
            prediction[..., 39:42]-target[..., 39:42],
            prediction[..., 7:36]-target[..., 7:36],
            prediction[..., 42:71]-target[..., 42:71])


class WorldModelStateLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mode = config.world_model_loss
        self.floor = config.world_model_scale_floor
        self.register_buffer("group_weights", torch.tensor(config.world_model_group_weights, dtype=torch.float32))
        self.register_buffer("initial_errors", torch.zeros(6))
        self.register_buffer("scales", torch.ones(6))
        self.register_buffer("calibrated", torch.tensor(False))

    @staticmethod
    def group_means(prediction, target):
        # Mean over batch, time and dimensions; each group has its own units.
        return torch.stack([error.abs().mean() for error in physical_errors(prediction, target)])

    @torch.no_grad()
    def calibrate(self, means):
        if self.mode != "supertrack_l1":
            raise ValueError("raw MSE does not use initial-scale calibration")
        if bool(self.calibrated):
            raise RuntimeError("world-model loss is already calibrated; scales are frozen")
        if means.shape != (6,) or not torch.isfinite(means).all() or (means < 0).any():
            raise ValueError("calibration requires six finite nonnegative group errors")
        self.initial_errors.copy_(means.detach())
        self.scales.copy_(means.detach().clamp_min(self.floor))
        self.calibrated.fill_(True)

    def forward(self, prediction, target):
        """Return time-summed, batch-mean loss and time-mean group diagnostics.

        Input axes are [batch, time, state]. Keeping the time sum for state and
        BCE avoids introducing a relative factor of the window length.
        """
        if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 71:
            raise ValueError("loss expects matching [batch, time, 71] states")
        if self.mode == "raw_mse":
            sign = torch.where((prediction[..., 3:7].detach()*target[..., 3:7]).sum(-1, keepdim=True)<0,
                               -1., 1.)
            aligned = torch.cat((target[..., :3], target[..., 3:7]*sign, target[..., 7:]), -1)
            loss = (prediction-aligned).square().sum(-1).mean(0).sum()
            return loss, None, None
        means = self.group_means(prediction, target)
        if not bool(self.calibrated):
            self.calibrate(means)
        contributions = means / self.scales * self.group_weights
        return contributions.sum()*prediction.shape[1], means, contributions

    def summary(self):
        return {"mode": self.mode, "error_version": ERROR_VERSION,
                "group_order": list(GROUP_NAMES), "calibrated": bool(self.calibrated),
                "scale_floor": self.floor, "initial_errors": self.initial_errors.detach().cpu().tolist(),
                "scales": self.scales.detach().cpu().tolist(),
                "group_weights": self.group_weights.detach().cpu().tolist(),
                "floor_hit": (self.initial_errors < self.floor).detach().cpu().tolist() if bool(self.calibrated) else None}

    def validate_saved_state(self, saved, completed_iterations):
        """Validate calibration before loading any training parameters."""
        if set(saved) != set(self.state_dict()):
            raise ValueError("checkpoint world-model loss state fields differ")
        for key in ("scales", "initial_errors", "group_weights"):
            value = saved[key]
            if not isinstance(value, torch.Tensor) or value.shape != (6,) or not torch.isfinite(value).all():
                raise ValueError(f"invalid checkpoint loss {key}")
        flag = saved["calibrated"]
        if not isinstance(flag, torch.Tensor) or flag.shape != () or flag.dtype != torch.bool:
            raise ValueError("invalid checkpoint loss calibrated flag")
        if not torch.equal(saved["group_weights"].cpu(), self.group_weights.cpu()):
            raise ValueError("checkpoint loss group weights differ from configuration")
        initial, scales = saved["initial_errors"], saved["scales"]
        if (initial < 0).any() or (scales <= 0).any():
            raise ValueError("checkpoint loss scales must be positive and errors nonnegative")
        if self.mode == "supertrack_l1":
            if completed_iterations > 0 and not bool(flag):
                raise ValueError("complete training checkpoint lacks calibrated loss scales")
            if bool(flag) and not torch.equal(scales, initial.clamp_min(self.floor)):
                raise ValueError("checkpoint loss scales do not match initial calibration")
        elif bool(flag):
            raise ValueError("raw MSE checkpoint must not contain calibrated scales")
