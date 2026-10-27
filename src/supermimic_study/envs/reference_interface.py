"""Connect the complete reference library to mjlab commands and policy inputs.

This is a deterministic initialization profile: fixed clips, no tracking
reward, no return-based reset sampler, no sensor noise or domain randomization.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from supermimic_study.data import MotionLibrary
from supermimic_study.models.state import heading_quat, quat_normalize, quat_rotate_inverse
from .g1 import make_g1_diagnostic_cfg


def reference_goal_features(reference: dict[str, torch.Tensor], default_pos: torch.Tensor, root_index: int) -> torch.Tensor:
    """68-d per-frame encoding, an explicit reference-implementation convention."""
    q = quat_normalize(reference["body_quat_w"][..., root_index, :])
    down = torch.zeros_like(reference["body_pos_w"][..., root_index, :])
    down[..., 2] = -1
    return torch.cat((reference["joint_pos"]-default_pos, reference["joint_vel"],
                      quat_rotate_inverse(q, down),
                      quat_rotate_inverse(q, reference["body_ang_vel_w"][..., root_index, :]),
                      quat_rotate_inverse(heading_quat(q), reference["body_lin_vel_w"][..., root_index, :]),
                      reference["body_pos_w"][..., root_index, 2:3]), dim=-1)


@dataclass(kw_only=True)
class ReferenceMotionCommandCfg(CommandTermCfg):
    library_directory: str
    clip_names: tuple[str, ...]
    future_frames: int = 10

    def build(self, env: ManagerBasedRlEnv) -> "ReferenceMotionCommand":
        return ReferenceMotionCommand(self, env)


class ReferenceMotionCommand(CommandTerm):
    def __init__(self, cfg: ReferenceMotionCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        if not math.isclose(env.step_dt, 0.02, abs_tol=1e-9) or cfg.future_frames < 1:
            raise ValueError("initialization reference interface requires 50 Hz and a positive horizon")
        self.library = MotionLibrary.from_directory(cfg.library_directory, device=env.device)
        robot = env.scene["robot"]
        if robot.joint_names != self.library.joint_names or robot.body_names != self.library.body_names:
            raise ValueError("reference and runtime robot joint/body orders differ")
        if len(cfg.clip_names) != env.num_envs:
            raise ValueError("provide exactly one fixed clip name per initialization environment")
        self.clip_ids = torch.tensor([self.library.clip_names.index(n) for n in cfg.clip_names], device=self.device)
        self.frame_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.finished = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last = torch.tensor([self.library.frame_counts[int(i)]-1 for i in self.clip_ids], device=self.device)
        self._default_pos = robot.data.default_joint_pos.clone()
        self._root = self.library.body_names.index("pelvis")
        self._goal = torch.zeros((self.num_envs, cfg.future_frames*68), device=self.device)
        self._update_command(None)

    @property
    def command(self) -> torch.Tensor:
        return self._goal

    @property
    def reference(self) -> dict[str, torch.Tensor]:
        """Current reference with scene origin applied to world positions exactly once."""
        result = self.library.get_frames(self.clip_ids, self.frame_ids)
        result["body_pos_w"] += self._env.scene.env_origins[:, None, :]
        return result

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # Deterministic initialization only; the training reset sampler is not defined here.
        self.frame_ids[env_ids] = 0
        self.finished[env_ids] = False
        self._update_command(env_ids)

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        if ids.numel() == 0:
            return
        frames = torch.minimum(
            self.frame_ids[ids, None] + torch.arange(self.cfg.future_frames, device=self.device)[None, :],
            self._last[ids, None],
        )
        clips = self.clip_ids[ids, None].expand_as(frames)
        ref = self.library.get_frames(clips.flatten(), frames.flatten())
        default = self._default_pos[ids, None, :].expand(-1, self.cfg.future_frames, -1).reshape(-1, 29)
        self._goal[ids] = reference_goal_features(ref, default, self._root).reshape(len(ids), -1)

    def compute(self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Advance only positive integral control steps; reset(dt=0) never advances time."""
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        duration = torch.as_tensor(dt, dtype=torch.float32, device=self.device)
        if duration.ndim == 0:
            duration = duration.expand(len(ids))
        elif duration.shape == (self.num_envs,):
            duration = duration[ids]
        else:
            raise ValueError("dt must be scalar or per-environment duration")
        steps = duration*self.library.fps
        if not torch.isfinite(steps).all() or (steps < 0).any() or not torch.allclose(steps, steps.round(), atol=1e-6, rtol=0):
            raise ValueError("reference clock must advance by nonnegative integral 50 Hz steps")
        new = self.frame_ids[ids] + steps.round().long()
        self.finished[ids] = new >= self._last[ids]
        self.frame_ids[ids] = torch.minimum(new, self._last[ids])
        self._update_command(ids)


def make_initialization_env_cfg(
    library_directory: str | Path,
    clip_names: tuple[str, ...] = ("walk1_subject1", "dance1_subject1"),
    *, seed: int = 0,
) -> ManagerBasedRlEnvCfg:
    cfg = make_g1_diagnostic_cfg(num_envs=len(clip_names), seed=seed)
    cfg.commands = {"motion": ReferenceMotionCommandCfg(
        resampling_time_range=(1e9, 1e9), library_directory=str(library_directory), clip_names=clip_names,
    )}
    cfg.observations["policy"] = ObservationGroupCfg(
        terms={
            "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, params={"biased": True}),
            "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
            "base_ang_vel": ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}),
            "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
            "previous_action": ObservationTermCfg(func=mdp.last_action),
        }, history_length=10, concatenate_terms=True, enable_corruption=False,
    )
    cfg.observations["goal"] = ObservationGroupCfg(
        terms={"motion": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "motion"})},
        concatenate_terms=True, enable_corruption=False,
    )
    return cfg


def read_model_state(env: ManagerBasedRlEnv) -> torch.Tensor:
    if env.sim.data.qpos.shape[-1] != 36 or env.sim.data.qvel.shape[-1] != 35:
        raise ValueError("model state reader requires the 29-joint floating G1")
    result = torch.cat((env.sim.data.qpos[:], env.sim.data.qvel[:]), dim=-1).detach().clone()
    result[:, :3] -= env.scene.env_origins
    return result


def read_initialization_context(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Only the unrandomized initialization profile, not a general DR context reader."""
    if set(env.cfg.events) != {"reset_scene_to_default"}:
        raise ValueError("context reader is only valid for the unrandomized initialization profile")
    robot = env.scene["robot"]
    _, names = robot.find_geoms(r"^(left|right)_foot[1-7]_collision$")
    if not names:
        raise ValueError("G1 foot collision geometry not found")
    friction = [float(env.sim.mj_model.geom(f"robot/{name}").friction[0]) for name in names]
    if max(friction)-min(friction) > 1e-6:
        raise ValueError("initialization expects one shared nominal foot friction")
    mu = torch.full((env.num_envs, 1), friction[0], device=env.device)
    nominal_com_offset = torch.zeros((env.num_envs, 3), device=env.device)
    return torch.cat((mu, robot.data.encoder_bias.clone(), nominal_com_offset), dim=-1)
