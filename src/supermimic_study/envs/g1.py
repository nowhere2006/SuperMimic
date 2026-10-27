"""Minimal G1 interface on mjlab, with explicitly named diagnostic state fields.

Reuses mjlab's G1 asset, actuator definitions and action scale. No reference
motion, policy, world model or tracking reward is defined in this module.
"""

from __future__ import annotations

import torch

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.terminations import time_out
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg

STATE_FIELDS = {
    "joint_pos": "joint_pos",
    "joint_vel": "joint_vel",
    "root_pos_w": "root_link_pos_w",
    "root_quat_w": "root_link_quat_w",
    "root_lin_vel_w": "root_link_lin_vel_w",
    "root_ang_vel_w": "root_link_ang_vel_w",
}


def _robot_field(env: ManagerBasedRlEnv, field: str) -> torch.Tensor:
    return getattr(env.scene["robot"].data, field)


def make_g1_diagnostic_cfg(
    *, num_envs: int = 2, seed: int = 0
) -> ManagerBasedRlEnvCfg:
    """Configure physical interaction only, not the paper's tracking MDP.

    5 ms physics steps and four substeps per action follow mjlab's tracking
    example. Randomization and observation noise are absent so interface
    checks have an explicit, fixed configuration.
    """
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=num_envs,
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": get_g1_robot_cfg()},
        ),
        sim=SimulationCfg(
            nconmax=35,
            njmax=250,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
        ),
        decimation=4,
        episode_length_s=10.0,
        seed=seed,
        auto_reset=False,
        actions={
            "joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=dict(G1_ACTION_SCALE),
                use_default_offset=True,
            ),
        },
        observations={
            "diagnostic": ObservationGroupCfg(
                terms={
                    name: ObservationTermCfg(func=_robot_field, params={"field": field})
                    for name, field in STATE_FIELDS.items()
                },
                concatenate_terms=False,
                enable_corruption=False,
            ),
        },
        terminations={"time_limit": TerminationTermCfg(func=time_out, time_out=True)},
    )


def read_g1_state(env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
    """Copy the current simulator state; all root quantities use world frame.

    Joint angles use radians, velocities rad/s, position meters, and linear
    velocity m/s. Root quaternion order is w,x,y,z. Call after env.reset/step;
    manually writing state requires sim.forward before reading derived fields.
    Copies prevent later simulation/reset operations from mutating this snapshot.
    This diagnostic dictionary is not yet the paper's world-model state vector.
    """
    return {
        name: _robot_field(env, field).detach().clone()
        for name, field in STATE_FIELDS.items()
    }


def describe_joint_interface(env: ManagerBasedRlEnv) -> list[dict]:
    """Map each action column to an entity joint and its configured position target."""
    robot = env.scene["robot"]
    term = env.action_manager.get_term("joint_pos")
    rows = []
    for action_index, (joint_index, name) in enumerate(
        zip(term.target_ids.tolist(), term.target_names, strict=True)
    ):
        scale = term.scale
        scale_value = float(scale[0, action_index]) if isinstance(scale, torch.Tensor) else float(scale)
        rows.append({
            "action_index": action_index,
            "joint_index": joint_index,
            "joint_name": name,
            "scale_rad_per_action_unit": scale_value,
            "default_pos_rad": float(robot.data.default_joint_pos[0, joint_index]),
        })
    return rows
