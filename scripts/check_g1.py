"""Check G1 state/action/reset interfaces, without motion data or learning."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(args: argparse.Namespace, report: dict) -> None:
    cache = ROOT / ".cache"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    import torch
    import warp as wp

    wp.config.kernel_cache_dir = str(cache / "warp")
    from mjlab.envs import ManagerBasedRlEnv
    from supermimic_study.envs.g1 import (
        describe_joint_interface,
        make_g1_diagnostic_cfg,
        read_g1_state,
    )

    report["packages"] = {
        name: metadata.version(name)
        for name in ("mjlab", "torch", "mujoco", "mujoco-warp", "warp-lang")
    }
    source = metadata.distribution("mjlab").read_text("direct_url.json")
    report["mjlab_source"] = json.loads(source) if source else None
    if not source or "8bb2bc9e187797689ec82c78c35019afd2fd0376" not in source:
        raise RuntimeError("mjlab is not installed from the expected pinned commit")
    if args.device != "cpu" and not args.device.startswith("cuda"):
        raise ValueError("Use cpu or cuda[:index], not MPS")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    cfg = make_g1_diagnostic_cfg(num_envs=args.num_envs, seed=args.seed)
    if args.steps * cfg.decimation * cfg.sim.mujoco.timestep >= cfg.episode_length_s:
        raise ValueError("Diagnostic rollout must end before its time limit")
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    try:
        obs, _ = env.reset()
        robot = env.scene["robot"]
        term = env.action_manager.get_term("joint_pos")
        initial = read_g1_state(env)
        reference_copy = {key: value.clone() for key, value in initial.items()}
        mapping = describe_joint_interface(env)
        action_dim = env.action_manager.total_action_dim
        if len(mapping) != action_dim or len({row["joint_index"] for row in mapping}) != action_dim:
            raise AssertionError("invalid or duplicated action-to-joint mapping")
        for row in mapping:
            if robot.joint_names[row["joint_index"]] != row["joint_name"]:
                raise AssertionError("joint name/index mismatch")
        torch.testing.assert_close(initial["joint_pos"], robot.data.default_joint_pos)
        torch.testing.assert_close(
            initial["root_pos_w"],
            robot.data.default_root_state[:, :3] + env.scene.env_origins,
        )
        for key, value in initial.items():
            if not torch.isfinite(value).all():
                raise AssertionError(f"non-finite initial {key}")
            torch.testing.assert_close(obs["diagnostic"][key], value)
        report["interface"] = {
            "num_envs": env.num_envs,
            "joint_count": robot.num_joints,
            "action_dim": action_dim,
            "body_count": robot.num_bodies,
            "joint_names": list(robot.joint_names),
            "body_names": list(robot.body_names),
            "action_to_joint": mapping,
            "state_shapes": {key: list(value.shape) for key, value in initial.items()},
            "root_quaternion_order": "w,x,y,z",
            "root_frame": "world",
            "physics_dt_s": env.physics_dt,
            "control_dt_s": env.step_dt,
            "auto_reset": cfg.auto_reset,
            "initial_root_pos_w": initial["root_pos_w"].cpu().tolist(),
            "env_origins_w": env.scene.env_origins.cpu().tolist(),
            "use_cuda_graph": env.sim.use_cuda_graph,
        }
        report["checks"]["load_and_map"] = {"status": "passed"}
        print("G1 loaded; mapping and initial state checked", flush=True)

        # A zero raw action requests the configured default pose, not zero torque.
        zero = torch.zeros((env.num_envs, action_dim), device=env.device)
        obs, reward, terminated, truncated, _ = env.step(zero)
        torch.testing.assert_close(
            robot.data.joint_pos_target[:, term.target_ids],
            robot.data.default_joint_pos[:, term.target_ids] - robot.data.encoder_bias[:, term.target_ids],
        )
        report["checks"]["zero_action_default_target"] = {"status": "passed"}

        # Exercise one named action column in world 0, while other worlds stay at zero.
        selected = next(row for row in mapping if row["joint_name"] == "left_elbow_joint")
        action = zero.clone()
        action[0, selected["action_index"]] = 0.2
        obs, reward, terminated, truncated, _ = env.step(action)
        expected = term.offset + term.scale * action
        expected = expected - robot.data.encoder_bias[:, term.target_ids]
        torch.testing.assert_close(robot.data.joint_pos_target[:, term.target_ids], expected)
        target_delta = float(expected[0, selected["action_index"]] - selected["default_pos_rad"])
        report["checks"]["named_action_target"] = {
            "status": "passed",
            "world": 0,
            "joint": selected["joint_name"],
            "raw_action": 0.2,
            "target_delta_rad": target_delta,
            "target_position_rad": float(expected[0, selected["action_index"]]),
            "measured_position_after_one_control_step_rad": float(
                robot.data.joint_pos[0, selected["joint_index"]]
            ),
        }
        for _ in range(args.steps - 2):
            obs, reward, terminated, truncated, _ = env.step(zero)
        current = read_g1_state(env)
        for key, value in current.items():
            if not torch.isfinite(value).all():
                raise AssertionError(f"non-finite stepped {key}")
            torch.testing.assert_close(obs["diagnostic"][key], value)
            torch.testing.assert_close(initial[key], reference_copy[key])
        if torch.allclose(current["joint_vel"], initial["joint_vel"]):
            raise AssertionError("joint dynamics did not advance")
        if not torch.isfinite(reward).all() or torch.any(terminated | truncated):
            raise AssertionError("unexpected reward or episode end in short diagnostic rollout")
        report["checks"]["step_and_snapshot"] = {
            "status": "passed", "control_steps": args.steps,
            "final_root_pos_w": current["root_pos_w"].cpu().tolist(),
            "max_abs_joint_velocity_rad_s": float(current["joint_vel"].abs().max()),
        }
        print("Action targets, physical stepping and snapshot ownership checked", flush=True)

        # Preserve evidence of unselected worlds across a real partial reset.
        ids = torch.tensor([0], device=env.device, dtype=torch.long)
        env.reset(env_ids=ids)
        partial = read_g1_state(env)
        for key in initial:
            torch.testing.assert_close(partial[key][0], initial[key][0], atol=1e-5, rtol=1e-5)
            if env.num_envs > 1:
                torch.testing.assert_close(partial[key][1:], current[key][1:], atol=1e-5, rtol=1e-5)
        report["checks"]["selective_reset"] = {
            "status": "passed", "reset_worlds": [0],
            "unselected_worlds_checked": env.num_envs - 1,
        }

        # Force only the episode counter to its limit; this is a test injection,
        # not a motion-tracking failure criterion or a training transition.
        env.episode_length_buf[0] = env.max_episode_length - 1
        obs, reward, terminated, truncated, _ = env.step(zero)
        if torch.any(terminated) or not bool(truncated[0]) or torch.any(truncated[1:]):
            raise AssertionError("time limit was not distinguished from task termination")
        final = read_g1_state(env)
        final_position = final["root_pos_w"][0].cpu().tolist()
        reset_position = initial["root_pos_w"][0].cpu().tolist()
        if torch.allclose(final["root_pos_w"][0], initial["root_pos_w"][0], atol=1e-7, rtol=0):
            raise AssertionError("timeout test did not retain a distinct terminal physical state")
        torch.testing.assert_close(obs["diagnostic"]["root_pos_w"], final["root_pos_w"])
        try:
            env.step(zero)
        except RuntimeError as exc:
            if "must be reset" not in str(exc):
                raise
        else:
            raise AssertionError("stepping an unreset completed world was not rejected")
        report["checks"]["manual_timeout_boundary"] = {
            "status": "passed", "test_injection": "episode counter only",
            "truncated_worlds": [0], "terminated_worlds": [],
            "terminal_root_pos_w": final_position, "reset_root_pos_w": reset_position,
            "next_step_requires_reset": True,
        }
        env.reset()
        reset = read_g1_state(env)
        for key in initial:
            torch.testing.assert_close(reset[key], initial[key], atol=1e-5, rtol=1e-5)
        report["checks"]["full_reset"] = {"status": "passed"}
        print("Selective reset, terminal-state retention and full reset checked", flush=True)
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "output/g1-interface.json")
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 2:
        parser.error("num-envs must be positive and steps must be at least 2")
    report = {
        "status": "running", "scope": "G1 diagnostic interface; no reference motion or training",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(),
        "device": args.device, "seed": args.seed, "checks": {},
    }
    try:
        run(args, report)
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        traceback.print_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"G1 interface check: {report['status']}; report: {args.output}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
