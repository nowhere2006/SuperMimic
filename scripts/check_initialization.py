"""Wire real reference targets to randomly initialized networks; never train."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parents[1]


def run(args: argparse.Namespace, report: dict) -> None:
    os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import torch
    import warp as wp
    wp.config.kernel_cache_dir = str(ROOT / ".cache/warp")
    from mjlab.envs import ManagerBasedRlEnv
    from supermimic_study.envs.reference_interface import (
        make_initialization_env_cfg, read_initialization_context, read_model_state,
    )
    from supermimic_study.models import Policy, Critic, DynamicsModel, NetworkConfig
    from supermimic_study.models.critic_features import root_tracking_features

    torch.set_num_threads(4)
    if args.steps < 1 or args.steps > 5:
        raise ValueError("initialization check is bounded to 1–5 control steps")
    env = ManagerBasedRlEnv(cfg=make_initialization_env_cfg(args.library, seed=args.seed), device="cpu")
    try:
        obs, _ = env.reset()
        command = env.command_manager.get_term("motion")
        robot = env.scene["robot"]
        action_term = env.action_manager.get_term("joint_pos")
        torch.manual_seed(args.seed)
        config = NetworkConfig()
        models = {"policy": Policy(config).eval(), "world_model_dynamics": DynamicsModel(config).eval(), "critic": Critic(config).eval()}

        def digest(model) -> str:
            h = hashlib.sha256()
            for key, tensor in model.state_dict().items():
                h.update(key.encode())
                h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()

        before = {name: digest(model) for name, model in models.items()}
        if obs["policy"].shape != (2, 930) or obs["goal"].shape != (2, 680):
            raise AssertionError("unexpected policy observation or reference goal shape")
        torch.testing.assert_close(command.frame_ids, torch.zeros(2, dtype=torch.long))
        ref = command.library.get_frames(command.clip_ids, command.frame_ids)
        torch.testing.assert_close(obs["goal"].reshape(2, 10, 68)[:, 0, :29] + robot.data.default_joint_pos, ref["joint_pos"])
        torch.testing.assert_close(command.reference["body_pos_w"], ref["body_pos_w"]+env.scene.env_origins[:, None, :])
        initial_relative_q = (robot.data.joint_pos+robot.data.encoder_bias-robot.data.default_joint_pos)
        torch.testing.assert_close(obs["policy"][:, :290].reshape(2, 10, 29), initial_relative_q[:, None, :].expand(-1, 10, -1))
        report["checks"]["target_and_history_after_reset"] = "passed"
        context = read_initialization_context(env)
        report["nominal_context"] = context.tolist()
        report["data"] = {"clips": command.library.num_clips, "frames": command.library.total_frames,
                          "selected_clips": [command.library.clip_names[int(i)] for i in command.clip_ids]}
        report["input_shapes"] = {"observation_history": list(obs["policy"].shape), "reference_goal": list(obs["goal"].shape),
                                  "state": list(read_model_state(env).shape), "context": list(context.shape)}
        print("Reference command and real observation history connected", flush=True)
        for step in range(args.steps):
            state = read_model_state(env)
            with torch.no_grad():
                current_ref = command.reference
                root_index = command.library.body_names.index("pelvis")
                # command.reference positions include env origins, state does not.
                root_error = root_tracking_features(state,
                    current_ref["body_pos_w"][:,root_index]-env.scene.env_origins,
                    current_ref["body_quat_w"][:,root_index])
                latent = models["policy"].encode_goal(obs["goal"])
                action = models["policy"](obs["policy"], obs["goal"])
                predicted = models["world_model_dynamics"](state, action, context)
                value = models["critic"](state, obs["goal"], context,root_error)
            for name, tensor, shape in (("latent", latent, (2,32)), ("action", action, (2,29)),
                                        ("predicted_state", predicted, (2,71)), ("value", value, (2,1))):
                if tensor.shape != shape or not torch.isfinite(tensor).all():
                    raise AssertionError(f"invalid initialized {name}")
            if torch.any(latent.abs() > 1):
                raise AssertionError("motion latent exceeds tanh bounds")
            torch.testing.assert_close(predicted[:, 3:7].norm(dim=-1), torch.ones(2), atol=1e-6, rtol=1e-6)
            obs, _, terminated, truncated, _ = env.step(action)
            if torch.any(terminated | truncated):
                raise AssertionError("unexpected diagnostic episode end")
            torch.testing.assert_close(command.frame_ids, torch.full((2,), step+1, dtype=torch.long))
            torch.testing.assert_close(obs["policy"][:, -29:], action)
            expected_target = action_term.offset + action_term.scale*action - robot.data.encoder_bias[:, action_term.target_ids]
            torch.testing.assert_close(robot.data.joint_pos_target[:, action_term.target_ids], expected_target)
        report["checks"]["forward_and_action_execution"] = "passed"
        report["input_shapes"]["critic_root_error"] = list(root_error.shape)
        report["output_shapes"] = {"latent": list(latent.shape), "policy_action": list(action.shape),
                                   "world_model_next_state": list(predicted.shape), "critic_value": list(value.shape)}
        report["max_abs_untrained_action"] = float(action.abs().max())
        report["reference_frames_after_steps"] = command.frame_ids.tolist()
        saved_goal = obs["goal"].clone()
        saved_history = obs["policy"].clone()
        ids = torch.tensor([0], dtype=torch.long)
        reset_obs, _ = env.reset(env_ids=ids)
        if command.frame_ids.tolist() != [0, args.steps]:
            raise AssertionError("partial reset disturbed the other reference timeline")
        torch.testing.assert_close(reset_obs["goal"][1], saved_goal[1])
        torch.testing.assert_close(reset_obs["policy"][1], saved_history[1])
        torch.testing.assert_close(reset_obs["policy"][0, -29:], torch.zeros(29))
        report["checks"]["partial_reset_isolation"] = "passed"
        command.frame_ids[0] = command._last[0]-2
        command.compute(0.0, env_ids=ids)
        tail = command.command[0].reshape(10, 68)
        torch.testing.assert_close(tail[2:], tail[2:3].expand(8, -1))
        env.reset(env_ids=ids)
        report["checks"]["future_window_stays_inside_clip"] = "passed"
        after = {name: digest(model) for name, model in models.items()}
        if before != after or any(p.grad is not None for model in models.values() for p in model.parameters()):
            raise AssertionError("model parameters or gradients changed in forward-only initialization")
        report["checks"]["parameters_unchanged_no_optimizer"] = "passed"
        report["parameter_counts"] = {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()}
        report["parameter_sha256"] = after
        report["network_config"] = asdict(config)
        report["interface_schema"] = {
            "version": "supermimic-study.initialization-v2",
            "observation_terms": [["relative_joint_pos",29],["joint_vel",29],["imu_ang_vel",3],["projected_gravity",3],["previous_action",29]],
            "history": "10 frames, oldest to newest, term-major",
            "goal_terms": [["relative_joint_pos",29],["joint_vel",29],["projected_gravity",3],["root_ang_vel_body",3],["root_lin_vel_heading",3],["root_height",1]],
            "goal_offsets": list(range(10)), "goal_tail": "repeat final reference frame",
            "state": "qpos36 | qvel35; root position relative to scene origin; angular velocity in body frame",
            "context": "nominal shared foot friction | encoder bias29 | nominal torso COM offset3",
            "critic_root_error": "4D: robot-minus-reference root position in reference heading frame (m), wrapped yaw (rad); same-time reference; critic only",
            "world_model_coverage": "dynamics core only",
        }
        report["implementation_sha256"] = {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (ROOT/"src/supermimic_study/models/networks.py", ROOT/"src/supermimic_study/models/state.py",
                         ROOT/"src/supermimic_study/models/critic_features.py",
                         ROOT/"src/supermimic_study/envs/reference_interface.py")
        }
        report["versions"] = {name: metadata.version(name) for name in ("mjlab", "torch", "mujoco", "mujoco-warp", "warp-lang")}
        checkpoint = args.output.parent / "initial_models.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format_version": 1, "untrained": True, "seed": args.seed,
                    "config": asdict(config), "interface_schema": report["interface_schema"],
                    "implementation_sha256": report["implementation_sha256"],
                    "models": {name: model.state_dict() for name, model in models.items()},
                    "scope": "initial weights only; world model dynamics core only; no optimizer"}, checkpoint)
        restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if restored["untrained"] is not True:
            raise AssertionError("checkpoint lost its untrained marker")
        for name, model in models.items():
            for key, tensor in model.state_dict().items():
                torch.testing.assert_close(restored["models"][name][key], tensor, rtol=0, atol=0)
        report["checkpoint"] = str(checkpoint)
        report["checks"]["initial_checkpoint_roundtrip"] = "passed"
        print("All three model cores initialized and checked; no parameter update", flush=True)
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=ROOT/"data/processed/lafan1")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT/"output/initialization/check.json")
    args = parser.parse_args()
    report = {"status": "running", "scope": "reference inputs and untrained model initialization only",
              "timestamp_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed, "checks": {}}
    try:
        run(args, report)
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        traceback.print_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(f"Initialization: {report['status']}; report: {args.output}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
