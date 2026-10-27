"""Load the complete motion library and check mjlab interoperability; no training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]


def run(args: argparse.Namespace, report: dict) -> None:
    os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import numpy as np
    import torch
    import warp as wp
    wp.config.kernel_cache_dir = str(ROOT / ".cache/warp")
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.tracking.mdp.commands import MotionLoader
    from supermimic_study.data.motion_library import MotionLibrary, MOTION_FIELDS
    from supermimic_study.data.motion_conversion import sha256
    from supermimic_study.envs.g1 import make_g1_diagnostic_cfg

    manifest = json.loads((args.directory / "manifest.json").read_text())
    library = MotionLibrary.from_directory(args.directory, device="cpu")
    if library.num_clips != 40:
        raise AssertionError("full G1 LAFAN1 load must contain 40 clips")
    inventory = json.loads((args.raw_directory / "source-index.json").read_text())
    expected_sources = {Path(row["file"]).stem: row["sha256"] for row in inventory["files"]}
    if set(library.clip_names) != set(expected_sources) or inventory["revision"] != manifest["revision"]:
        raise AssertionError("loaded clips differ from the pinned source inventory")
    for row in manifest["clips"]:
        if row["source_sha256"] != expected_sources[row["name"]]:
            raise AssertionError("converted manifest source differs from the source inventory")
    report.update(
        num_clips=library.num_clips, total_frames=library.total_frames,
        fps=library.fps, duration_hours=manifest["duration_s"] / 3600,
        tensor_storage_bytes=library.storage_bytes,
        dataset_revision=manifest["revision"], manifest_sha256=sha256(args.directory / "manifest.json"),
        clip_names=list(library.clip_names), source_sha256_rechecked=0,
    )
    print(f"Complete library loaded: {library.num_clips} clips, {library.total_frames} frames", flush=True)
    env = ManagerBasedRlEnv(cfg=make_g1_diagnostic_cfg(num_envs=1), device="cpu")
    try:
        env.reset()
        robot = env.scene["robot"]
        if library.joint_names != robot.joint_names or library.body_names != robot.body_names:
            raise AssertionError("library joint/body order differs from runtime G1")
        max_errors = {key: 0.0 for key in MOTION_FIELDS}
        checked = 0
        for clip_id, row in enumerate(manifest["clips"]):
            source = args.raw_directory / "g1" / (row["name"] + ".csv")
            if sha256(source) != row["source_sha256"]:
                raise AssertionError(f"raw source hash mismatch: {row['name']}")
            report["source_sha256_rechecked"] += 1
            loaded = MotionLoader(str(args.directory / row["path"]), torch.arange(robot.num_bodies), "cpu")
            if loaded.time_step_total != row["frames"]:
                raise AssertionError("mjlab runtime frame count mismatch")
            frames = [0, row["frames"] // 2, row["frames"] - 1]
            selected = library.get_frames([clip_id] * 3, frames)
            for key in MOTION_FIELDS:
                torch.testing.assert_close(selected[key], getattr(loaded, key)[frames], rtol=0, atol=0)
            for local, frame in enumerate(frames):
                # Reconstruct each checked reference pose/velocity in mjlab Warp.
                root = torch.cat((loaded.body_pos_w[frame, 0], loaded.body_quat_w[frame, 0],
                                  loaded.body_lin_vel_w[frame, 0], loaded.body_ang_vel_w[frame, 0]))[None, :]
                robot.write_root_state_to_sim(root)
                robot.write_joint_state_to_sim(loaded.joint_pos[frame:frame + 1], loaded.joint_vel[frame:frame + 1])
                env.sim.forward()
                values = {
                    "joint_pos": robot.data.joint_pos,
                    "joint_vel": robot.data.joint_vel,
                    "body_pos_w": robot.data.body_link_pos_w,
                    "body_quat_w": robot.data.body_link_quat_w,
                    "body_lin_vel_w": robot.data.body_link_lin_vel_w,
                    "body_ang_vel_w": robot.data.body_link_ang_vel_w,
                }
                for key, value in values.items():
                    actual = value[0].cpu().numpy()
                    expected = selected[key][local].numpy()
                    if key == "body_quat_w":
                        signs = np.where(np.sum(actual * expected, axis=-1, keepdims=True) < 0, -1, 1)
                        actual = actual * signs
                    max_errors[key] = max(max_errors[key], float(np.abs(actual - expected).max()))
                    np.testing.assert_allclose(actual, expected, atol=5e-4, rtol=5e-4, err_msg=f"{row['name']} frame {frame} {key}")
                checked += 1
            try:
                library.get_frames([clip_id], [row["frames"]])
            except IndexError:
                pass
            else:
                raise AssertionError("clip-local frame boundary was not enforced")
            print(f"Verified {clip_id + 1}/40: {row['name']}", flush=True)
        report.update(
            mjlab_motion_loader_clips=40, warp_reconstruction_frames=checked,
            warp_reconstruction_max_abs_errors=max_errors, warp_reconstruction_atol=5e-4,
            warp_reconstruction_rtol=5e-4, all_clip_boundaries_checked=True,
            sample_batch_shapes={key: list(value.shape) for key, value in library.get_frames([0, 39], [0, library.frame_counts[39] - 1]).items()},
        )
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "data/processed/lafan1")
    parser.add_argument("--raw-directory", type=Path, default=ROOT / "data/raw/lafan1")
    parser.add_argument("--output", type=Path, default=ROOT / "output/lafan1/load-check.json")
    args = parser.parse_args()
    report = {"status": "running", "scope": "complete reference library loading and sparse kinematic reconstruction; no tracking or training", "python": sys.version,
              "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    try:
        run(args, report)
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        traceback.print_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Full LAFAN1 load: {report['status']}; report: {args.output}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
