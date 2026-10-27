"""Verify the simulation foundation, not SuperMimic training or tracking quality."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MJLAB_COMMIT = "8bb2bc9e187797689ec82c78c35019afd2fd0376"

# Deliberately no robot assets, viewer, motion data, or contact-rich task yet.
XML = """<mujoco model="platform_check">
  <option timestep="0.005"/>
  <worldbody>
    <geom name="ground" type="plane" size="2 2 0.1"/>
    <body name="box" pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>"""


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def run(args: argparse.Namespace, report: dict) -> None:
    (ROOT / ".cache").mkdir(exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    import mujoco
    import numpy as np
    import torch
    import warp as wp

    # Keep JIT artifacts inside this project, rather than the shared user cache.
    cache = ROOT / ".cache" / "warp"
    cache.mkdir(parents=True, exist_ok=True)
    wp.config.kernel_cache_dir = str(cache)

    from mjlab.sim import MujocoCfg, Simulation, SimulationCfg

    report["packages"] = {
        name: metadata.version(name)
        for name in ("mjlab", "mujoco", "mujoco-warp", "warp-lang", "torch", "numpy")
    }
    raw_source = metadata.distribution("mjlab").read_text("direct_url.json")
    report["mjlab_install_source"] = json.loads(raw_source) if raw_source else None
    report["expected_mjlab_commit"] = MJLAB_COMMIT
    if not raw_source or MJLAB_COMMIT not in raw_source:
        raise RuntimeError("mjlab install does not identify the expected pinned commit")
    if args.device != "cpu" and not args.device.startswith("cuda"):
        raise ValueError("Use cpu or cuda[:index]; MPS is not the mjlab Warp backend")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    report["capabilities"] = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "simulation_device": args.device,
    }
    x = torch.tensor([2.0, -3.0], device=args.device, requires_grad=True)
    x.square().sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([4.0, -6.0], device=args.device))
    report["checks"]["torch_autograd"] = {
        "status": "passed",
        "gradient": x.grad.detach().cpu().tolist(),
        "scope": "scalar derivative only, not the SuperMimic gradient path",
    }

    cfg = SimulationCfg(mujoco=MujocoCfg(timestep=0.005))
    model = mujoco.MjModel.from_xml_string(XML)
    sim = Simulation(num_envs=args.num_envs, cfg=cfg, model=model, device=args.device)
    start_qpos = sim.data.qpos.clone()
    start_qvel = sim.data.qvel.clone()
    started = time.perf_counter()
    for _ in range(args.steps):
        sim.step()
    sim.forward()  # Refresh derived positions after integration.
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
    elapsed = time.perf_counter() - started
    qpos = sim.data.qpos.clone()
    qvel = sim.data.qvel.clone()
    if not (torch.isfinite(qpos).all() and torch.isfinite(qvel).all()):
        raise AssertionError("non-finite simulation state")
    if torch.allclose(qpos, start_qpos):
        raise AssertionError("physics did not advance")

    # Independent native MuJoCo trajectory with the same applied options/state.
    native_model = mujoco.MjModel.from_xml_string(XML)
    cfg.mujoco.apply(native_model)
    native_data = mujoco.MjData(native_model)
    mujoco.mj_forward(native_model, native_data)
    for _ in range(args.steps):
        mujoco.mj_step(native_model, native_data)
    np.testing.assert_allclose(
        qpos.detach().cpu().numpy(),
        np.repeat(native_data.qpos[None, :], args.num_envs, axis=0),
        atol=1e-4,
        rtol=1e-4,
    )
    report["checks"]["mjlab_step"] = {
        "status": "passed",
        "num_envs": args.num_envs,
        "steps": args.steps,
        "timestep_s": 0.005,
        "qpos_shape": list(qpos.shape),
        "qvel_shape": list(qvel.shape),
        "initial_height": start_qpos[:, 2].cpu().tolist(),
        "final_height": qpos[:, 2].cpu().tolist(),
        "use_cuda_graph": sim.use_cuda_graph,
        "elapsed_s_including_first_step_jit": elapsed,
        "native_mujoco_qpos_agreement": "atol=1e-4, rtol=1e-4",
    }
    sim.reset()
    sim.forward()
    torch.testing.assert_close(sim.data.qpos.clone(), start_qpos)
    torch.testing.assert_close(sim.data.qvel.clone(), start_qvel)
    report["checks"]["reset_all_worlds"] = {"status": "passed"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=positive_int, default=1)
    parser.add_argument("--steps", type=positive_int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "output/platform-check.json")
    args = parser.parse_args()
    report = {
        "status": "running",
        "scope": "foundation only; no G1, motion tracking, world model, or training loop",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "checks": {},
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
    print(json.dumps(report, indent=2))
    print(f"Report: {args.output}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
