"""Prepare G1 reference trajectories with mjlab conventions, entirely locally.

CSV column/joint conventions follow pinned mjlab; local float64 resampling
fixes its near-parallel quaternion interpolation. Offline forward kinematics
uses the same compiled G1 model through native MuJoCo on CPU;
the simulation/training backend is unchanged. Cross-backend checks live in
scripts/check_lafan1.py. No physical feasibility or learning claim is made.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.scene import Scene
from mjlab.scripts import csv_to_npz
from mjlab.utils.lab_api.math import quat_apply_inverse
from supermimic_study.envs.g1 import make_g1_diagnostic_cfg
from .motion_library import MOTION_FIELDS

PREPROCESS_VERSION = 2


def _normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms < 1e-12):
        raise ValueError("quaternions must be finite and non-zero")
    return quaternions / norms


def _quat_slerp_wxyz(q0: np.ndarray, q1: np.ndarray, fraction: np.ndarray) -> np.ndarray:
    """Normalized shortest-arc SLERP, including numerically close rotations."""
    q0 = _normalize_quaternions(np.asarray(q0, dtype=np.float64))
    q1 = _normalize_quaternions(np.asarray(q1, dtype=np.float64))
    fraction = np.asarray(fraction, dtype=np.float64)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    flip = dot < 0.0
    q1 = np.where(flip, -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    # The sine-weight form loses precision as theta -> 0. Normalized lerp is
    # its stable limit and still interpolates (unlike the pinned float32 code).
    close = dot > 1.0 - 1e-8
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    f = fraction[..., None]
    denom = np.where(close, 1.0, sin_theta)
    w0 = np.where(close, 1.0 - f, np.sin((1.0 - f) * theta) / denom)
    w1 = np.where(close, f, np.sin(f * theta) / denom)
    return _normalize_quaternions(w0 * q0 + w1 * q1)


def _quat_multiply_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack((aw*bw-ax*bx-ay*by-az*bz,
                     aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx,
                     aw*bz+ax*by-ay*bx+az*bw), axis=-1)


def _quat_conjugate_wxyz(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def _rotation_vector_wxyz(q: np.ndarray) -> np.ndarray:
    """Principal rotation vector for unit wxyz quaternions."""
    q = _normalize_quaternions(q)
    q = np.where(q[..., :1] < 0.0, -q, q)
    v = q[..., 1:]
    length = np.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(length, np.clip(q[..., :1], 0.0, 1.0))
    scale = np.divide(angle, length, out=np.full_like(angle, 2.0), where=length > 1e-12)
    return v * scale


def resample_motion_arrays(base_pos: np.ndarray, base_quat_wxyz: np.ndarray,
                           dof_pos: np.ndarray, input_fps: int = 30,
                           output_fps: int = 50):
    """Resample positions/joints and derive velocities on the output time grid.

    Returns times and six arrays: base position, unit wxyz rotation, DOF
    position, base linear velocity, base angular velocity in world coordinates,
    and DOF velocity. Calculation uses float64; callers choose storage dtype.
    """
    pos, quat, joints = (np.asarray(x, dtype=np.float64) for x in
                         (base_pos, base_quat_wxyz, dof_pos))
    if input_fps <= 0 or output_fps <= 0 or pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("invalid FPS or base position shape")
    if quat.shape != (len(pos), 4) or joints.ndim != 2 or len(joints) != len(pos) or len(pos) < 3:
        raise ValueError("expected matching [frames,4] quaternions and [frames,dofs], >=3 frames")
    if not (np.isfinite(pos).all() and np.isfinite(quat).all() and np.isfinite(joints).all()):
        raise ValueError("motion arrays must be finite")
    quat = _normalize_quaternions(quat)
    duration = (len(pos) - 1) / float(input_fps)
    # Retain MotionLoader's inclusive endpoint rule and tolerance exactly.
    n = int(np.floor(duration * output_fps + 1e-6)) + 1
    times = np.arange(n, dtype=np.float64) / float(output_fps)
    phase = times * input_fps
    i0 = np.minimum(np.floor(phase).astype(np.int64), len(pos) - 1)
    i1 = np.minimum(i0 + 1, len(pos) - 1)
    frac = np.clip(phase - i0, 0.0, 1.0)
    p = pos[i0] * (1.0 - frac[:, None]) + pos[i1] * frac[:, None]
    j = joints[i0] * (1.0 - frac[:, None]) + joints[i1] * frac[:, None]
    q = _quat_slerp_wxyz(quat[i0], quat[i1], frac)
    dt = 1.0 / float(output_fps)
    pvel = np.gradient(p, dt, axis=0)
    jvel = np.gradient(j, dt, axis=0)
    relative = _quat_multiply_wxyz(q[2:], _quat_conjugate_wxyz(q[:-2]))
    omega_mid = _rotation_vector_wxyz(relative) / (2.0 * dt)
    omega = np.concatenate((omega_mid[:1], omega_mid, omega_mid[-1:]), axis=0)
    return times, p, q, j, pvel, omega, jvel


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def official_csv_joint_names() -> tuple[str, ...]:
    """Read the input order from the pinned official converter, not simulator order."""
    tree = ast.parse(Path(csv_to_npz.__file__).read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "run_sim"]
    values = [k.value for call in calls for k in call.keywords if k.arg == "joint_names"]
    if len(values) != 1:
        raise RuntimeError("cannot identify pinned mjlab CSV joint convention")
    names = tuple(ast.literal_eval(values[0]))
    if len(names) != 29 or len(set(names)) != 29:
        raise ValueError("invalid G1 CSV joint convention")
    return names


@dataclass
class G1MotionConverter:
    model: mujoco.MjModel
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    csv_joint_names: tuple[str, ...]
    joint_qadr: np.ndarray
    joint_vadr: np.ndarray
    csv_to_joint: np.ndarray
    body_ids: np.ndarray
    free_qadr: int
    free_vadr: int

    @classmethod
    def from_mjlab(cls) -> "G1MotionConverter":
        scene = Scene(make_g1_diagnostic_cfg(num_envs=1).scene, device="cpu")
        model = scene.compile()
        robot = scene["robot"]
        csv_names = official_csv_joint_names()
        if set(csv_names) != set(robot.joint_names):
            raise ValueError("CSV and compiled G1 joint sets differ")
        joint_ids = np.array([model.joint(f"robot/{n}").id for n in robot.joint_names])
        free_ids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free_ids) != 1:
            raise ValueError("diagnostic model must contain one floating G1 root")
        return cls(
            model=model, joint_names=robot.joint_names, body_names=robot.body_names,
            csv_joint_names=csv_names,
            joint_qadr=model.jnt_qposadr[joint_ids].copy(),
            joint_vadr=model.jnt_dofadr[joint_ids].copy(),
            csv_to_joint=np.array([robot.joint_names.index(n) for n in csv_names]),
            body_ids=np.array([model.body(f"robot/{n}").id for n in robot.body_names]),
            free_qadr=int(model.jnt_qposadr[free_ids[0]]),
            free_vadr=int(model.jnt_dofadr[free_ids[0]]),
        )

    def convert(self, source: Path, destination: Path, *, input_frames: int | None = None) -> dict:
        raw = np.loadtxt(source, delimiter=",", max_rows=input_frames)
        if raw.ndim != 2 or raw.shape[1] != 36 or raw.shape[0] < 3:
            raise ValueError(f"expected at least three G1 CSV rows with 36 columns: {source}")
        if not np.isfinite(raw).all():
            raise ValueError(f"non-finite CSV values: {source}")
        if not np.allclose(np.linalg.norm(raw[:, 3:7], axis=-1), 1, atol=1e-4):
            raise ValueError(f"non-unit source root quaternion: {source}")
        # Keep the pinned converter's CSV column and joint conventions, but
        # perform resampling in float64 to avoid its near-parallel SLERP branch.
        raw_wxyz = raw[:, 3:7][:, [3, 0, 1, 2]]
        (times, pos, quat, q, lin_vel, ang_vel_world, v) = resample_motion_arrays(
            raw[:, :3], raw_wxyz, raw[:, 7:], 30, 50)
        n = len(times)
        result = {name: np.empty((n, *shape), dtype=np.float32) for name, shape in MOTION_FIELDS.items()}
        # MuJoCo's free-joint angular qvel is local to the root; the sampled
        # SO(3) derivative above is world-frame, as in MotionLoader.
        quat_t = torch.from_numpy(quat.astype(np.float32))
        ang_t = torch.from_numpy(ang_vel_world.astype(np.float32))
        body_angular_velocity = quat_apply_inverse(quat_t, ang_t).numpy()
        pos, quat, lin_vel, q, v = (x.astype(np.float32) for x in (pos, quat, lin_vel, q, v))
        data = mujoco.MjData(self.model)
        qa, va = self.free_qadr, self.free_vadr
        for t in range(n):
            data.qpos[qa:qa + 3], data.qpos[qa + 3:qa + 7] = pos[t], quat[t]
            data.qvel[va:va + 3], data.qvel[va + 3:va + 6] = lin_vel[t], body_angular_velocity[t]
            data.qpos[self.joint_qadr[self.csv_to_joint]] = q[t]
            data.qvel[self.joint_vadr[self.csv_to_joint]] = v[t]
            # Pure position/velocity kinematics; no contacts, forces or integration.
            mujoco.mj_kinematics(self.model, data)
            mujoco.mj_comPos(self.model, data)
            mujoco.mj_comVel(self.model, data)
            body_pos = data.xpos[self.body_ids]
            cvel = data.cvel[self.body_ids]
            result["joint_pos"][t] = data.qpos[self.joint_qadr]
            result["joint_vel"][t] = data.qvel[self.joint_vadr]
            result["body_pos_w"][t] = body_pos
            result["body_quat_w"][t] = data.xquat[self.body_ids]
            result["body_ang_vel_w"][t] = cvel[:, :3]
            result["body_lin_vel_w"][t] = cvel[:, 3:] - np.cross(
                cvel[:, :3], data.subtree_com[self.body_ids[0]] - body_pos
            )
        for name, array in result.items():
            if not np.isfinite(array).all():
                raise ValueError(f"non-finite converted {name}: {source}")
        if not np.allclose(np.linalg.norm(result["body_quat_w"], axis=-1), 1, atol=1e-5):
            raise ValueError(f"non-unit converted body quaternion: {source}")
        np.testing.assert_allclose(result["joint_pos"][0, self.csv_to_joint], raw[0, 7:], atol=1e-6)
        destination.parent.mkdir(parents=True, exist_ok=True)
        part = destination.with_suffix(".npz.part")
        with part.open("wb") as stream:
            np.savez(stream, fps=np.array([50]), input_fps=np.array([30]),
                     joint_names=np.asarray(self.joint_names), body_names=np.asarray(self.body_names), **result)
        part.replace(destination)
        return {
            "name": source.stem, "path": destination.name,
            "input_frames": int(raw.shape[0]), "frames": n,
            "duration_s": (n - 1) / 50,
            "source_sha256": sha256(source), "npz_sha256": sha256(destination),
            "bytes": destination.stat().st_size,
        }
