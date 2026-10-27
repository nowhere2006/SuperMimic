"""Compare every converted root trajectory to independent SciPy interpolation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]


def check(directory, raw_directory, baseline=None):
    from supermimic_study.data.motion_conversion import official_csv_joint_names, sha256
    manifest = json.loads((directory/"manifest.json").read_text())
    if manifest["preprocess_version"] != 2:
        raise ValueError("expected corrected preprocessing version 2")
    mapping = [manifest["joint_names"].index(n) for n in official_csv_joint_names()]
    pelvis = manifest["body_names"].index("pelvis")
    maxima = dict(rotation_rad=0.,position_m=0.,joint_rad=0.,linear_velocity_m_s=0.,
                  angular_velocity_rad_s=0.,joint_velocity_rad_s=0.)
    old_max_rotation = 0.
    total = 0
    for row in manifest["clips"]:
        raw = np.loadtxt(raw_directory/"g1"/(row["name"]+".csv"),delimiter=",")
        if sha256(raw_directory/"g1"/(row["name"]+".csv")) != row["source_sha256"]:
            raise AssertionError("raw source hash mismatch")
        t = np.arange(row["frames"],dtype=np.float64)/50
        source_t = np.arange(len(raw),dtype=np.float64)/30
        rotations = Slerp(source_t,Rotation.from_quat(raw[:,3:7]))(t)
        pos = np.stack([np.interp(t,source_t,raw[:,i]) for i in range(3)],-1)
        joints = np.empty((len(t),29),dtype=np.float64)
        joints[:,mapping] = np.stack([np.interp(t,source_t,raw[:,i]) for i in range(7,36)],-1)
        mid = (rotations[2:]*rotations[:-2].inv()).as_rotvec()/.04
        omega = np.concatenate((mid[:1],mid,mid[-1:]),axis=0)
        with np.load(directory/row["path"]) as out:
            q = Rotation.from_quat(out["body_quat_w"][:,pelvis][:,[1,2,3,0]])
            errors = {
                "rotation_rad":float(np.max((q.inv()*rotations).magnitude())),
                "position_m":float(np.max(np.abs(out["body_pos_w"][:,pelvis]-pos))),
                "joint_rad":float(np.max(np.abs(out["joint_pos"]-joints))),
                "linear_velocity_m_s":float(np.max(np.abs(out["body_lin_vel_w"][:,pelvis]-np.gradient(pos,.02,axis=0)))),
                "angular_velocity_rad_s":float(np.max(np.abs(out["body_ang_vel_w"][:,pelvis]-omega))),
                "joint_velocity_rad_s":float(np.max(np.abs(out["joint_vel"]-np.gradient(joints,.02,axis=0))))}
        for key,value in errors.items():
            maxima[key] = max(maxima[key],value)
        if errors["rotation_rad"] > 1e-5 or errors["position_m"] > 2e-5 or errors["joint_rad"] > 2e-6:
            raise AssertionError(f"pose interpolation mismatch: {row['name']}: {errors}")
        if any(errors[k] > 1e-4 for k in ("linear_velocity_m_s","angular_velocity_rad_s","joint_velocity_rad_s")):
            raise AssertionError(f"velocity mismatch: {row['name']}: {errors}")
        if baseline is not None:
            with np.load(baseline/row["path"]) as old:
                old_q = Rotation.from_quat(old["body_quat_w"][:,pelvis][:,[1,2,3,0]])
                old_max_rotation = max(old_max_rotation,float((old_q.inv()*rotations).magnitude().max()))
        total += len(t)
    return {"status":"passed","clips":len(manifest["clips"]),"frames":total,
        "preprocess_version":manifest["preprocess_version"],"manifest_sha256":sha256(directory/"manifest.json"),
        "independent_max_errors":maxima,"baseline_max_rotation_rad":old_max_rotation if baseline else None,
        "baseline_manifest_sha256":sha256(baseline/"manifest.json") if baseline else None,
        "oracle":"SciPy Slerp from raw source xyzw; np.interp and output-grid finite differences",
        "scope":"all-frame root/joint interpolation and velocities; body FK separately verified by check_lafan1"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,default=ROOT/"data/processed/lafan1")
    p.add_argument("--raw-directory",type=Path,default=ROOT/"data/raw/lafan1")
    p.add_argument("--baseline",type=Path)
    p.add_argument("--output",type=Path,default=ROOT/"output/lafan1/resampling-v2-check.json")
    args = p.parse_args()
    try:
        report = check(args.directory,args.raw_directory,args.baseline)
    except Exception:
        report = {"status":"failed","error":traceback.format_exc()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    return 0 if report["status"]=="passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
