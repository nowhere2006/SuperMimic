"""Run bounded local training with durable records and model continuation."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _initial_checks(task, loop, report):
    import torch
    from mjlab.envs import mdp
    from supermimic_study.models.heads import observation_from_state, tracking_reward, hard_termination
    config = loop.config
    with torch.no_grad():
        state, context = task.state(),task.read_context()
        report["initial_context"] = context.cpu().tolist()
        if not ((context[:,0]>=.3)&(context[:,0]<=1.2)).all() or (context[:,1:30].abs()>.010001).any():
            raise AssertionError("actual DR context outside configured ranges")
        bound = context.new_tensor([.025,.05,.05])
        if (context[:,30:].abs()>bound+1e-6).any():
            raise AssertionError("actual COM offsets outside configured ranges")
        fk, native = loop.world_model.kinematics(state),task.native_bodies()
        errors = {}
        for key in native:
            if key == "body_quat_w":
                error = torch.minimum((fk[key]-native[key]).abs().amax(-1),
                                       (fk[key]+native[key]).abs().amax(-1)).max()
            else:
                error = (fk[key]-native[key]).abs().max()
            errors[key] = float(error)
            if error > 5e-5:
                raise AssertionError(f"native/Torch FK mismatch: {key}={error}")
        report["checks"]["native_fk_parity"] = errors
        native_frame = torch.cat((mdp.joint_pos_rel(task.env,biased=True),mdp.joint_vel_rel(task.env),
            mdp.builtin_sensor(task.env,sensor_name="robot/imu_ang_vel"),mdp.projected_gravity(task.env),
            mdp.last_action(task.env)),-1)
        analytic = observation_from_state(state,task.previous_action,context,task.default_q)
        torch.testing.assert_close(analytic,native_frame,atol=2e-5,rtol=1e-5)
        report["checks"]["native_observation_parity"] = "passed"
        reference = task.reference(task.command.clip_ids,task.command.frame_ids)
        torch.testing.assert_close(state[:,7:36],reference["joint_pos"],atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(state[:,:3],reference["body_pos_w"][:,task.root],atol=1e-5,rtol=1e-5)
        report["checks"]["reference_state_reset"] = "passed"
        zero = task.previous_action
        reward_native = tracking_reward(native,reference,zero,zero,state[:,7:36],task.limits,body_names=task.body_names)
        reward_model = loop.world_model.predict_reward(state,reference,zero,zero)
        torch.testing.assert_close(reward_native,reward_model,atol=1e-6,rtol=1e-5)
        self_reward = tracking_reward(native,native,zero,zero,state[:,7:36],task.limits,body_names=task.body_names)
        report["checks"]["native_model_reward_parity"] = {"max_error":float((reward_native-reward_model).abs().max()),
                                                           "self_reward":self_reward.cpu().tolist()}
        torch.testing.assert_close(hard_termination(native,reference,task.body_names),
                                   hard_termination(fk,reference,task.body_names))
        # A partial reset must preserve every unaffected env's episode context/history.
        context1, history1, state1 = context[1].clone(),task.history[1].clone(),state[1].clone()
        task.reset(torch.tensor([0],device=config.device))
        torch.testing.assert_close(task.read_context()[1],context1)
        torch.testing.assert_close(task.history[1],history1)
        torch.testing.assert_close(task.state()[1],state1)
        if torch.equal(task.read_context()[0],context[0]):
            raise AssertionError("episode DR context did not resample")
        report["checks"]["partial_reset_and_context_isolation"] = "passed"


def _run(args, output, stamp):
    from supermimic_study.training.artifacts import RunArtifacts
    report = {"status":"starting","started_at":stamp,"command":sys.argv,
              "scope":"bounded local training; no learning-effect claim","checks":{}}
    def persist():
        RunArtifacts._atomic_json(report,output/"report.json")
    persist()
    task = loop = artifacts = None
    exit_code = 0
    try:
        import torch
        import warp as wp
        from supermimic_study.envs.tracking_task import TrackingTask
        from supermimic_study.training.config import TrainingConfig
        from supermimic_study.training.loop import TrainingLoop
        from supermimic_study.models.networks import NetworkConfig
        wp.config.kernel_cache_dir = str(ROOT/".cache/warp")
        torch.set_num_threads(4)
        restored = torch.load(args.resume,map_location="cpu",weights_only=True) if args.resume else None
        if restored is not None and "world_model_loss" not in restored:
            raise ValueError("checkpoint predates explicit loss definitions; start a new run rather than silently changing loss")
        if restored is not None and restored.get("config",{}).get("action_semantics") != "clean-policy_executed-dynamics-v2":
            raise ValueError("checkpoint predates clean/executed action separation; start a new run")
        config = TrainingConfig(**restored["config"]) if restored is not None else TrainingConfig()
        overrides = {k:getattr(args,k) for k in ("seed","iterations","device","max_seconds") if getattr(args,k) is not None}
        if args.world_model_loss is not None:
            overrides["world_model_loss"] = args.world_model_loss
        if args.world_model_group_weights is not None:
            overrides["world_model_group_weights"] = tuple(args.world_model_group_weights)
        manifest = json.loads((args.library/"manifest.json").read_text())
        if args.all_clips:
            overrides["clip_names"] = tuple(c["name"] for c in manifest["clips"])
        elif args.clips:
            overrides["clip_names"] = tuple(args.clips)
        config = replace(config,**overrides)
        torch.manual_seed(config.seed)
        head = subprocess.run(["git","rev-parse","--verify","HEAD"],cwd=ROOT,text=True,capture_output=True)
        source_files = [p for folder in ("src","scripts","tests") for p in (ROOT/folder).rglob("*.py")]
        source_files += [ROOT/p for p in ("AGENTS.md","pyproject.toml","requirements/macos-arm64-py312.txt")]
        for source in source_files:
            target = output/"source"/source.relative_to(ROOT)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
        report.update({"status":"initializing","config":asdict(config),
            "recording":{"checkpoint_every":args.checkpoint_every,"keep_checkpoints":args.keep_checkpoints,
                         "probe_every":args.probe_every,"record_transitions":not args.no_record_transitions},
            "hardware":{"platform":platform.platform(),"machine":platform.machine(),"processor":platform.processor(),
                        "torch_threads":torch.get_num_threads()},
            "versions":{p:metadata.version(p) for p in ("torch","mjlab","mujoco","mujoco-warp","warp-lang","numpy")},
            "installed_packages":{d.metadata["Name"]:d.version for d in metadata.distributions() if d.metadata["Name"]},
            "python":sys.version,"code":{"branch":subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip(),
                "head":head.stdout.strip() if head.returncode == 0 else None,
                "git_status":subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True)},
            "source_sha256":{str(p.relative_to(ROOT)):sha256(p) for p in source_files},
            "data":{"manifest_sha256":sha256(args.library/"manifest.json"),"library":str(args.library.resolve()),
                    "dataset":manifest["dataset"],"revision":manifest["revision"],"split":"none; no validation/test claim"}})
        shutil.copyfile(args.library/"manifest.json",output/"data-manifest.json")
        RunArtifacts._atomic_json(asdict(config),output/"config.json")
        persist()
        artifacts = RunArtifacts(output,checkpoint_every=args.checkpoint_every,keep_checkpoints=args.keep_checkpoints,
                                 record_transitions=not args.no_record_transitions)
        task = TrackingTask(args.library,num_envs=config.num_envs,device=config.device,seed=config.seed,
                 clip_names=config.clip_names,sensor_noise=config.sensor_noise,
                 domain_randomization=config.domain_randomization,episode_seconds=config.episode_seconds,
                 solver_iterations=config.solver_iterations,solver_ls_iterations=config.solver_ls_iterations)
        loop = TrainingLoop(task,config,artifacts=artifacts,probe_every=args.probe_every,
                network_config=NetworkConfig(**restored["network_config"]) if restored is not None else None,
                metadata={"source_sha256":report["source_sha256"],"manifest_sha256":report["data"]["manifest_sha256"]})
        report["network_config"] = asdict(loop.policy.config)
        report["parameter_counts"] = {name:sum(p.numel() for p in module.parameters()) for name,module in
                            (("policy",loop.policy),("world_model",loop.world_model),("critic",loop.critic))}
        report["data"].update({"loaded_clips":task.library.num_clips,"loaded_frames":task.library.total_frames,
                               "joint_names":task.library.joint_names,"body_names":task.library.body_names})
        _initial_checks(task,loop,report)
        if restored is not None:
            loop.load_checkpoint(restored)
            report["resume"] = {"path":str(args.resume.resolve()),"sha256":sha256(args.resume),
                                "from_completed_iterations":loop.completed_iterations,
                                "mode":"weights/optimizers/sampler/schedule; fresh env/replay and warmup"}
            loop.metadata["resume_from"] = report["resume"]
        report["status"] = "training"
        persist()
        print(f"Run directory: {output}",flush=True)
        report["training"] = loop.run()
        checkpoint = loop.last_checkpoint
        payload = torch.load(checkpoint,map_location="cpu",weights_only=True)
        for name in ("policy","world_model","critic"):
            for key,tensor in getattr(loop,name).state_dict().items():
                torch.testing.assert_close(payload[name][key],tensor.detach().cpu(),rtol=0,atol=0)
        report["checks"]["checkpoint_weights_roundtrip"] = "passed"
        report["checkpoint"] = str(checkpoint)
        report["status"] = "completed"
        print("Training run completed; learning quality was not evaluated.",flush=True)
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc,KeyboardInterrupt) else "failed"
        exit_code = 130 if isinstance(exc,KeyboardInterrupt) else 1
        report["error"] = traceback.format_exc()
        print(report["error"],file=sys.stderr,flush=True)
        if loop is not None:
            report["completed_iterations"] = loop.completed_iterations
            report["completed_iteration_metrics"] = loop.metrics
            report["interrupted_phase"] = loop.phase
            try:
                # Diagnostic partial-phase snapshot never replaces latest complete.
                report["emergency_checkpoint"] = str(loop.save_checkpoint(tag="emergency"))
            except BaseException:
                report["emergency_save_error"] = traceback.format_exc()
    finally:
        if artifacts is not None:
            try:
                artifacts.flush_transitions()
            except BaseException:
                report["transition_flush_error"] = traceback.format_exc()
                report["status"],exit_code = "failed",1
        if task is not None:
            try:
                task.close()
            except BaseException:
                report["close_error"] = traceback.format_exc()
                report["status"],exit_code = "failed",1
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        persist()
        print(f"Report: {output/'report.json'}",flush=True)
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library",type=Path,default=ROOT/"data/processed/lafan1")
    parser.add_argument("--seed",type=int)
    parser.add_argument("--iterations",type=int,default=2,help="additional iterations in this invocation")
    parser.add_argument("--device")
    parser.add_argument("--max-seconds",type=float)
    parser.add_argument("--world-model-loss",choices=("supertrack_l1","raw_mse"),
                        help="default: initially balanced physical-group L1; raw_mse is the previous baseline")
    parser.add_argument("--world-model-group-weights",nargs=6,type=float,
                        metavar=("POS","ROT","VEL","OMEGA","JOINT","JOINT_VEL"),
                        help="fixed positive multipliers after initial calibration; saved in config")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--resume",type=Path,help="format-v2 complete checkpoint; environment/replay will be rebuilt")
    parser.add_argument("--checkpoint-every",type=int,default=10)
    parser.add_argument("--keep-checkpoints",type=int,default=3)
    parser.add_argument("--probe-every",type=int,default=10)
    parser.add_argument("--no-record-transitions",action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--clips",nargs="+")
    group.add_argument("--all-clips",action="store_true")
    args = parser.parse_args()
    if min(args.iterations,args.checkpoint_every,args.keep_checkpoints,args.probe_every) < 1:
        parser.error("iterations and recording intervals/retention must be positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = (args.output or ROOT/"output/training-runs"/stamp).resolve()
    output.mkdir(parents=True,exist_ok=False)
    os.environ.setdefault("XDG_CACHE_HOME",str(ROOT/".cache"))
    os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache/matplotlib"))
    from supermimic_study.training.console import capture_console
    with capture_console(output/"run.log"):
        return _run(args,output,stamp)


if __name__ == "__main__":
    raise SystemExit(main())
