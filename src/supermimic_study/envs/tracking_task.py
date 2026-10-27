"""A tracking task around mjlab physics, with post-forward shared task signals.

The underlying managers still own simulation/actions/reset/DR. This adapter owns
reference-state resets and computes reward/termination after forward and reference
clock advancement, avoiding mjlab manager signals' documented substep lag.
"""
from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .reference_interface import make_initialization_env_cfg, read_model_state, reference_goal_features
from supermimic_study.models.heads import observation_from_state, tracking_reward, hard_termination
from supermimic_study.training.history import initial_history, advance_history, add_sensor_noise
from supermimic_study.training.sampler import ReturnBinSampler


class TrackingTask:
    def __init__(self, library_directory, *, num_envs=2, device="cpu", seed=0,
                 clip_names=("walk1_subject1", "dance1_subject1"),
                 sensor_noise=True, domain_randomization=True, episode_seconds=2.,
                 solver_iterations=10, solver_ls_iterations=50):
        cfg = make_initialization_env_cfg(library_directory,
                  tuple(clip_names[i % len(clip_names)] for i in range(num_envs)), seed=seed)
        cfg.episode_length_s = episode_seconds
        cfg.sim.mujoco.iterations = solver_iterations
        cfg.sim.mujoco.ls_iterations = solver_ls_iterations
        if domain_randomization:
            cfg.events.update({
                "foot_friction": EventTermCfg(func=dr.geom_friction, mode="reset", params={
                    "asset_cfg": SceneEntityCfg("robot", geom_names=r"^(left|right)_foot[1-7]_collision$"),
                    "ranges": (.3,1.2), "operation": "abs", "shared_random": True}),
                "encoder_bias": EventTermCfg(func=dr.encoder_bias, mode="reset", params={
                    "asset_cfg": SceneEntityCfg("robot"), "bias_range": (-.01,.01)}),
                "torso_com": EventTermCfg(func=dr.body_com_offset, mode="reset", params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
                    "ranges": {0:(-.025,.025),1:(-.05,.05),2:(-.05,.05)}, "operation":"add"}),
            })
        self.env = ManagerBasedRlEnv(cfg=cfg, device=device)
        self.device, self.num_envs = device, num_envs
        self.robot = self.env.scene["robot"]
        self.command = self.env.command_manager.get_term("motion")
        self.library = self.command.library
        self.default_q = self.robot.data.default_joint_pos[0].clone()
        self.limits = self.robot.data.soft_joint_pos_limits[0].clone()
        self.body_names = tuple(self.robot.body_names)
        self.root = self.body_names.index("pelvis")
        self.sensor_noise, self.domain_randomization = sensor_noise, domain_randomization
        self.clip_counts = torch.tensor(self.library.frame_counts, device=device)
        selected = [self.library.clip_names.index(name) for name in clip_names]
        self.sampler = ReturnBinSampler(self.library.frame_counts, clip_ids=selected, device=device)
        self._torso_id = self.env.sim.mj_model.body("robot/torso_link").id
        self._nominal_com = torch.tensor(self.env.sim.mj_model.body_ipos[self._torso_id].copy(),
                                         device=device, dtype=torch.float32)
        _, foot_names = self.robot.find_geoms(r"^(left|right)_foot[1-7]_collision$")
        self._foot_ids = [self.env.sim.mj_model.geom(f"robot/{name}").id for name in foot_names]
        self.previous_policy_action = torch.zeros(num_envs, 29, device=device)
        self.previous_executed_action = torch.zeros(num_envs, 29, device=device)
        self.history = torch.zeros(num_envs, 930, device=device)
        self.clean_frame = torch.zeros(num_envs, 93, device=device)
        self._episodes = [[] for _ in range(num_envs)]
        self.episodes_completed = 0
        self.reset()

    def read_context(self):
        if self.domain_randomization:
            friction = self.env.sim.model.geom_friction[:, self._foot_ids, 0]
            if not torch.allclose(friction, friction[:, :1].expand_as(friction)):
                raise RuntimeError("foot friction must be shared within each episode")
            com = self.env.sim.model.body_ipos[:, self._torso_id] - self._nominal_com
            mu = friction[:, :1]
        else:
            mu = torch.full((self.num_envs,1), float(self.env.sim.mj_model.geom_friction[self._foot_ids[0],0]), device=self.device)
            com = torch.zeros(self.num_envs,3,device=self.device)
        return torch.cat((mu, self.robot.data.encoder_bias, com), -1).detach().clone()

    def state(self):
        return read_model_state(self.env)

    @property
    def previous_action(self):
        """Compatibility alias: last clean policy output, never executed noise."""
        return self.previous_policy_action

    def reference(self, clips, frames):
        return self.library.get_frames(clips.long(), frames.long())

    def goals(self, clips, frames):
        future = torch.minimum(frames[:,None] + torch.arange(10,device=self.device),
                               self.clip_counts[clips,None]-1)
        refs = self.reference(clips[:,None].expand_as(future).flatten(), future.flatten())
        return reference_goal_features(refs, self.default_q, self.root).reshape(len(clips),680)

    def native_bodies(self):
        data = self.robot.data
        return {"body_pos_w": data.body_link_pos_w.clone()-self.env.scene.env_origins[:,None,:],
                "body_quat_w": data.body_link_quat_w.clone(),
                "body_lin_vel_w": data.body_link_lin_vel_w.clone(),
                "body_ang_vel_w": data.body_link_ang_vel_w.clone()}

    @torch.no_grad()
    def reset(self, ids=None):
        ids = torch.arange(self.num_envs,device=self.device) if ids is None else ids
        if len(ids) == 0:
            return
        for i in ids.tolist():
            log = self._episodes[i]
            if log:
                c, f, r = zip(*log)
                self.sampler.update_episode(c, f, r)
                self.episodes_completed += 1
                self._episodes[i] = []
        self.env.reset(env_ids=ids)
        clips, frames, _ = self.sampler.sample(len(ids))
        self.command.clip_ids[ids] = clips
        self.command.frame_ids[ids] = frames
        self.command._last[ids] = self.clip_counts[clips]-1
        self.command.finished[ids] = False
        ref = self.reference(clips, frames)
        root_state = torch.cat((ref["body_pos_w"][:,self.root]+self.env.scene.env_origins[ids],
                                ref["body_quat_w"][:,self.root],
                                ref["body_lin_vel_w"][:,self.root],
                                ref["body_ang_vel_w"][:,self.root]), -1)
        self.robot.write_root_state_to_sim(root_state, env_ids=ids)
        self.robot.write_joint_state_to_sim(ref["joint_pos"], ref["joint_vel"], env_ids=ids)
        self.env.scene.write_data_to_sim()
        self.env.sim.forward()
        self.command.compute(0.,env_ids=ids)
        self.env.sim.sense()
        self.env.observation_manager.reset(ids)
        self.env.obs_buf = self.env.observation_manager.compute(update_history=True,env_ids=ids)
        self.previous_policy_action[ids] = 0
        self.previous_executed_action[ids] = 0
        clean = observation_from_state(self.state()[ids], self.previous_policy_action[ids],
                                        self.read_context()[ids], self.default_q)
        self.clean_frame[ids] = clean
        self.history[ids] = initial_history(add_sensor_noise(clean,self.sensor_noise))

    @torch.no_grad()
    def step(self, policy_action, *, executed_action):
        """Execute noisy controls; task regularization/history use clean policy outputs."""
        for name, value in (("policy_action",policy_action),("executed_action",executed_action)):
            if value.shape != (self.num_envs,29) or not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite [num_envs,29]")
        state, context = self.state(), self.read_context()
        previous = self.previous_policy_action.clone()
        previous_executed = self.previous_executed_action.clone()
        history = self.history.clone()
        clips, frames = self.command.clip_ids.clone(), self.command.frame_ids.clone()
        clean = self.clean_frame.clone()
        _, _, native_term, timeout, _ = self.env.step(executed_action)
        if native_term.any():
            raise RuntimeError("unexpected native termination: task adapter owns failure conditions")
        next_state = self.state()
        next_frames = self.command.frame_ids.clone()
        reference = self.reference(clips, next_frames)
        bodies = self.native_bodies()
        reward = tracking_reward(bodies,reference,policy_action,previous,next_state[:,7:36],
                                  self.limits,dt=self.env.step_dt,body_names=self.body_names)
        terminated = hard_termination(bodies,reference,body_names=self.body_names)
        truncated = timeout | self.command.finished
        next_clean = observation_from_state(next_state,policy_action,context,self.default_q)
        self.history = advance_history(history,add_sensor_noise(next_clean,self.sensor_noise))
        self.previous_policy_action = policy_action.detach().clone()
        self.previous_executed_action = executed_action.detach().clone()
        self.clean_frame = next_clean
        for i in range(self.num_envs):
            self._episodes[i].append((int(clips[i]),int(frames[i]),float(reward[i])))
        # Caller must insert this immutable pre-reset transition before calling reset.
        return {"state":state,"action":executed_action.detach().clone(),"next_state":next_state,
                "policy_action":policy_action.detach().clone(),"executed_action":executed_action.detach().clone(),
                "previous_policy_action":previous,"previous_executed_action":previous_executed,
                "previous_action":previous.clone(),"context":context,"observation":history,
                "clean_observation":clean,"reward":reward,"terminated":terminated,
                "truncated":truncated,"time_limit":timeout.clone(),"clip_end":self.command.finished.clone(),
                "clip_id":clips,"frame_id":frames,"next_frame_id":next_frames}

    def close(self):
        self.env.close()
