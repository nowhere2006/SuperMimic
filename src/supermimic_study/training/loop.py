"""The collect -> fit model -> imagine actor/critic loop of Algorithm A.1."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math
import random
import time

import numpy as np

import torch
from torch.nn import functional as F

from supermimic_study.models.networks import Policy, Critic, NetworkConfig
from supermimic_study.models.kinematics import DifferentiableG1Kinematics
from supermimic_study.models.world_model import WorldModel
from supermimic_study.models.critic_features import root_tracking_features
from .history import advance_history, add_sensor_noise
from .replay import ReplayBuffer
from .returns import imagined_objective, gae_targets
from .model_loss import WorldModelStateLoss, GROUP_NAMES, ERROR_VERSION


def parameter_digest(module):
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def frozen(*modules):
    parameters = [p for module in modules for p in module.parameters()]
    old = [p.requires_grad for p in parameters]
    for p in parameters:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, required in zip(parameters,old):
            p.requires_grad_(required)


def finite_gradient_step(loss, module, optimizer, *, element_clip=None):
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    params = [p for p in module.parameters() if p.grad is not None]
    if not params or any(not torch.isfinite(p.grad).all() for p in params):
        raise FloatingPointError("missing or non-finite gradients")
    norm = torch.nn.utils.clip_grad_norm_(params,1.)
    if float(norm) <= 0 or not torch.isfinite(norm):
        raise FloatingPointError("zero or non-finite gradient norm")
    if element_clip is not None:
        torch.nn.utils.clip_grad_value_(params,element_clip)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in module.parameters()):
        raise FloatingPointError("non-finite updated parameter")
    return float(norm)


class TrainingLoop:
    def __init__(self, task, config, *, network_config=None, artifacts=None, metadata=None,
                 probe_every=10):
        self.task, self.config = task, config
        self.started = time.monotonic()
        net = network_config or NetworkConfig()
        self.policy = Policy(net).to(config.device)
        self.critic = Critic(net).to(config.device)
        fk = DifferentiableG1Kinematics(task.env.sim.mj_model).to(config.device)
        if tuple(fk.body_names) != task.body_names:
            raise ValueError("differentiable FK and reference body orders differ")
        self.world_model = WorldModel(kinematics=fk, default_q=task.default_q,
                                       joint_limits=task.limits, config=net).to(config.device)
        self.model_loss = WorldModelStateLoss(config).to(config.device)
        self.loss_calibration = None
        self.buffer = ReplayBuffer(config.num_envs,config.replay_capacity,config.device,config.seed)
        self.model_optimizer = torch.optim.Adam(self.world_model.parameters(),lr=config.learning_rate)
        self.actor_optimizer = torch.optim.Adam(self.policy.parameters(),lr=config.learning_rate,betas=(.9,.95))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=config.learning_rate)
        self.collected_steps = 0
        self.metrics = []
        self.artifacts, self.metadata = artifacts, metadata or {}
        self.probe_every = probe_every
        self.completed_iterations = 0
        self.phase = "initialized"
        self.last_saved_iteration = None
        self.last_checkpoint = None
        self.episode_ids = torch.zeros(config.num_envs,dtype=torch.long,device=config.device)
        self._probe_data = {}
        self.logged_reward_parity_checks = 0
        self.logged_reward_max_error = 0.

    def check_deadline(self):
        if time.monotonic()-self.started > self.config.max_seconds:
            raise TimeoutError("CPU smoke time budget exhausted")

    def event(self, name, **fields):
        if self.artifacts is not None:
            self.artifacts.event(name,completed_iterations=self.completed_iterations,
                                 phase=self.phase,elapsed_seconds=time.monotonic()-self.started,**fields)

    def save_checkpoint(self, tag=None):
        if self.artifacts is None:
            return None
        path = self.artifacts.checkpoint(self.checkpoint(),self.completed_iterations,tag=tag)
        if tag not in ("emergency","interrupted"):
            self.last_saved_iteration = self.completed_iterations
            self.last_checkpoint = path
        return path

    def _record_probe(self):
        return self.artifacts is not None and (self.completed_iterations == 0 or
                (self.completed_iterations+1) % self.probe_every == 0)

    @torch.no_grad()
    def collect(self, count):
        rewards, failures, truncations = [], 0, 0
        for _ in range(count):
            self.check_deadline()
            task = self.task
            goal = task.goals(task.command.clip_ids,task.command.frame_ids)
            policy_action = self.policy(task.history,goal)
            executed_action = policy_action+torch.randn_like(policy_action)*self.config.action_noise_std
            transition = task.step(policy_action,executed_action=executed_action)
            if self.collected_steps < 4 or self.collected_steps % 16 == 0:
                # Analytic heads have no training loss, so explicitly verify
                # them against post-transition simulator labels, not predictions.
                reference = task.reference(transition["clip_id"],transition["next_frame_id"])
                reconstructed = self.world_model.predict_reward(transition["next_state"],reference,
                                  transition["policy_action"],transition["previous_policy_action"])
                error = float((reconstructed-transition["reward"]).abs().max())
                torch.testing.assert_close(reconstructed,transition["reward"],atol=2e-6,rtol=1e-4)
                self.logged_reward_max_error = max(self.logged_reward_max_error,error)
                self.logged_reward_parity_checks += self.config.num_envs
            self.buffer.add(transition)
            if self.artifacts is not None:
                self.artifacts.transitions({**transition,
                    "env_id":torch.arange(self.config.num_envs,device=self.config.device),
                    "episode_id":self.episode_ids,
                    "control_step":torch.full_like(self.episode_ids,self.collected_steps),
                    "policy_iteration":torch.full_like(self.episode_ids,self.completed_iterations)})
            # Crucially, next_state is already retained in replay before reset.
            done = transition["terminated"] | transition["truncated"]
            failures += int(transition["terminated"].sum())
            truncations += int(transition["truncated"].sum())
            rewards.append(float(transition["reward"].mean()))
            for index in done.nonzero(as_tuple=False).flatten().tolist():
                episode = task._episodes[index]
                self.event("episode_end",env_id=index,episode_id=int(self.episode_ids[index]),
                           clip_id=int(transition["clip_id"][index]),
                           clip_name=task.library.clip_names[int(transition["clip_id"][index])],
                           start_frame=episode[0][1],end_frame=int(transition["next_frame_id"][index]),
                           length=len(episode),reward_sum=sum(x[2] for x in episode),
                           terminated=bool(transition["terminated"][index]),
                           truncated=bool(transition["truncated"][index]))
            task.reset(done.nonzero(as_tuple=False).flatten())
            self.episode_ids += done.long()
            self.collected_steps += 1
        if self.artifacts is not None:
            self.artifacts.flush_transitions()
        return {"mean_reward":sum(rewards)/len(rewards), "failures":failures,
                "truncations":truncations,"transitions":count*self.config.num_envs}

    def fit_world_model(self):
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        policy_before, critic_before = parameter_digest(self.policy), parameter_digest(self.critic)
        model_before = parameter_digest(self.world_model)
        logs = []
        for _ in range(self.config.world_model_batches):
            self.check_deadline()
            window = self.buffer.sample_windows(self.config.world_model_batch_size,self.config.world_model_window)
            if self._record_probe():
                # Save already selected examples; never draw diagnostic samples.
                self._probe_data.setdefault("model_fit_windows",[]).append({
                    k:window[k][:2].detach().cpu().clone() for k in
                    ("state","action","policy_action","executed_action","previous_policy_action","previous_executed_action",
                     "next_state","context","terminated","clip_id","frame_id","next_frame_id","episode_id","env_id")})
            state = window["state"][:,0]
            predicted_states, done_losses = [], []
            for j in range(self.config.world_model_window):
                state = self.world_model.predict_next(state,window["executed_action"][:,j],window["context"][:,j])
                reference = self.task.reference(window["clip_id"][:,j],window["next_frame_id"][:,j])
                logit = self.world_model.predict_termination_logit(state,reference)
                predicted_states.append(state)
                done_losses.append(F.binary_cross_entropy_with_logits(logit,window["terminated"][:,j].float()))
            was_calibrated = bool(self.model_loss.calibrated)
            state_loss, raw_groups, group_contributions = self.model_loss(
                torch.stack(predicted_states, dim=1), window["next_state"])
            if not was_calibrated and bool(self.model_loss.calibrated):
                self.loss_calibration = {"iteration":self.completed_iterations,
                    "collected_control_steps":self.collected_steps,
                    "batch_size":self.config.world_model_batch_size,"window":self.config.world_model_window,
                    "window_starts":{"env_id":window["env_id"].detach().cpu().tolist(),
                        **{key:window[key][:,0].detach().cpu().tolist()
                           for key in ("clip_id","frame_id","episode_id")}}}
                self.event("world_model_loss_calibrated",calibration=self.loss_calibration,
                           loss_definition=self.model_loss.summary())
            done_loss = torch.stack(done_losses).sum()
            loss = state_loss+self.config.world_model_termination_weight*done_loss
            grad = finite_gradient_step(loss,self.world_model,self.model_optimizer)
            record = {"loss":float(loss.detach()),"state_loss":float(state_loss.detach()),
                      "termination_bce":float(done_loss.detach()),"gradient_norm":grad}
            if raw_groups is not None:
                record["group_l1_mean"] = dict(zip(GROUP_NAMES,raw_groups.detach().cpu().tolist()))
                record["group_contribution_mean"] = dict(zip(GROUP_NAMES,group_contributions.detach().cpu().tolist()))
            logs.append(record)
        if policy_before != parameter_digest(self.policy) or critic_before != parameter_digest(self.critic):
            raise AssertionError("model fitting changed actor or critic")
        if model_before == parameter_digest(self.world_model):
            raise AssertionError("world model did not update")
        return {"batches":logs,"loss_definition":self.model_loss.summary(),
                "world_model_updated":True,"policy_critic_unchanged":True}

    def _imagination_starts(self):
        # Exclude externally truncated clip tails; do not imagine across clips.
        for _ in range(100):
            start = self.buffer.sample_starts(self.config.imagination_batch_size*4)
            valid = (start["frame_id"]+self.config.imagination_horizon <
                     self.task.clip_counts[start["clip_id"]]).nonzero(as_tuple=False).flatten()
            if len(valid) >= self.config.imagination_batch_size:
                ids = valid[:self.config.imagination_batch_size]
                return {key:value[ids] for key,value in start.items()}
        raise ValueError("no imagination starts with a complete within-clip horizon")

    def imagine_and_improve(self):
        self.model_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        model_before, critic_before = parameter_digest(self.world_model),parameter_digest(self.critic)
        actor_before = parameter_digest(self.policy)
        start = self._imagination_starts()
        state, history = start["state"],start["observation"]
        previous, context = start["previous_policy_action"],start["context"]
        clips, frames = start["clip_id"],start["frame_id"]
        states, goals, root_errors, rewards, dones, values = [], [], [], [], [], []
        jacobian_norm = None
        trace = []
        reference_at_state = self.task.reference(clips,frames)
        with frozen(self.world_model,self.critic):
            for t in range(self.config.imagination_horizon):
                self.check_deadline()
                goal = self.task.goals(clips,frames+t)
                states.append(state.detach())
                goals.append(goal.detach())
                root_error = root_tracking_features(state,reference_at_state["body_pos_w"][:,self.task.root],
                                                   reference_at_state["body_quat_w"][:,self.task.root])
                root_errors.append(root_error.detach())
                values.append(self.critic(state,goal,context,root_error).squeeze(-1))
                action = self.policy(history,goal)
                predicted = self.world_model.predict_next(state,action,context)
                if t == 0:
                    derivative = torch.autograd.grad(predicted.square().sum(),action,retain_graph=True)[0]
                    jacobian_norm = float(derivative.detach().norm())
                    if not torch.isfinite(derivative).all() or jacobian_norm <= 0:
                        raise AssertionError("fitted dynamics has no finite action-to-state gradient")
                reference = self.task.reference(clips,frames+t+1)
                rewards.append(self.world_model.predict_reward(predicted,reference,action,previous))
                dones.append(self.world_model.termination_probability(predicted,reference))
                frame = self.world_model.predict_observation(predicted,action,context)
                noisy_frame = add_sensor_noise(frame,self.config.sensor_noise)
                if self._record_probe():
                    trace.append({k:v[:2].detach().cpu().clone() for k,v in {
                        "state":state,"action":action,"next_state":predicted,
                        "reward":rewards[-1],"termination_probability":dones[-1],
                        "critic_root_error":root_error,"reference_frame":frames+t,
                        "critic_reference_position":reference_at_state["body_pos_w"][:,self.task.root],
                        "critic_reference_quaternion":reference_at_state["body_quat_w"][:,self.task.root],
                        "value":values[-1],"observation_noise":noisy_frame-frame}.items()})
                history = advance_history(history,noisy_frame)
                state, previous = predicted, action
                reference_at_state = reference
            final_goal = self.task.goals(clips,frames+self.config.imagination_horizon)
            final_root_error = root_tracking_features(state,reference_at_state["body_pos_w"][:,self.task.root],
                                                     reference_at_state["body_quat_w"][:,self.task.root])
            values.append(self.critic(state,final_goal,context,final_root_error).squeeze(-1))
            feature_gradient = torch.autograd.grad(values[-1].sum(),final_root_error,retain_graph=True,
                                                  allow_unused=True)[0]
            feature_gradient_norm = float(feature_gradient.detach().norm()) if feature_gradient is not None else 0.
            if self.critic.config.critic_root_error and (feature_gradient is None or
                    not torch.isfinite(feature_gradient).all() or feature_gradient_norm <= 0):
                raise AssertionError("critic lost its relative-root input gradient")
            bootstrap_gradient = torch.autograd.grad(values[-1].sum(),state,retain_graph=True)[0]
            bootstrap_gradient_norm = float(bootstrap_gradient.detach().norm())
            if not torch.isfinite(bootstrap_gradient).all() or bootstrap_gradient_norm <= 0:
                raise AssertionError("frozen critic lost its bootstrap input gradient")
            reward_tensor, done_tensor, value_tensor = torch.stack(rewards),torch.stack(dones),torch.stack(values)
            objective = imagined_objective(reward_tensor,done_tensor,value_tensor[-1],self.config.gamma)
            # Targets are computed once before either optimizer update, then fixed.
            targets = gae_targets(reward_tensor,done_tensor,value_tensor,self.config.gamma,self.config.gae_lambda)
            if self._record_probe():
                self._probe_data["imagination"] = {
                    "weight_stage":"after_WM_fit_before_actor_update; end-iteration checkpoint has later weights",
                    "start":{k:v[:2].detach().cpu().clone() for k,v in start.items()},
                    "trace":trace,"gae_targets":targets[:,:2].detach().cpu().clone(),
                    "bootstrap_root_error":final_root_error[:2].detach().cpu().clone(),
                    "bootstrap_reference_position":reference_at_state["body_pos_w"][:2,self.task.root].detach().cpu().clone(),
                    "bootstrap_reference_quaternion":reference_at_state["body_quat_w"][:2,self.task.root].detach().cpu().clone(),
                    "bootstrap_reference_frame":(frames+self.config.imagination_horizon)[:2].detach().cpu().clone(),
                    "bootstrap":value_tensor[-1,:2].detach().cpu().clone()}
            actor_grad = finite_gradient_step(-objective,self.policy,self.actor_optimizer,element_clip=.1)
            encoder_grad = float(torch.stack([p.grad.square().sum() for p in self.policy.motion_encoder.parameters()
                                             if p.grad is not None]).sum().sqrt())
            decoder_grad = float(torch.stack([p.grad.square().sum() for p in self.policy.decoder.parameters()
                                             if p.grad is not None]).sum().sqrt())
            if encoder_grad <= 0 or decoder_grad <= 0:
                raise AssertionError("motion encoder or shared decoder has no gradient")
        if model_before != parameter_digest(self.world_model) or critic_before != parameter_digest(self.critic):
            raise AssertionError("actor update changed frozen world model or critic")
        if any(p.grad is not None for module in (self.world_model,self.critic) for p in module.parameters()):
            raise AssertionError("actor backward accumulated gradients on frozen parameters")
        if actor_before == parameter_digest(self.policy):
            raise AssertionError("policy did not update")
        # The actor graph is no longer needed. Critic inputs and targets are fixed.
        training_state = torch.stack(states).flatten(0,1)
        training_goal = torch.stack(goals).flatten(0,1)
        training_root_error = torch.stack(root_errors).flatten(0,1)
        training_context = context.unsqueeze(0).expand(self.config.imagination_horizon,-1,-1).flatten(0,1)
        target = targets.flatten()
        target_copy = target.clone()
        actor_after = parameter_digest(self.policy)
        critic_losses, critic_gradients = [], []
        for _ in range(self.config.critic_epochs):
            order = torch.randperm(len(target),device=target.device)
            for ids in order.tensor_split(self.config.critic_minibatches):
                if not len(ids):
                    continue
                self.check_deadline()
                prediction = self.critic(training_state[ids],training_goal[ids],training_context[ids],
                                         training_root_error[ids]).squeeze(-1)
                loss = (prediction-target[ids]).square().mean()
                critic_gradients.append(finite_gradient_step(loss,self.critic,self.critic_optimizer))
                critic_losses.append(float(loss.detach()))
        if not torch.equal(target,target_copy):
            raise AssertionError("GAE targets changed during critic fitting")
        if critic_before == parameter_digest(self.critic):
            raise AssertionError("critic did not update")
        if actor_after != parameter_digest(self.policy) or model_before != parameter_digest(self.world_model):
            raise AssertionError("critic update changed actor or world model")
        return {"actor_objective":float(objective.detach()),"actor_gradient_norm":actor_grad,
                "action_to_predicted_state_gradient_norm":jacobian_norm,
                "bootstrap_state_gradient_norm":bootstrap_gradient_norm,
                "bootstrap_root_error_gradient_norm":feature_gradient_norm,
                "motion_encoder_gradient_norm":encoder_grad,"decoder_gradient_norm":decoder_grad,
                "policy_updated":True,"frozen_weights_unchanged":True,
                "critic_updated":True,"gae_targets_fixed":True,
                "critic_loss_first":critic_losses[0],"critic_loss_last":critic_losses[-1],
                "critic_gradient_norm_max":max(critic_gradients),
                "mean_predicted_termination":float(done_tensor.detach().mean())}

    def run(self):
        start_control_steps = self.collected_steps
        start_iterations = self.completed_iterations
        if self.artifacts is not None:
            self.save_checkpoint(tag="initial")
        self.phase = "warmup"
        self.event("phase_start")
        warmup = self.collect(self.config.warmup_steps)
        self.event("phase_end",collection=warmup)
        for _ in range(self.config.iterations):
            iteration = self.completed_iterations
            self._probe_data = {}
            progress = min(iteration/self.config.lr_decay_iterations,1.)
            ratio = self.config.minimum_lr_ratio+(1-self.config.minimum_lr_ratio)*.5*(1+math.cos(math.pi*progress))
            rate = self.config.learning_rate*ratio
            for optimizer in (self.model_optimizer,self.actor_optimizer,self.critic_optimizer):
                for group in optimizer.param_groups:
                    group["lr"] = rate
            self.phase = "collect"
            self.event("phase_start")
            collection = self.collect(self.config.collect_steps)
            self.event("phase_end",collection=collection)
            self.phase = "fit_world_model"
            self.event("phase_start")
            model = self.fit_world_model()
            self.event("phase_end",model=model)
            self.phase = "imagine_actor_critic"
            self.event("phase_start")
            control = self.imagine_and_improve()
            self.event("phase_end",control=control)
            metrics = {"iteration":iteration,"learning_rate":rate,"collection":collection,"model":model,"control":control}
            self.metrics.append(metrics)
            self.completed_iterations += 1
            self.phase = "between_iterations"
            if self.artifacts is not None:
                self.artifacts.metrics(metrics)
                if self._probe_data:
                    self.artifacts.probe(iteration,{"iteration_index":iteration,**self._probe_data})
                if self.completed_iterations % self.artifacts.checkpoint_every == 0:
                    self.save_checkpoint()
            print(f"iteration={iteration} WM={model['batches'][-1]['loss']:.4f} "
                  f"actor_grad={control['actor_gradient_norm']:.4g} "
                  f"critic={control['critic_loss_last']:.4f}",flush=True)
        if self.artifacts is not None and self.last_saved_iteration != self.completed_iterations:
            self.save_checkpoint(tag="final")
        return {"warmup":warmup,"iterations":self.metrics,"collected_control_steps":self.collected_steps,
                "collected_transitions":self.collected_steps*self.config.num_envs,
                "collected_transitions_this_run":(self.collected_steps-start_control_steps)*self.config.num_envs,
                "starting_completed_iterations":start_iterations,
                "completed_iterations":self.completed_iterations,
                "episodes_completed":self.task.episodes_completed,
                "logged_reward_parity_checks":self.logged_reward_parity_checks,
                "logged_reward_max_error":self.logged_reward_max_error,
                "world_model_loss":{**self.model_loss.summary(),"calibration":self.loss_calibration},
                "elapsed_seconds":time.monotonic()-self.started}

    def checkpoint(self):
        from dataclasses import asdict
        numpy_state = np.random.get_state()
        return {"format_version":2,"scope":"training continuation with fresh env/replay, not exact simulator resume",
                "phase":self.phase,"completed_iterations":self.completed_iterations,
                "collected_control_steps":self.collected_steps,
                "config":asdict(self.config),"network_config":asdict(self.policy.config),
                "metadata":self.metadata,
                "policy":self.policy.state_dict(),"world_model":self.world_model.state_dict(),
                "critic":self.critic.state_dict(),"sampler":self.task.sampler.state_dict(),
                "world_model_loss":{"error_version":ERROR_VERSION,"state_dict":self.model_loss.state_dict(),
                                    "calibration":self.loss_calibration},
                "optimizers":{"world_model":self.model_optimizer.state_dict(),
                              "policy":self.actor_optimizer.state_dict(),"critic":self.critic_optimizer.state_dict()},
                "torch_rng_state":torch.get_rng_state(),
                "cuda_rng_states":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "python_rng_state":random.getstate(),
                "numpy_rng_state":(numpy_state[0],numpy_state[1].tolist(),*numpy_state[2:])}

    def load_checkpoint(self, payload):
        """Continue weights/optimizers/sampler/schedule; reset and refill physics replay."""
        from dataclasses import asdict
        if payload.get("format_version") != 2:
            raise ValueError("continuation requires a format_version=2 training checkpoint")
        if payload.get("artifact_tag") in ("emergency","interrupted") or payload.get("phase") not in ("initialized","between_iterations"):
            raise ValueError("partial-phase snapshots are diagnostic only; resume the last complete checkpoint")
        expected = asdict(self.config)
        if set(payload["config"]) != set(expected):
            raise ValueError("checkpoint training configuration fields differ")
        for key,value in payload["config"].items():
            if key not in ("iterations","max_seconds","device") and expected.get(key) != value:
                raise ValueError(f"checkpoint training configuration differs: {key}")
        if payload["network_config"] != asdict(self.policy.config):
            raise ValueError("checkpoint network configuration differs")
        old_manifest = payload.get("metadata",{}).get("manifest_sha256")
        if not old_manifest or old_manifest != self.metadata.get("manifest_sha256"):
            raise ValueError("checkpoint reference data manifest differs")
        loss_state = payload.get("world_model_loss")
        if not loss_state or loss_state.get("error_version") != ERROR_VERSION:
            raise ValueError("checkpoint lacks matching world-model loss definition/calibration")
        self.model_loss.validate_saved_state(loss_state["state_dict"],int(payload["completed_iterations"]))
        if bool(loss_state["state_dict"]["calibrated"]) and not loss_state.get("calibration"):
            raise ValueError("checkpoint lacks world-model loss calibration provenance")
        self.model_loss.load_state_dict(loss_state["state_dict"])
        self.loss_calibration = loss_state["calibration"]
        for name in ("policy","world_model","critic"):
            getattr(self,name).load_state_dict(payload[name])
        for key,optimizer in (("world_model",self.model_optimizer),("policy",self.actor_optimizer),("critic",self.critic_optimizer)):
            optimizer.load_state_dict(payload["optimizers"][key])
        self.task.sampler.load_state_dict(payload["sampler"])
        self.completed_iterations = int(payload["completed_iterations"])
        self.collected_steps = int(payload["collected_control_steps"])
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        if payload.get("cuda_rng_states") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["cuda_rng_states"])
        random.setstate(payload["python_rng_state"])
        numpy_state = payload["numpy_rng_state"]
        np.random.set_state((numpy_state[0],np.asarray(numpy_state[1],dtype=np.uint32),*numpy_state[2:]))
        self.task.reset()
        self.phase = "initialized"
        self.event("checkpoint_loaded",resume_mode="fresh env/replay; warmup will run")
