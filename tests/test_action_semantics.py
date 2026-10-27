"""Boundary checks distinguish clean policy controls from executed exploration."""
from types import SimpleNamespace
import unittest

import mujoco
import torch
from torch import nn
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

from supermimic_study.envs.tracking_task import TrackingTask
from supermimic_study.models.heads import observation_from_state, tracking_reward
from supermimic_study.models.kinematics import DifferentiableG1Kinematics
from supermimic_study.models.networks import NetworkConfig
from supermimic_study.models.state import StateCodec
from supermimic_study.models.world_model import WorldModel
from supermimic_study.training.history import initial_history
from supermimic_study.training.replay import ReplayBuffer


class ActionSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fk = DifferentiableG1Kinematics(mujoco.MjModel.from_xml_path(str(G1_XML)))

    def setUp(self):
        self.state = torch.zeros(2,71)
        self.state[:,2],self.state[:,3] = .8,1.
        self.context = torch.zeros(2,33)
        self.context[:,0] = .75

    def task(self):
        # Stub physics only: the actual task method owns all reward/history/data wiring.
        task = TrackingTask.__new__(TrackingTask)
        task.num_envs,task.default_q = 2,torch.zeros(29)
        task.limits = torch.tensor([[-10.,10.]]).expand(29,2)
        task.body_names = self.fk.body_names
        task.sensor_noise = False
        task.previous_policy_action = torch.full((2,29),.3)
        task.previous_executed_action = torch.full((2,29),.35)
        task.clean_frame = observation_from_state(self.state,task.previous_policy_action,self.context,task.default_q)
        task.history = initial_history(task.clean_frame)
        task._episodes = [[],[]]
        task.command = SimpleNamespace(clip_ids=torch.zeros(2,dtype=torch.long),
            frame_ids=torch.zeros(2,dtype=torch.long),finished=torch.zeros(2,dtype=torch.bool))
        task.state = lambda:self.state.clone()
        task.read_context = lambda:self.context.clone()
        task.native_bodies = lambda:self.fk(self.state)
        task.reference = lambda clips,frames:self.fk(self.state)
        self.sent = []
        def step(executed):
            self.sent.append(executed.clone())
            task.command.frame_ids += 1
            return None,None,torch.zeros(2,dtype=torch.bool),torch.zeros(2,dtype=torch.bool),{}
        task.env = SimpleNamespace(step=step,step_dt=.02)
        return task

    def test_noise_changes_executed_boundary_but_not_direct_smoothness_or_action_history(self):
        task = self.task()
        clean = torch.full((2,29),.3)
        executed = torch.full((2,29),.25)
        transition = task.step(clean,executed_action=executed)
        torch.testing.assert_close(self.sent[0],executed)
        # Same physical state/reference and constant clean output: all six tracking
        # terms are maximal, no direct noise contribution enters action-rate cost.
        torch.testing.assert_close(transition["reward"],torch.full((2,),.1))
        torch.testing.assert_close(task.history[:,-29:],clean)
        torch.testing.assert_close(task.previous_policy_action,clean)
        torch.testing.assert_close(task.previous_executed_action,executed)
        torch.testing.assert_close(transition["previous_executed_action"],torch.full((2,29),.35))
        torch.testing.assert_close(transition["previous_policy_action"],torch.full((2,29),.3))
        torch.testing.assert_close(transition["executed_action"],executed)
        torch.testing.assert_close(transition["policy_action"],clean)
        replay = ReplayBuffer(2,capacity=4)
        replay.add(transition)
        window = replay.sample_windows(3,1)
        torch.testing.assert_close(window["executed_action"],torch.full((3,1,29),.25))
        torch.testing.assert_close(window["policy_action"],torch.full((3,1,29),.3))
        starts = replay.sample_starts(2)
        torch.testing.assert_close(starts["previous_policy_action"],torch.full((2,29),.3))

    def test_clean_output_change_has_expected_direct_penalty(self):
        task = self.task()
        transition = task.step(torch.full((2,29),.4),executed_action=torch.full((2,29),.25))
        torch.testing.assert_close(transition["reward"],torch.full((2,),.1-.02*.1*29*.1**2))

    def test_alias_or_clean_action_continuity_mismatch_cannot_enter_training_window(self):
        from test_replay import batch
        replay = ReplayBuffer(1,capacity=4)
        incoming = batch(1)
        incoming["executed_action"].add_(1.)
        with self.assertRaisesRegex(ValueError,"alias"):
            replay.add(incoming)
        replay.add(batch(1,frame=0))
        second = batch(1,frame=1)
        second["previous_policy_action"].add_(.01)
        second["previous_action"].copy_(second["previous_policy_action"])
        replay.add(second)
        with self.assertRaisesRegex(ValueError,"not enough contiguous"):
            replay.sample_windows(1,2)

    def test_world_model_wrapper_requires_and_uses_explicit_action_roles(self):
        class SpyDynamics(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = NetworkConfig(critic_hidden=(8,4))
                self.codec = StateCodec()
            def forward(self,state,executed,context):
                self.seen = executed.clone()
                return torch.cat((state[:,:7],state[:,7:36]+.01*executed,state[:,36:]),-1)
        dynamics = SpyDynamics()
        wm = WorldModel(dynamics=dynamics,kinematics=self.fk,
            default_q=torch.zeros(29),joint_limits=torch.tensor([[-10.,10.]]).expand(29,2))
        clean,executed = torch.full((2,29),.3),torch.full((2,29),.25)
        reference = self.fk(self.state)
        with self.assertRaisesRegex(ValueError,"policy_action must be explicit"):
            wm(self.state,executed,self.context,previous_policy_action=clean)
        result = wm(self.state,executed,self.context,previous_policy_action=clean,
                    policy_action=clean,reference=reference)
        torch.testing.assert_close(dynamics.seen,executed)
        torch.testing.assert_close(result["next_state"][:,7:36],executed*.01)
        torch.testing.assert_close(result["observation"][:,-29:],clean)


if __name__ == "__main__":
    unittest.main()
