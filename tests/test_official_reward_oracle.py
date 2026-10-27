"""Use pinned mjlab's actual alignment and reward functions as an external oracle."""
from types import SimpleNamespace
import unittest

import mujoco
import torch
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.tasks.tracking.mdp import rewards as native_rewards
from mjlab.envs.mdp.rewards import action_rate_l2

from supermimic_study.models.heads import TRACKING_BODY_NAMES, tracking_reward
from supermimic_study.models.kinematics import DifferentiableG1Kinematics
from supermimic_study.models.state import quat_from_rotation_vector


class OfficialRewardOracleTests(unittest.TestCase):
    def test_native_alignment_and_six_reward_terms_match_local_reward(self):
        model = mujoco.MjModel.from_xml_path(str(G1_XML))
        self.assertEqual(float(abs(model.jnt_pos).max()),0.)
        fk = DifferentiableG1Kinematics(model)
        gen = torch.Generator().manual_seed(111)
        state = torch.randn(4,71,generator=gen)*.2
        state[:,2] += .8
        state[:,3:7] = quat_from_rotation_vector(torch.randn(4,3,generator=gen)*.3)
        reference_state = state.clone()
        reference_state[:,:3] += torch.tensor([.1,-.2,.03])
        reference_state[:,3:7] = quat_from_rotation_vector(torch.randn(4,3,generator=gen)*.3)
        reference_state[:,7:36] += .05
        reference_state[:,36:] += .1
        current,reference = fk(state),fk(reference_state)
        indices = [fk.body_names.index(name) for name in TRACKING_BODY_NAMES]
        anchor = fk.body_names.index("torso_link")
        origins = torch.tensor([[0.,0.,0.],[3.,-2.,0.],[0.,4.,0.],[-1.,1.,0.]])
        robot_data = SimpleNamespace(**{
            "body_link_pos_w":current["body_pos_w"]+origins[:,None,:],
            "body_link_quat_w":current["body_quat_w"],
            "body_link_lin_vel_w":current["body_lin_vel_w"],
            "body_link_ang_vel_w":current["body_ang_vel_w"]})
        command = MotionCommand.__new__(MotionCommand)
        command.cfg = SimpleNamespace(body_names=TRACKING_BODY_NAMES)
        command._env = SimpleNamespace(scene=SimpleNamespace(env_origins=origins))
        command.robot = SimpleNamespace(data=robot_data)
        command.body_indexes = indices
        command.robot_anchor_body_index = anchor
        command.motion_anchor_body_index = TRACKING_BODY_NAMES.index("torso_link")
        command.motion = SimpleNamespace(**{key:value[:,indices] for key,value in reference.items()})
        command.time_steps = torch.arange(4)
        # Execute the official alignment method rather than copying its equations.
        command.update_relative_body_poses()
        clean = torch.randn(4,29,generator=gen)*.2
        previous = torch.randn(4,29,generator=gen)*.2
        env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name:command),
                              action_manager=SimpleNamespace(action=clean,prev_action=previous))
        functions = (native_rewards.motion_global_anchor_position_error_exp,
            native_rewards.motion_global_anchor_orientation_error_exp,
            native_rewards.motion_relative_body_position_error_exp,
            native_rewards.motion_relative_body_orientation_error_exp,
            native_rewards.motion_global_body_linear_velocity_error_exp,
            native_rewards.motion_global_body_angular_velocity_error_exp)
        expected = sum(weight*func(env,command_name="motion",std=std)
                       for weight,func,std in zip((.5,.5,1.,1.,1.,1.),functions,(.3,.4,.3,.4,1.,3.14)))
        expected = .02*(expected-.1*action_rate_l2(env))
        actual = tracking_reward(current,reference,clean,previous,state[:,7:36],
                                 torch.tensor([[-10.,10.]]).expand(29,2),body_names=fk.body_names)
        torch.testing.assert_close(actual,expected,atol=2e-7,rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
