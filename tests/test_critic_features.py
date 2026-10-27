from dataclasses import replace
import math
import unittest

import torch

from supermimic_study.models.critic_features import root_tracking_features
from supermimic_study.models.networks import Critic, NetworkConfig
from supermimic_study.models.state import quat_from_rotation_vector, quat_multiply, quat_rotate
from supermimic_study.training.loop import frozen


def state():
    s = torch.zeros(2,71,dtype=torch.float64)
    s[:,3] = 1
    return s


class CriticFeatureTests(unittest.TestCase):
    def test_known_reference_heading_position_and_wrap(self):
        s = state()
        s[:,:3] = s.new_tensor([1.,2.,3.])
        refp = torch.zeros(2,3,dtype=s.dtype)
        refq = quat_from_rotation_vector(s.new_tensor([0.,0.,math.pi/2])).expand(2,-1)
        feature = root_tracking_features(s,refp,refq)
        torch.testing.assert_close(feature,s.new_tensor([2.,-1.,3.,-math.pi/2]).expand(2,-1))
        s[:,3:7] = quat_from_rotation_vector(s.new_tensor([0.,0.,-math.pi+.01]))
        refq = quat_from_rotation_vector(s.new_tensor([0.,0.,math.pi-.01])).expand(2,-1)
        torch.testing.assert_close(root_tracking_features(s,refp,refq)[:,-1],s.new_full((2,),.02))

    def test_common_global_translation_and_yaw_do_not_change_features(self):
        s = state()
        s[:,:3] = s.new_tensor([.3,-.1,.8])
        s[:,3:7] = quat_from_rotation_vector(s.new_tensor([0.,0.,.2]))
        p = s.new_tensor([.1,.4,.9]).expand(2,-1)
        q = quat_from_rotation_vector(s.new_tensor([0.,0.,1.])).expand(2,-1)
        before = root_tracking_features(s,p,q)
        rotation = quat_from_rotation_vector(s.new_tensor([0.,0.,.7])).expand(2,-1)
        shift = s.new_tensor([10.,-2.,3.])
        changed = s.clone()
        changed[:,:3] = quat_rotate(rotation,s[:,:3])+shift
        changed[:,3:7] = quat_multiply(rotation,s[:,3:7])
        after = root_tracking_features(changed,quat_rotate(rotation,p)+shift,quat_multiply(rotation,q))
        torch.testing.assert_close(before,after)
        torch.testing.assert_close(root_tracking_features(s,p,-q),before)

    def test_identity_and_vertical_forward_axis_have_finite_gradients(self):
        for angle in (0.,math.pi/2):
            s = state()
            s[:,3:7] = quat_from_rotation_vector(s.new_tensor([0.,angle,0.]))
            s.requires_grad_()
            feature = root_tracking_features(s,torch.zeros_like(s[:,:3]),s.detach()[:,3:7])
            grad = torch.autograd.grad(feature.sum(),s)[0]
            self.assertTrue(torch.isfinite(feature).all())
            self.assertTrue(torch.isfinite(grad).all())

    def test_root_features_pass_finite_difference_check(self):
        s = state()[:1].requires_grad_()
        p = torch.zeros(1,3,dtype=s.dtype)
        q = s.detach()[:,3:7].clone()
        self.assertTrue(torch.autograd.gradcheck(lambda x:root_tracking_features(x,p,q),(s,)))

    def test_same_goal_features_can_hide_target_translation_but_new_features_distinguish(self):
        from supermimic_study.envs.reference_interface import reference_goal_features
        s = state()
        ref = {"body_pos_w":s.new_zeros(2,1,3),"body_quat_w":s[:,None,3:7].clone(),
               "body_lin_vel_w":s.new_zeros(2,1,3),"body_ang_vel_w":s.new_zeros(2,1,3),
               "joint_pos":s.new_zeros(2,29),"joint_vel":s.new_zeros(2,29)}
        shifted = {k:v.clone() for k,v in ref.items()}
        shifted["body_pos_w"][...,0] += 1
        torch.testing.assert_close(reference_goal_features(ref,s.new_zeros(29),0),
                                   reference_goal_features(shifted,s.new_zeros(29),0),atol=0,rtol=0)
        a = root_tracking_features(s,ref["body_pos_w"][:,0],ref["body_quat_w"][:,0])
        b = root_tracking_features(s,shifted["body_pos_w"][:,0],shifted["body_quat_w"][:,0])
        self.assertFalse(torch.equal(a,b))
        torch.testing.assert_close(b[:,0],s.new_full((2,),-1.))

    def test_frozen_critic_can_use_only_new_features_and_backprop_to_state(self):
        cfg = NetworkConfig(critic_hidden=(1,))
        critic = Critic(cfg).double()
        with torch.no_grad():
            for parameter in critic.parameters():
                parameter.zero_()
            critic.value_network[0].weight[:,-4:] = 1
            critic.value_network[-1].weight.fill_(1)
        s = state().requires_grad_()
        p = s.new_zeros(2,3)
        q = s.detach()[:,3:7]
        goal,context = s.new_zeros(2,680),s.new_zeros(2,33)
        with frozen(critic):
            a = critic(s,goal,context,root_tracking_features(s,p,q))
            b = critic(s,goal,context,root_tracking_features(s,p+1,q))
            self.assertFalse(torch.equal(a,b))
            gradient = torch.autograd.grad(a.sum(),s)[0]
        self.assertGreater(float(gradient[:,:3].norm()),0)
        self.assertGreater(float(gradient[:,3:7].norm()),0)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue(all(p.grad is None for p in critic.parameters()))

    def test_root_input_is_required_and_old_baseline_is_explicit(self):
        cfg = NetworkConfig(critic_hidden=(8,4))
        s = state().float()
        goal,ctx = torch.zeros(2,680),torch.zeros(2,33)
        critic = Critic(cfg)
        with self.assertRaisesRegex(ValueError,"requires same-time"):
            critic(s,goal,ctx)
        with self.assertRaisesRegex(ValueError,"root_error.*last dimension 4"):
            critic(s,goal,ctx,torch.zeros(2,3))
        with self.assertRaisesRegex(ValueError,"batch dimensions"):
            critic(s,goal,ctx,torch.zeros(1,4))
        old = Critic(replace(cfg,critic_root_error=False))
        self.assertEqual(old.value_network[0].in_features,784)
        self.assertEqual(old(s,goal,ctx).shape,(2,1))
        with self.assertRaises(ValueError):
            root_tracking_features(s,torch.zeros(1,3),torch.zeros(2,4))


if __name__ == "__main__":
    unittest.main()
