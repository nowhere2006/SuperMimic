from dataclasses import replace
import io
import math
import unittest

import torch

from supermimic_study.models.state import quat_from_rotation_vector, quat_multiply
from supermimic_study.training.config import TrainingConfig
from supermimic_study.training.model_loss import WorldModelStateLoss, physical_errors


def states(batch=2, window=3, dtype=torch.float64):
    target = torch.zeros(batch, window, 71, dtype=dtype)
    target[..., 3] = 1
    prediction = target.clone()
    prediction[..., :3] += .01
    prediction[..., 3:7] = quat_from_rotation_vector(target.new_tensor([.02, .03, .04]))
    prediction[..., 36:39] += .1
    prediction[..., 39:42] += .2
    prediction[..., 7:36] += .3
    prediction[..., 42:71] += .4
    return prediction, target


class ModelLossTests(unittest.TestCase):
    def test_physical_group_order_units_and_known_rotation(self):
        prediction, target = states()
        errors = physical_errors(prediction, target)
        for error, width, expected in zip(errors, (3,3,3,3,29,29),
                                          (.01,None,.1,.2,.3,.4)):
            self.assertEqual(error.shape, (2,3,width))
            if expected is not None:
                torch.testing.assert_close(error, torch.full_like(error, expected))
        torch.testing.assert_close(errors[1], prediction.new_tensor([.02,.03,.04]).expand(2,3,3))

    def test_rotation_antipodes_and_shortest_branch(self):
        prediction, target = states()
        baseline = physical_errors(prediction, target)[1]
        for flip_prediction, flip_target in ((True,False),(False,True),(True,True)):
            p,t = prediction.clone(),target.clone()
            if flip_prediction:
                p[...,3:7] *= -1
            if flip_target:
                t[...,3:7] *= -1
            torch.testing.assert_close(physical_errors(p,t)[1],baseline)
        prediction[...,3:7] = quat_from_rotation_vector(prediction.new_tensor([0.,0.,2*math.pi-.1]))
        torch.testing.assert_close(physical_errors(prediction,target)[1],
                                  prediction.new_tensor([0.,0.,-.1]).expand(2,3,3))

    def test_relative_rotation_is_in_true_body_coordinates(self):
        prediction, target = states()
        target[...,3:7] = quat_from_rotation_vector(target.new_tensor([0.,0.,1.2]))
        delta = quat_from_rotation_vector(target.new_tensor([.2,0.,0.]))
        prediction[...,3:7] = quat_multiply(target[...,3:7],delta)
        torch.testing.assert_close(physical_errors(prediction,target)[1],
                                  prediction.new_tensor([.2,0.,0.]).expand(2,3,3))

    def test_identity_small_angle_and_near_pi_have_finite_derivatives(self):
        for angle in (0.,1e-9,math.pi-1e-6):
            p,t = states(dtype=torch.float32)
            p[...,3:7] = quat_from_rotation_vector(p.new_tensor([angle,0.,0.]))
            p.requires_grad_()
            error = physical_errors(p,t)[1]
            gradient = torch.autograd.grad(error.sum(),p)[0]
            self.assertTrue(torch.isfinite(error).all())
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient[...,3:7].abs().sum()),0.)
        # The exact pi cut is not smooth, but antipodes must have equal errors.
        p,t = states()
        p[...,3:7] = p.new_tensor([0.,-1.,0.,0.])
        error = physical_errors(p,t)[1]
        p[...,3:7] *= -1
        torch.testing.assert_close(physical_errors(p,t)[1],error)
        self.assertTrue(torch.isfinite(error).all())

    def test_rotation_log_matches_finite_difference_at_identity_and_nonzero(self):
        p,t = states(batch=1,window=1)
        for q in (t[...,3:7].clone(),p[...,3:7].clone()):
            q.requires_grad_()
            def rotation_error(quat):
                state = torch.cat((p[...,:3],quat,p[...,7:]),-1)
                return physical_errors(state,t)[1]
            self.assertTrue(torch.autograd.gradcheck(rotation_error,(q,),atol=2e-5,rtol=2e-4))

    def test_calibration_balances_groups_once_without_graph_or_rng_change(self):
        p,t = states()
        p.requires_grad_()
        criterion = WorldModelStateLoss(TrainingConfig()).double()
        rng = torch.get_rng_state().clone()
        loss,raw,weighted = criterion(p,t)
        torch.testing.assert_close(weighted,torch.ones(6,dtype=torch.float64))
        torch.testing.assert_close(loss,p.new_tensor(18.))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertFalse(criterion.scales.requires_grad)
        self.assertFalse(criterion.scales.grad_fn is not None)
        scales = criterion.scales.clone()
        p2 = p.detach().clone()
        p2[...,:3] *= 2
        _,_,new_weighted = criterion(p2,t)
        torch.testing.assert_close(new_weighted[0],p.new_tensor(2.))
        torch.testing.assert_close(criterion.scales,scales,atol=0,rtol=0)
        with self.assertRaisesRegex(RuntimeError,"already calibrated"):
            criterion.calibrate(raw)
        loss.backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(float(p.grad.abs().sum()),0)

    def test_floor_zero_error_and_manual_weights_are_explicit(self):
        config = replace(TrainingConfig(),world_model_group_weights=(2.,1.,1.,1.,1.,1.))
        criterion = WorldModelStateLoss(config).double()
        p,t = states()
        p[...,:3] = t[...,:3]
        loss,_,weighted = criterion(p,t)
        self.assertEqual(criterion.scales[0].item(),config.world_model_scale_floor)
        self.assertTrue(criterion.summary()["floor_hit"][0])
        self.assertEqual(weighted[0].item(),0.)
        self.assertTrue(torch.isfinite(loss))
        p[...,:3] += config.world_model_scale_floor
        _,_,weighted = criterion(p,t)
        torch.testing.assert_close(weighted[0],p.new_tensor(2.))

    def test_calibration_roundtrip_survives_new_batch_and_rejects_missing_scales(self):
        config = TrainingConfig()
        p,t = states(dtype=torch.float32)
        criterion = WorldModelStateLoss(config)
        criterion(p,t)
        buffer = io.BytesIO()
        torch.save(criterion.state_dict(),buffer)
        buffer.seek(0)
        saved = torch.load(buffer,weights_only=True)
        restored = WorldModelStateLoss(config)
        restored.validate_saved_state(saved,completed_iterations=1)
        restored.load_state_dict(saved)
        p[...,7:36] *= 3
        _,_,weighted = restored(p,t)
        torch.testing.assert_close(weighted[4],p.new_tensor(3.))
        torch.testing.assert_close(restored.scales,criterion.scales,atol=0,rtol=0)
        bad = {**saved,"calibrated":torch.tensor(False)}
        with self.assertRaisesRegex(ValueError,"lacks calibrated"):
            restored.validate_saved_state(bad,completed_iterations=1)
        bad = {**saved,"scales":saved["scales"]*2}
        with self.assertRaisesRegex(ValueError,"do not match"):
            restored.validate_saved_state(bad,completed_iterations=1)

    def test_raw_mode_reproduces_previous_batch_mean_time_sum(self):
        p,t = states()
        t[...,3:7] *= -1
        criterion = WorldModelStateLoss(replace(TrainingConfig(),world_model_loss="raw_mse")).double()
        result,raw,weighted = criterion(p,t)
        expected = sum(((p[:,j]-torch.cat((t[:,j,:3],-t[:,j,3:7],t[:,j,7:]),-1))**2)
                       .sum(-1).mean() for j in range(p.shape[1]))
        torch.testing.assert_close(result,expected)
        self.assertIsNone(raw)
        self.assertIsNone(weighted)
        self.assertFalse(bool(criterion.calibrated))

    def test_invalid_shapes_config_and_calibration_fail(self):
        for kwargs in ({"world_model_loss":"unknown"},{"world_model_scale_floor":0.},
                       {"world_model_group_weights":(1.,)*5},
                       {"world_model_group_weights":(1.,)*5+(float("nan"),)},
                       {"world_model_termination_weight":float("inf")}):
            with self.assertRaises(ValueError):
                replace(TrainingConfig(),**kwargs)
        p,t = states()
        with self.assertRaises(ValueError):
            physical_errors(p,t[...,:70])
        criterion = WorldModelStateLoss(TrainingConfig())
        with self.assertRaises(ValueError):
            criterion.calibrate(torch.full((6,),float("nan")))


if __name__ == "__main__":
    unittest.main()
