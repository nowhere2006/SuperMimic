import mujoco
import numpy as np
import torch
import unittest

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
from supermimic_study.models.heads import (
    TRACKING_BODY_NAMES, hard_termination, tracking_reward,
)
from supermimic_study.models.kinematics import DifferentiableG1Kinematics
from supermimic_study.models.world_model import WorldModel


def _native_model():
    return mujoco.MjModel.from_xml_path(str(G1_XML))


def test_torch_fk_matches_mujoco_random_state_and_twist():
    model = _native_model()
    data = mujoco.MjData(model)
    fk = DifferentiableG1Kinematics(model).double()
    rng = np.random.default_rng(1729)
    for _ in range(3):
        qpos = np.zeros(model.nq)
        qpos[:3] = rng.normal(size=3)
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        angle = rng.uniform(-1.2, 1.2)
        qpos[3:7] = np.r_[np.cos(angle/2), axis*np.sin(angle/2)]
        qpos[7:] = rng.uniform(-0.7, 0.7, model.nq-7)
        qvel = rng.normal(0., 0.4, model.nv)
        data.qpos[:] = qpos; data.qvel[:] = qvel
        mujoco.mj_forward(model, data)
        state = torch.tensor(np.concatenate((qpos, qvel)), dtype=torch.float64)[None]
        predicted = fk(state)
        for output_i, i in enumerate(fk.body_ids):
            assert np.allclose(predicted["body_pos_w"][0, output_i].numpy(), data.xpos[i], atol=2e-6)
            assert abs(float(np.dot(predicted["body_quat_w"][0, output_i].numpy(), data.xquat[i]))) > 1-2e-7
            jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jacp, jacr, data.xpos[i], i)
            assert np.allclose(predicted["body_lin_vel_w"][0, output_i].numpy(), jacp @ qvel, atol=3e-6)
            assert np.allclose(predicted["body_ang_vel_w"][0, output_i].numpy(), jacr @ qvel, atol=3e-6)


def test_fk_excludes_unrelated_scene_bodies():
    spec = mujoco.MjSpec.from_file(str(G1_XML))
    terrain = spec.worldbody.add_body(name="terrain")
    terrain.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0., 0., .1])
    model = spec.compile()
    fk = DifferentiableG1Kinematics(model)
    assert len(fk.body_names) == 30
    assert "terrain" not in fk.body_names
    assert fk.body_ids[0] == next(i for i in range(1, model.nbody) if model.body(i).name == "pelvis")
    data = mujoco.MjData(model)
    qpos = np.zeros(model.nq); qpos[2] = .8; qpos[3] = 1.
    qpos[7:] = np.linspace(-.3, .3, 29)
    qvel = np.linspace(-.2, .2, model.nv)
    data.qpos[:] = qpos; data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    state = torch.tensor(np.concatenate((qpos, qvel)), dtype=torch.float32)[None]
    predicted = fk(state)
    for output_i, body_id in enumerate(fk.body_ids):
        assert np.allclose(predicted["body_pos_w"][0, output_i].numpy(), data.xpos[body_id], atol=2e-6)


def test_tracking_reward_has_finite_nonzero_state_gradient_and_identity_is_safe():
    model = _native_model()
    fk = DifferentiableG1Kinematics(model)
    state = torch.zeros(2, 71, requires_grad=True)
    # Upright identity root and nonzero height; vary joints to give reward a path.
    with torch.no_grad():
        state[:, 2] = 0.8
        state[:, 3] = 1.0
        state[:, 7:36] = torch.linspace(-0.2, 0.2, 29)
    current = fk(state)
    reference = {k: v.detach().clone() for k, v in current.items()}
    reference["body_pos_w"] = reference["body_pos_w"] + torch.tensor([0.03, 0., 0.]).view(1, 1, 3)
    zeros = torch.zeros(2, 29)
    limits = torch.tensor([[-1.0, 1.0]]).expand(29, 2)
    reward = tracking_reward(current, reference, zeros, zeros, state[:, 7:36], limits,
                             body_names=fk.body_names)
    assert reward.shape == (2,) and torch.isfinite(reward).all()
    reward.sum().backward()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert state.grad.abs().sum() > 0
    # Equal rotations use the stable identity-safe SO(3) distance path.
    identical_reward = tracking_reward(current, current, zeros, zeros, state[:, 7:36], limits,
                                       body_names=fk.body_names)
    assert torch.isfinite(identical_reward).all()
    term = hard_termination(current, current, fk.body_names)
    assert not term.any()


def test_world_model_exposes_separate_differentiable_heads():
    model = _native_model()
    fk = DifferentiableG1Kinematics(model)
    wm = WorldModel(kinematics=fk, default_q=torch.zeros(29),
                    joint_limits=torch.tensor([[-1., 1.]]).expand(29, 2))
    state = torch.zeros(1, 71, requires_grad=True)
    with torch.no_grad():
        state[:, 2] = 0.8; state[:, 3] = 1
    action, context = torch.zeros(1, 29), torch.zeros(1, 33)
    context[:, 0] = .6
    current = fk(state)
    result = wm(state, action, context, previous_policy_action=action, policy_action=action,
                reference={k: v.detach() for k, v in current.items()}, body_names=fk.body_names)
    assert result["observation"].shape == (1, 93)
    assert result["next_state"].shape == (1, 71)
    assert result["reward"].shape == (1,)
    assert result["termination_logit"].shape == (1,)
    objective = result["reward"].sum() + result["termination_logit"].sum()
    objective.backward()
    assert state.grad is not None and torch.isfinite(state.grad).all()


def test_paper_reward_weights_and_pelvis_failure_thresholds():
    model = _native_model()
    fk = DifferentiableG1Kinematics(model)
    state = torch.zeros(1,71)
    state[:,2],state[:,3] = .8,1.
    reference = fk(state)
    current = {k:v.clone() for k,v in reference.items()}
    current["body_pos_w"][...,0] += .3
    zero = torch.zeros(1,29)
    limits = torch.tensor([[-10.,10.]]).expand(29,2)
    reward = tracking_reward(current,reference,zero,zero,state[:,7:36],limits,body_names=fk.body_names)
    torch.testing.assert_close(reward,torch.tensor([.02*(4.5+.5*np.exp(-1))],dtype=torch.float32))
    action_reward = tracking_reward(current,reference,zero+.5,zero,state[:,7:36],limits,body_names=fk.body_names)
    torch.testing.assert_close(reward-action_reward,torch.tensor([.02*.1*29*.25]))
    pelvis = fk.body_names.index("pelvis")
    wrist = fk.body_names.index("left_wrist_yaw_link")
    failed = {k:v.clone() for k,v in reference.items()}
    failed["body_pos_w"][:,wrist] += 100
    assert not hard_termination(failed,reference,fk.body_names).any()
    failed["body_pos_w"][:,pelvis,2] += .249
    assert not hard_termination(failed,reference,fk.body_names).any()
    failed["body_pos_w"][:,pelvis,2] += .002
    assert hard_termination(failed,reference,fk.body_names).all()
    tilted = {k:v.clone() for k,v in reference.items()}
    tilted["body_quat_w"][:,pelvis] = torch.tensor([0.,1.,0.,0.])
    assert hard_termination(tilted,reference,fk.body_names).all()


def load_tests(loader, standard_tests, pattern):
    """Expose the function-style checks to ``python -m unittest discover``."""
    suite = unittest.TestSuite()
    for check in (
        test_torch_fk_matches_mujoco_random_state_and_twist,
        test_fk_excludes_unrelated_scene_bodies,
        test_tracking_reward_has_finite_nonzero_state_gradient_and_identity_is_safe,
        test_world_model_exposes_separate_differentiable_heads,
        test_paper_reward_weights_and_pelvis_failure_thresholds,
    ):
        suite.addTest(unittest.FunctionTestCase(check, description=check.__name__))
    return suite
