"""Differentiable G1 forward kinematics from MuJoCo model constants.

The model is inspected once on construction. Runtime positions and velocities
are then composed in Torch, keeping gradients from tracked bodies to state.
"""
from __future__ import annotations

import torch
from torch import nn

from .state import quat_multiply, quat_normalize, quat_rotate


class DifferentiableG1Kinematics(nn.Module):
    """FK for the floating-base G1 with zero-anchor scalar joints and fixed links.

    State layout is ``[root p(3), root quat wxyz(4), q(29), v_world(3),
    omega_root_body(3), qdot(29)]``. Outputs follow MuJoCo body id order for
    the floating-root body and its descendants only; unrelated scene bodies
    such as terrain are excluded.
    """

    def __init__(self, mj_model, body_names=None):
        super().__init__()
        import mujoco
        free_joint_bodies = [int(mj_model.jnt_bodyid[j]) for j in range(mj_model.njnt)
                             if mj_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if len(free_joint_bodies) != 1:
            raise ValueError(f"expected exactly one floating free joint, found {len(free_joint_bodies)}")
        root_id = free_joint_bodies[0]
        selected_ids = []
        for bi in range(1, mj_model.nbody):
            ancestor = bi
            while ancestor != 0 and ancestor != root_id:
                ancestor = int(mj_model.body_parentid[ancestor])
            if ancestor == root_id:
                selected_ids.append(bi)
        if root_id not in selected_ids:
            raise ValueError("floating free-joint body is not in the MuJoCo body tree")
        self.body_ids = tuple(selected_ids)
        output_index = {body_id: i + 1 for i, body_id in enumerate(self.body_ids)}
        self.body_names = tuple(mj_model.body(i).name.removeprefix("robot/")
                                for i in self.body_ids)
        if body_names is not None and tuple(body_names) != self.body_names:
            raise ValueError("body_names must match the floating robot's MuJoCo body order")
        parent, pos, quat, joint_axis = [], [], [], []
        for bi in self.body_ids:
            parent_id = int(mj_model.body_parentid[bi])
            if bi == root_id:
                parent.append(0)
            elif parent_id in output_index:
                parent.append(output_index[parent_id])
            else:
                raise ValueError(f"robot body {mj_model.body(bi).name} has no selected parent")
            pos.append(mj_model.body_pos[bi].copy())
            quat.append(mj_model.body_quat[bi].copy())
            adr = int(mj_model.body_jntadr[bi])
            count = int(mj_model.body_jntnum[bi])
            entries = []
            for ji in range(adr, adr + count):
                typ = int(mj_model.jnt_type[ji])
                # The floating root transform and twist come directly from the
                # explicit state; do not apply the MuJoCo free joint twice.
                if typ == mujoco.mjtJoint.mjJNT_FREE:
                    continue
                if typ not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                    raise ValueError(f"unsupported non-scalar joint on {mj_model.body(bi).name}")
                if (abs(mj_model.jnt_pos[ji]) > 1e-10).any():
                    raise ValueError("nonzero joint anchors are unsupported by this G1 kinematics codec")
                entries.append((mj_model.jnt_axis[ji].copy(), int(mj_model.jnt_qposadr[ji]),
                                int(mj_model.jnt_dofadr[ji]), typ == mujoco.mjtJoint.mjJNT_SLIDE))
            joint_axis.append(entries)
        self.register_buffer("parent", torch.tensor(parent, dtype=torch.long))
        import numpy as np
        self.register_buffer("rest_pos", torch.as_tensor(np.asarray(pos), dtype=torch.float32))
        self.register_buffer("rest_quat", quat_normalize(torch.as_tensor(np.asarray(quat), dtype=torch.float32)))
        # Variable-length joints are kept as immutable Python metadata.
        self._joints = tuple(tuple(x) for x in joint_axis)
        self._nq = int(mj_model.nq)
        self._nv = int(mj_model.nv)
        if self._nq != 36 or self._nv != 35:
            raise ValueError(f"expected floating G1 nq/nv=36/35, got {self._nq}/{self._nv}")

    def forward(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        if state.shape[-1] != 71:
            raise ValueError("state must end in 71 values")
        batch = state.shape[:-1]
        p0, q0 = state[..., :3], quat_normalize(state[..., 3:7])
        v0 = state[..., 36:39]
        w0 = quat_rotate(q0, state[..., 39:42])
        pos_out, quat_out, lin_out, ang_out = [], [], [], []
        for i, parent in enumerate(self.parent.tolist()):
            # A floating body's compiled rest pose is also the free-joint
            # initial pose. qpos already stores its absolute pose.
            if parent == 0:
                pos_out.append(p0)
                quat_out.append(q0)
                lin_out.append(v0)
                ang_out.append(w0)
                continue
            parent_pos = p0 if parent == 0 else pos_out[parent - 1]
            parent_quat = q0 if parent == 0 else quat_out[parent - 1]
            parent_lin = v0 if parent == 0 else lin_out[parent - 1]
            parent_ang = w0 if parent == 0 else ang_out[parent - 1]
            rp = self.rest_pos[i].to(state)
            rq = self.rest_quat[i].to(state)
            local_pos = rp.expand(*batch, 3)
            local_quat = rq.expand(*batch, 4)
            # MuJoCo scalar joints rotate/translate about the body-local joint axis.
            for axis_np, qa, da, slide in self._joints[i]:
                axis = torch.as_tensor(axis_np, dtype=state.dtype, device=state.device)
                axis = axis / torch.linalg.vector_norm(axis).clamp_min(1e-12)
                # MuJoCo qpos address includes the floating root (7); this
                # state's joint-position slice also begins at absolute index 7.
                value = state[..., qa]
                rate = state[..., 42 + (da - 6)]
                if slide:
                    local_pos = local_pos + quat_rotate(local_quat, axis * value[..., None])
                else:
                    half = value * 0.5
                    jq = torch.cat((half.cos()[..., None], axis * half.sin()[..., None]), dim=-1)
                    local_quat = quat_normalize(quat_multiply(local_quat, jq))
            body_pos = parent_pos + quat_rotate(parent_quat, local_pos)
            body_quat = quat_normalize(quat_multiply(parent_quat, local_quat))
            angular = parent_ang
            linear = parent_lin + torch.linalg.cross(parent_ang, body_pos - parent_pos, dim=-1)
            for axis_np, qa, da, slide in self._joints[i]:
                axis = torch.as_tensor(axis_np, dtype=state.dtype, device=state.device)
                axis = axis / torch.linalg.vector_norm(axis).clamp_min(1e-12)
                rate = state[..., 42 + (da - 6)]
                world_axis = quat_rotate(body_quat, axis.expand(*batch, 3))
                if slide:
                    linear = linear + world_axis * rate[..., None]
                else:
                    angular = angular + world_axis * rate[..., None]
            pos_out.append(body_pos)
            quat_out.append(body_quat)
            lin_out.append(linear)
            ang_out.append(angular)
        return {
            "body_pos_w": torch.stack(pos_out, dim=-2),
            "body_quat_w": torch.stack(quat_out, dim=-2),
            "body_lin_vel_w": torch.stack(lin_out, dim=-2),
            "body_ang_vel_w": torch.stack(ang_out, dim=-2),
        }
