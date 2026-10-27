"""Known-geometry checks independent of network shapes or loss implementations."""
import math
import unittest
import torch

from supermimic_study.models.state import (
    StateCodec, heading_quat, quat_rotate, quat_rotate_inverse, rotation_features,
)


class StateCodecTest(unittest.TestCase):
    def test_yaw_rotation_and_canonical_invariance(self) -> None:
        yaw = torch.tensor([[math.sqrt(0.5), 0, 0, math.sqrt(0.5)]])
        world_x = torch.tensor([[1.0, 0, 0]])
        torch.testing.assert_close(quat_rotate_inverse(yaw, world_x), torch.tensor([[0.0, -1, 0]]), atol=1e-6, rtol=0)
        torch.testing.assert_close(rotation_features(yaw), torch.tensor([[0.0, -1, 1, 0, 0, 0]]), atol=1e-6, rtol=0)
        state = torch.zeros(1, 71)
        state[:, 2] = 0.8
        state[:, 3] = 1
        state[:, 36] = 1
        rotated = state.clone()
        rotated[:, :2] = torch.tensor([[4.0, -3.0]])
        rotated[:, 3:7] = yaw
        rotated[:, 36:39] = quat_rotate(yaw, state[:, 36:39])
        torch.testing.assert_close(StateCodec().canonical(state), StateCodec().canonical(rotated), atol=1e-6, rtol=0)

    def test_zero_delta_extrapolates_known_motion(self) -> None:
        state = torch.zeros(1, 71)
        state[:, 3] = 1
        state[:, 36:39] = torch.tensor([[1.0, -2, 0.5]])
        state[:, 41] = 1
        state[:, 42:] = 0.5
        result = StateCodec().integrate(state, torch.zeros(1, 70))
        torch.testing.assert_close(result[:, :3], torch.tensor([[0.02, -0.04, 0.01]]))
        torch.testing.assert_close(result[:, 3:7], torch.tensor([[math.cos(0.01), 0, 0, math.sin(0.01)]]))
        torch.testing.assert_close(result[:, 7:36], torch.full((1, 29), 0.01))

    def test_small_angle_integration_has_finite_derivatives(self) -> None:
        state = torch.zeros(2, 71)
        state[:, 3] = 1
        state.requires_grad_()
        delta = torch.zeros(2, 70, requires_grad=True)
        StateCodec().integrate(state, delta).sum().backward()
        self.assertTrue(torch.isfinite(state.grad).all())
        self.assertTrue(torch.isfinite(delta.grad).all())


if __name__ == "__main__":
    unittest.main()
