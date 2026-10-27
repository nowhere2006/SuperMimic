"""Small CPU checks for the Table A4 network interfaces and initialization."""

import unittest

import torch

from supermimic_study.models import Critic, DynamicsModel, NetworkConfig, Policy


def tiny_config() -> NetworkConfig:
    return NetworkConfig(
        motion_encoder_hidden=(16, 8),
        policy_decoder_hidden=(16, 8),
        critic_hidden=(8, 4),
        dynamics_hidden=(16, 16),
    )


class NetworkInitializationTests(unittest.TestCase):
    def test_default_dimensions_match_table_a4_interfaces(self) -> None:
        config = NetworkConfig()
        self.assertEqual(config.motion_encoder_hidden, (2048, 1024, 512, 256))
        self.assertEqual(config.policy_decoder_hidden, (2048, 1024, 512, 256))
        self.assertEqual(config.critic_hidden, (256, 128))
        self.assertEqual(config.dynamics_hidden, (2048, 2048, 2048))
        self.assertEqual(config.history * 93, 930)
        self.assertEqual(config.future_frames * 68, 680)
        self.assertEqual(config.latent_dim, 32)

    def test_policy_has_expected_shape_and_bounded_motion_latent(self) -> None:
        model = Policy(tiny_config())
        observation = torch.randn(2, 930)
        goal = torch.randn(2, 680)
        latent = model.encode_goal(goal)
        self.assertTrue(torch.all(latent >= -1.0))
        self.assertTrue(torch.all(latent <= 1.0))
        self.assertEqual(model(observation, goal).shape, (2, 29))
        self.assertTrue(torch.all(model.decoder[-1].bias == 0))
        torch.testing.assert_close(
            torch.linalg.svdvals(model.decoder[-1].weight),
            torch.full((8,), 0.01),
            atol=1e-6,
            rtol=1e-5,
        )

    def test_critic_and_dynamics_output_shapes(self) -> None:
        config = tiny_config()
        critic = Critic(config)
        dynamics = DynamicsModel(config)
        state = torch.randn(3, 71)
        action = torch.randn(3, 29)
        context = torch.randn(3, 33)
        goal = torch.randn(3, 680)
        self.assertEqual(critic(state, goal, context,torch.zeros(3,4)).shape, (3, 1))
        self.assertEqual(critic.value_network[0].in_features,788)
        # Zero initialized delta yields the codec's zero-delta integration.
        torch.testing.assert_close(
            dynamics(state, action, context), dynamics.codec.integrate(state, torch.zeros(3, 70))
        )
        self.assertTrue(torch.count_nonzero(dynamics.delta_network[-1].weight) == 0)
        self.assertTrue(torch.count_nonzero(dynamics.delta_network[-1].bias) == 0)

    def test_invalid_last_dimensions_raise_clear_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "observation.*last dimension 930"):
            Policy(tiny_config())(torch.randn(2, 929), torch.randn(2, 680))
        with self.assertRaisesRegex(ValueError, "goal.*last dimension 680"):
            Critic(tiny_config())(torch.randn(2, 71), torch.randn(2, 679), torch.randn(2, 33))
        with self.assertRaisesRegex(ValueError, "action.*last dimension 29"):
            DynamicsModel(tiny_config())(torch.randn(2, 71), torch.randn(2, 28), torch.randn(2, 33))

    def test_forward_preserves_autograd_and_does_not_change_parameters(self) -> None:
        models_and_inputs = [
            (Policy(tiny_config()), (torch.randn(2, 930), torch.randn(2, 680))),
            (Critic(tiny_config()), (torch.randn(2, 71), torch.randn(2, 680), torch.randn(2, 33),torch.randn(2,4))),
            (DynamicsModel(tiny_config()), (torch.randn(2, 71), torch.randn(2, 29), torch.randn(2, 33))),
        ]
        for model, inputs in models_and_inputs:
            before = [p.detach().clone() for p in model.parameters()]
            output = model(*inputs)
            self.assertTrue(output.requires_grad)
            self.assertEqual([p.grad for p in model.parameters()], [None] * len(before))
            for previous, current in zip(before, model.parameters(), strict=True):
                torch.testing.assert_close(current, previous)

    def test_observation_scaling_matches_term_major_history(self) -> None:
        model = Policy(tiny_config())
        captured = []
        handle = model.decoder[0].register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        try:
            model(torch.ones(1, 930), torch.zeros(1, 680))
        finally:
            handle.remove()
        scaled = captured[0][0, -930:]
        torch.testing.assert_close(scaled[:290], torch.ones(290))
        torch.testing.assert_close(scaled[290:580], torch.full((290,), 0.2))
        torch.testing.assert_close(scaled[580:610], torch.full((30,), 0.5))
        torch.testing.assert_close(scaled[610:], torch.ones(320))


if __name__ == "__main__":
    unittest.main()
