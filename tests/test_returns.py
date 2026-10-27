import unittest
import torch
from supermimic_study.training.returns import imagined_objective, gae_targets
from supermimic_study.training.loop import frozen


class ReturnTests(unittest.TestCase):
    def test_frozen_weights_preserve_input_gradient_and_restore_flags(self):
        critic = torch.nn.Linear(2,1,bias=False)
        with torch.no_grad():
            critic.weight.copy_(torch.tensor([[2.,3.]]))
        state = torch.ones(1,2,requires_grad=True)
        with frozen(critic):
            critic(state).sum().backward()
            self.assertIsNone(critic.weight.grad)
        torch.testing.assert_close(state.grad,torch.tensor([[2.,3.]]))
        self.assertTrue(critic.weight.requires_grad)

    def test_terminal_reward_is_included_but_later_reward_is_not(self):
        reward = torch.tensor([[2.],[100.]],requires_grad=True)
        termination = torch.tensor([[1.],[0.]],requires_grad=True)
        value = imagined_objective(reward,termination,torch.tensor([1000.]),.9)
        self.assertEqual(float(value.detach()),2.)
        value.backward()
        self.assertEqual(float(reward.grad[0]),1.)
        self.assertEqual(float(reward.grad[1]),0.)
        self.assertNotEqual(float(termination.grad[0]),0.)

    def test_fixed_gae_matches_hand_calculation(self):
        r = torch.tensor([[1.],[2.]])
        d = torch.tensor([[0.],[.5]])
        v = torch.tensor([[3.],[4.],[5.]],requires_grad=True)
        target = gae_targets(r,d,v,.9,.95)
        last = 2+.9*.5*5
        first = 1+.9*(.05*4+.95*last)
        torch.testing.assert_close(target,torch.tensor([[first],[last]]))
        self.assertFalse(target.requires_grad)
