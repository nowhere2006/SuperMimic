import unittest
import torch
from supermimic_study.training.history import initial_history, advance_history, add_sensor_noise
from supermimic_study.training.sampler import ReturnBinSampler


class TrainingPrimitiveTests(unittest.TestCase):
    def test_history_term_major_and_gradient(self):
        first = torch.arange(93.).repeat(2, 1)
        next_frame = (first+100).requires_grad_()
        old = initial_history(first)
        new = advance_history(old, next_frame)
        torch.testing.assert_close(new[:, 261:290], next_frame[:, :29])
        torch.testing.assert_close(new[:, -29:], next_frame[:, -29:])
        new.sum().backward()
        torch.testing.assert_close(next_frame.grad, torch.ones_like(next_frame))
        noisy = add_sensor_noise(first)
        torch.testing.assert_close(noisy[:, -29:], first[:, -29:])
        self.assertTrue((noisy[:, :29]-first[:, :29]).abs().max() <= .01001)

    def test_sampler_mixture_and_return_updates(self):
        sampler = ReturnBinSampler([123, 70])
        self.assertAlmostEqual(float(sampler.probabilities().sum()), 1., places=6)
        c, f, b = sampler.sample(100)
        self.assertTrue((f < torch.tensor([122, 69])[c]).all())
        sampler.update_episode([0,0], [0,1], [.1,.2])
        self.assertEqual(int(sampler.visits[0]), 1)
        self.assertAlmostEqual(float(sampler.returns[0]), .1*((.1+.99*.2)+.2)/2, places=6)
        restored = ReturnBinSampler([123,70])
        restored.load_state_dict(sampler.state_dict())
        for name,value in sampler.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name],value)
        incompatible = ReturnBinSampler([124,70])
        with self.assertRaisesRegex(ValueError,"bins differ"):
            incompatible.load_state_dict(sampler.state_dict())


if __name__ == "__main__":
    unittest.main()
