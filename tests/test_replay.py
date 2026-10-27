"""Replay chronology, episode boundaries, and storage isolation checks."""
from __future__ import annotations

import unittest

import torch

from supermimic_study.training.replay import ReplayBuffer


def batch(num_envs: int, *, frame: int = 0, terminated=None, truncated=None) -> dict[str, torch.Tensor]:
    terminated = torch.zeros(num_envs, dtype=torch.bool) if terminated is None else torch.tensor(terminated, dtype=torch.bool)
    truncated = torch.zeros(num_envs, dtype=torch.bool) if truncated is None else torch.tensor(truncated, dtype=torch.bool)
    base = torch.arange(num_envs, dtype=torch.float32)
    observation_rows = []
    for env in range(num_envs):
        terms = []
        for term, width in enumerate((29, 29, 3, 3, 29)):
            time = torch.arange(10, dtype=torch.float32).reshape(10, 1)
            feature = torch.arange(width, dtype=torch.float32).reshape(1, width)
            terms.append((frame * 10000 + env * 100000 + term * 1000 + time * 100 + feature).reshape(-1))
        observation_rows.append(torch.cat(terms))
    return {
        "state": torch.ones(num_envs, 71) * (base[:, None] * 1000 + frame),
        "action": torch.full((num_envs, 29), float(frame)),
        "policy_action": torch.full((num_envs,29),float(frame)+.1),
        "executed_action": torch.full((num_envs,29),float(frame)),
        "previous_policy_action": torch.full((num_envs,29),float(frame-1)+.1),
        "previous_executed_action": torch.full((num_envs,29),float(frame-1)),
        "next_state": torch.ones(num_envs, 71) * (base[:, None] * 1000 + frame + 1),
        "reward": torch.full((num_envs,), float(frame)),
        "terminated": terminated,
        "truncated": truncated,
        "context": torch.full((num_envs, 33), 2.0),
        "clip_id": torch.zeros(num_envs, dtype=torch.long),
        "frame_id": torch.full((num_envs,), frame, dtype=torch.long),
        "next_frame_id": torch.full((num_envs,), frame + 1, dtype=torch.long),
        "observation": torch.stack(observation_rows),
        "previous_action": torch.full((num_envs, 29), float(frame - 1)+.1),
    }


class ReplayBufferTest(unittest.TestCase):
    def test_ring_wrap_keeps_only_contiguous_recent_windows(self) -> None:
        replay = ReplayBuffer(1, capacity=4, seed=3)
        for frame in range(6):
            replay.add(batch(1, frame=frame))
        sampled = replay.sample_windows(20, 3)
        for frames in sampled["frame_id"]:
            self.assertEqual(frames.tolist(), list(range(int(frames[0]), int(frames[0]) + 3)))
            self.assertGreaterEqual(int(frames[0]), 2)
            self.assertLessEqual(int(frames[-1]), 5)
        with self.assertRaisesRegex(ValueError, "not enough contiguous replay"):
            replay.sample_windows(1, 5)

    def test_terminal_may_end_window_but_no_window_crosses_reset(self) -> None:
        replay = ReplayBuffer(1, capacity=10, seed=0)
        for frame in range(3):
            replay.add(batch(1, frame=frame, terminated=[frame == 2]))
        window = replay.sample_windows(1, 3)
        self.assertEqual(window["terminated"].tolist(), [[False, False, True]])
        # The next frame number restarts at zero, but belongs to a new episode.
        replay.add(batch(1, frame=0))
        windows = replay.sample_windows(20, 2)
        self.assertFalse(any(row.tolist() == [2, 0] for row in windows["frame_id"]))

    def test_truncated_clamped_tail_is_an_allowed_last_transition(self) -> None:
        replay = ReplayBuffer(1, capacity=4)
        replay.add(batch(1, frame=7))
        final = batch(1, frame=8, truncated=[True])
        final["next_frame_id"].fill_(8)
        final["next_state"].fill_(123.0)  # pre-reset state is retained verbatim
        replay.add(final)
        result = replay.sample_windows(1, 2)
        self.assertTrue(result["truncated"][0, 1])
        torch.testing.assert_close(result["next_state"][0, 1], torch.full((71,), 123.0))

    def test_windows_reject_clip_context_or_state_discontinuity(self) -> None:
        for field in ("clip_id", "context", "state"):
            with self.subTest(field=field):
                replay = ReplayBuffer(1, capacity=4)
                replay.add(batch(1, frame=0))
                second = batch(1, frame=1)
                if field == "clip_id":
                    second[field].fill_(1)
                elif field == "context":
                    second[field].fill_(3.0)
                else:
                    second[field].add_(0.1)
                replay.add(second)
                with self.assertRaisesRegex(ValueError, "not enough contiguous replay"):
                    replay.sample_windows(1, 2)

    def test_add_detaches_and_copies_simulator_buffers(self) -> None:
        replay = ReplayBuffer(1, capacity=3)
        incoming = batch(1, frame=0)
        incoming["state"].requires_grad_()
        replay.add(incoming)
        incoming["state"].data.fill_(999.0)
        starts = replay.sample_starts(1)
        self.assertFalse(starts["state"].requires_grad)
        self.assertNotEqual(float(starts["state"][0, 0]), 999.0)

    def test_window_batch_preserves_time_axis_and_start_has_term_major_history(self) -> None:
        replay = ReplayBuffer(2, capacity=8, seed=14)
        for frame in range(5):
            replay.add(batch(2, frame=frame))
        windows = replay.sample_windows(8, 4)
        self.assertEqual(windows["state"].shape, (8, 4, 71))
        self.assertEqual(windows["action"].shape, (8, 4, 29))
        for row in windows["frame_id"]:
            self.assertEqual(row.tolist(), list(range(int(row[0]), int(row[0]) + 4)))
        starts = replay.sample_starts(3)
        self.assertEqual(starts["history"].shape, (3, 10, 93))
        obs = starts["observation"][0]
        widths = (29, 29, 3, 3, 29)
        offsets = [0]
        for width in widths:
            offsets.append(offsets[-1] + 10 * width)
        expected = torch.cat([
            obs[offsets[i]:offsets[i + 1]].reshape(10, width)
            for i, width in enumerate(widths)
        ], dim=-1)
        torch.testing.assert_close(starts["history"][0], expected)
        # Non-constant sentinels detect both term-boundary and time-axis mistakes.
        self.assertNotEqual(float(expected[0, 0]), float(expected[1, 0]))
        self.assertNotEqual(float(expected[0, 0]), float(expected[0, 29]))
        self.assertEqual(starts["previous_action"].shape, (3, 29))
        self.assertEqual(starts["context"].shape, (3, 33))

    def test_invalid_schema_or_nonfinite_data_fails_before_write(self) -> None:
        replay = ReplayBuffer(1, capacity=4)
        incomplete = batch(1)
        del incomplete["previous_action"]
        with self.assertRaisesRegex(ValueError, "previous_action"):
            replay.add(incomplete)
        malformed = batch(1)
        malformed["state"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "state contains non-finite"):
            replay.add(malformed)
        with self.assertRaisesRegex(ValueError, "not enough contiguous replay"):
            replay.sample_windows(1, 1)


if __name__ == "__main__":
    unittest.main()
