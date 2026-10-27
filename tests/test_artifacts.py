"""Persistence and isolation checks for RunArtifacts."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from supermimic_study.training.artifacts import RunArtifacts


def transition_batch(num_envs=2):
    state = torch.arange(num_envs * 3, dtype=torch.float32).reshape(num_envs, 3).requires_grad_()
    return {
        "state": state,
        "action": torch.ones(num_envs, 2),
        "next_state": torch.ones(num_envs, 3) * 2,
        "history": torch.arange(num_envs * 4).reshape(num_envs, 4),
        "context": torch.ones(num_envs, 2),
        "clip_id": torch.arange(num_envs),
        "terminated": torch.zeros(num_envs, dtype=torch.bool),
        "truncated": torch.zeros(num_envs, dtype=torch.bool),
        "extra_debug": torch.arange(num_envs) + 100,
    }


class RunArtifactsTest(unittest.TestCase):
    def test_jsonl_records_are_flushed_and_reject_nan(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = RunArtifacts(directory)
            artifacts.event("start", seed=4)
            artifacts.metrics({"iteration": 1, "loss": 0.25})
            self.assertEqual(len((Path(directory) / "events.jsonl").read_text().splitlines()), 1)
            self.assertEqual(json.loads((Path(directory) / "metrics.jsonl").read_text()),
                             {"iteration": 1, "loss": 0.25})
            with self.assertRaises(ValueError):
                artifacts.metrics({"loss": float("nan")})
            self.assertEqual(len((Path(directory) / "metrics.jsonl").read_text().splitlines()), 1)

    def test_transition_chunks_have_one_detached_cpu_copy_per_env(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = RunArtifacts(directory)
            batch = transition_batch()
            artifacts.transitions(batch)
            batch["state"].data.fill_(999)
            saved_path = artifacts.flush_transitions()
            saved = torch.load(saved_path, map_location="cpu", weights_only=True)
            self.assertEqual(saved["vector_steps"], 1)
            self.assertEqual(len(saved["transitions"]), 2)
            self.assertEqual(set(saved["transitions"][0]), set(batch))
            self.assertFalse(saved["transitions"][0]["state"].requires_grad)
            self.assertEqual(saved["transitions"][0]["state"].tolist(), [0.0, 1.0, 2.0])
            self.assertEqual(saved["transitions"][1]["extra_debug"].item(), 101)

    def test_transition_autoflush_every_32_vector_steps_and_disabled_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = RunArtifacts(directory)
            for _ in range(32):
                artifacts.transitions(transition_batch(1))
            saved = torch.load(Path(directory) / "transitions/chunk-000001.pt",
                               map_location="cpu", weights_only=True)
            self.assertEqual(saved["vector_steps"], 32)
            self.assertEqual(len(saved["transitions"]), 32)
            disabled = RunArtifacts(Path(directory) / "off", record_transitions=False)
            disabled.transitions(transition_batch())
            self.assertIsNone(disabled.flush_transitions())

    def test_probe_is_weights_only_safe_and_does_not_change_rng_or_retain_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = RunArtifacts(directory)
            value = torch.tensor([3.0], requires_grad=True)
            rng = torch.random.get_rng_state().clone()
            path = artifacts.probe(7, {"value": value, "nested": [value]})
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            saved = torch.load(path, map_location="cpu", weights_only=True)
            self.assertFalse(saved["value"].requires_grad)
            self.assertFalse(saved["nested"][0].requires_grad)
            value.data.add_(10)
            self.assertEqual(saved["value"].item(), 3.0)

    def test_failed_atomic_torch_save_preserves_last_good_file(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = RunArtifacts(directory)
            path = artifacts.probe(1, {"v": torch.tensor([1])})
            before = path.read_bytes()
            with mock.patch("supermimic_study.training.artifacts.torch.save",
                            side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    artifacts.probe(1, {"v": torch.tensor([2])})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_periodic_retention_latest_and_emergency_do_not_replace_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = RunArtifacts(directory, checkpoint_every=10, keep_checkpoints=2)
            artifacts.checkpoint({"weight": torch.tensor([0.0])}, 0, tag="initial")
            for iteration in (10, 20, 30):
                artifacts.checkpoint({"weight": torch.tensor([float(iteration)], requires_grad=True)}, iteration)
            checkpoint_dir = Path(directory) / "checkpoints"
            self.assertTrue((checkpoint_dir / "initial-iteration-000000.pt").exists())
            self.assertFalse((checkpoint_dir / "iteration-000010.pt").exists())
            self.assertTrue((checkpoint_dir / "iteration-000020.pt").exists())
            latest_before = json.loads((checkpoint_dir / "latest.json").read_text())
            self.assertEqual(latest_before, {"path": "checkpoints/iteration-000030.pt",
                                             "iteration": 30, "tag": None})
            tagged = artifacts.checkpoint({"weight": torch.tensor([99])}, 31, tag="emergency")
            self.assertTrue(tagged.name.startswith("emergency-iteration-000031-"))
            self.assertEqual(json.loads((checkpoint_dir / "latest.json").read_text()), latest_before)
            loaded = torch.load(checkpoint_dir / latest_before["path"].split("/")[-1],
                                map_location="cpu", weights_only=True)
            self.assertFalse(loaded["weight"].requires_grad)
            artifacts.checkpoint({"weight": torch.tensor([7])}, 31, tag="final")
            with self.assertRaises(FileExistsError):
                artifacts.checkpoint({"weight": torch.tensor([8])}, 31, tag="final")

    def test_same_iteration_tags_have_distinct_files_and_only_complete_saves_update_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = RunArtifacts(directory, checkpoint_every=10, keep_checkpoints=2)
            initial = artifacts.checkpoint({"step": torch.tensor(0)}, 0, tag="initial")
            latest_initial = json.loads((Path(directory) / "checkpoints/latest.json").read_text())
            emergency_a = artifacts.checkpoint({"step": torch.tensor(0)}, 0, tag="emergency")
            emergency_b = artifacts.checkpoint({"step": torch.tensor(0)}, 0, tag="emergency")
            self.assertNotEqual(emergency_a, emergency_b)
            self.assertNotEqual(initial, emergency_a)
            self.assertEqual(json.loads((Path(directory) / "checkpoints/latest.json").read_text()), latest_initial)

            periodic = artifacts.checkpoint({"step": torch.tensor(10)}, 10)
            final = artifacts.checkpoint({"step": torch.tensor(10)}, 10, tag="final")
            self.assertNotEqual(periodic, final)
            latest_final = json.loads((Path(directory) / "checkpoints/latest.json").read_text())
            self.assertEqual(latest_final["path"], "checkpoints/final-iteration-000010.pt")
            self.assertEqual(latest_final["iteration"], 10)
            self.assertEqual(latest_final["tag"], "final")
            self.assertTrue(initial.exists())
            self.assertTrue(emergency_a.exists())
            self.assertTrue(emergency_b.exists())
            self.assertTrue(periodic.exists())
            self.assertTrue(final.exists())

    def test_retention_does_not_load_checkpoint_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = RunArtifacts(directory, checkpoint_every=1, keep_checkpoints=1)
            artifacts.checkpoint({"step": torch.tensor(1)}, 1)
            with mock.patch("supermimic_study.training.artifacts.torch.load",
                            side_effect=AssertionError("retention must not deserialize checkpoints")):
                artifacts.checkpoint({"step": torch.tensor(2)}, 2)
            checkpoint_dir = Path(directory) / "checkpoints"
            self.assertFalse((checkpoint_dir / "iteration-000001.pt").exists())
            self.assertTrue((checkpoint_dir / "iteration-000002.pt").exists())


if __name__ == "__main__":
    unittest.main()
