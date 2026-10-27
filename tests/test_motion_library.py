from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from supermimic_study.data import MotionLibrary


JOINT_NAMES = [f"joint_{i}" for i in range(29)]
BODY_NAMES = [f"body_{i}" for i in range(30)]


def _arrays(frames: int, value: float) -> dict[str, np.ndarray]:
    arrays = {
        "joint_pos": np.full((frames, 29), value, dtype=np.float32),
        "joint_vel": np.full((frames, 29), value + 1, dtype=np.float32),
        "body_pos_w": np.full((frames, 30, 3), value + 2, dtype=np.float32),
        "body_quat_w": np.full((frames, 30, 4), value + 3, dtype=np.float32),
        "body_lin_vel_w": np.full((frames, 30, 3), value + 4, dtype=np.float32),
        "body_ang_vel_w": np.full((frames, 30, 3), value + 5, dtype=np.float32),
    }
    for array in arrays.values():
        array += np.arange(frames, dtype=np.float32).reshape(frames, *([1] * (array.ndim - 1))) * 0.1
    return arrays


class MotionLibraryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.clips_dir = self.root / "clips"
        self.clips_dir.mkdir()
        self.manifest: dict = {
            "schema_version": 1,
            "fps": 50,
            "joint_names": JOINT_NAMES,
            "body_names": BODY_NAMES,
            "clips": [],
        }
        self._write_clip("first", 2, 10.0)
        self._write_clip("second", 3, 20.0)
        self._write_manifest()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_clip(self, name: str, frames: int, value: float, *, fps: int = 50,
                    shape_frames: int | None = None) -> Path:
        path = self.clips_dir / f"{name}.npz"
        arrays = _arrays(shape_frames if shape_frames is not None else frames, value)
        np.savez(path, fps=np.array([fps], dtype=np.int64),
                 joint_names=np.asarray(JOINT_NAMES), body_names=np.asarray(BODY_NAMES), **arrays)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.manifest["clips"].append({
            "name": name,
            "path": f"clips/{name}.npz",
            "frames": frames,
            "duration_s": (frames - 1) / 50,
            "source_sha256": "a" * 64,
            "npz_sha256": digest,
        })
        return path

    def _write_manifest(self) -> None:
        (self.root / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_concatenation_and_clip_local_indices_cross_correctly(self) -> None:
        library = MotionLibrary.from_directory(self.root)
        self.assertEqual(library.clip_names, ("first", "second"))
        self.assertEqual(library.frame_counts, (2, 3))
        self.assertEqual(library.offsets, (0, 2))
        self.assertEqual((library.total_frames, library.num_clips, library.fps), (5, 2, 50))
        self.assertGreater(library.storage_bytes, 0)
        result = library.get_frames([0, 1, 1], [1, 0, 2])
        self.assertEqual(result["joint_pos"].shape, (3, 29))
        torch.testing.assert_close(result["joint_pos"][:, 0], torch.tensor([10.1, 20.0, 20.2]))
        torch.testing.assert_close(result["body_pos_w"][:, 0, 0], torch.tensor([12.1, 22.0, 22.2]))
        result["joint_pos"][0, 0] = -999
        again = library.get_frames([0], [1])
        self.assertAlmostEqual(float(again["joint_pos"][0, 0]), 10.1, places=5)

    def test_local_end_frame_out_of_range_is_rejected(self) -> None:
        library = MotionLibrary.from_directory(self.root)
        with self.assertRaises(IndexError):
            library.get_frames([0], [2])
        with self.assertRaises(IndexError):
            library.get_frames([1], [-1])

    def test_shape_fps_and_npz_hash_errors_are_rejected(self) -> None:
        malformed = self.root / "bad-shape"
        malformed.mkdir()
        npz = malformed / "clip.npz"
        values = _arrays(1, 0)
        values["joint_pos"] = np.zeros((2, 29), dtype=np.float32)
        np.savez(npz, fps=np.array([50]), **values)
        manifest = {"schema_version": 1, "fps": 50, "joint_names": JOINT_NAMES,
                    "body_names": BODY_NAMES, "clips": [{"name": "bad", "path": "clip.npz",
                    "frames": 1, "duration_s": 0, "source_sha256": "b" * 64}]}
        (malformed / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "joint_pos shape"):
            MotionLibrary.from_directory(malformed)

        wrong_fps = self.root / "bad-fps"
        wrong_fps.mkdir()
        np.savez(wrong_fps / "clip.npz", fps=np.array([30]), **_arrays(1, 0))
        manifest["clips"][0]["path"] = "clip.npz"
        (wrong_fps / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "NPZ fps"):
            MotionLibrary.from_directory(wrong_fps)

        self.manifest["clips"][0]["npz_sha256"] = "0" * 64
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            MotionLibrary.from_directory(self.root)

    def test_empty_manifest_and_path_escape_are_rejected(self) -> None:
        empty = self.root / "empty"
        empty.mkdir()
        manifest = {"schema_version": 1, "fps": 50, "joint_names": JOINT_NAMES,
                    "body_names": BODY_NAMES, "clips": []}
        (empty / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "non-empty"):
            MotionLibrary.from_directory(empty)

        self.manifest["clips"][0]["path"] = "../outside.npz"
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, "escapes"):
            MotionLibrary.from_directory(self.root)

    def test_duration_and_nonfinite_data_are_rejected(self) -> None:
        self.manifest["clips"][0]["duration_s"] = 99
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, "duration_s"):
            MotionLibrary.from_directory(self.root)
        self.manifest["clips"][0]["duration_s"] = 1 / 50
        path = self.root / self.manifest["clips"][0]["path"]
        arrays = _arrays(2, 10)
        arrays["joint_vel"][0, 0] = np.nan
        np.savez(path, fps=np.array([50]), **arrays)
        self.manifest["clips"][0]["npz_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, "non-finite"):
            MotionLibrary.from_directory(self.root)


if __name__ == "__main__":
    unittest.main()
