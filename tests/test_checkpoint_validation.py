from dataclasses import asdict
from types import SimpleNamespace
import unittest

from supermimic_study.models.networks import NetworkConfig
from supermimic_study.training.config import TrainingConfig
from supermimic_study.training.loop import TrainingLoop
from supermimic_study.training.model_loss import WorldModelStateLoss


class CheckpointValidationTests(unittest.TestCase):
    def setUp(self):
        self.loop = TrainingLoop.__new__(TrainingLoop)
        self.loop.config = TrainingConfig()
        self.loop.policy = SimpleNamespace(config=NetworkConfig())
        self.loop.metadata = {"manifest_sha256":"a"*64}
        self.loop.model_loss = WorldModelStateLoss(self.loop.config)
        self.payload = {"format_version":2,"phase":"between_iterations",
                        "config":asdict(self.loop.config),"network_config":asdict(NetworkConfig()),
                        "metadata":{"manifest_sha256":"a"*64}}

    def test_partial_phase_and_emergency_are_not_continuation_checkpoints(self):
        self.payload["phase"] = "fit_world_model"
        with self.assertRaisesRegex(ValueError,"partial-phase"):
            self.loop.load_checkpoint(self.payload)
        self.payload["phase"] = "between_iterations"
        self.payload["artifact_tag"] = "emergency"
        with self.assertRaisesRegex(ValueError,"partial-phase"):
            self.loop.load_checkpoint(self.payload)

    def test_incomplete_config_and_changed_method_are_rejected_before_loading(self):
        del self.payload["config"]["gamma"]
        with self.assertRaisesRegex(ValueError,"fields differ"):
            self.loop.load_checkpoint(self.payload)
        self.payload["config"] = asdict(self.loop.config)
        self.payload["config"]["gamma"] = .8
        with self.assertRaisesRegex(ValueError,"configuration differs: gamma"):
            self.loop.load_checkpoint(self.payload)

    def test_reference_manifest_is_required_and_must_match(self):
        self.payload["metadata"] = {}
        with self.assertRaisesRegex(ValueError,"manifest differs"):
            self.loop.load_checkpoint(self.payload)
        self.payload["metadata"] = {"manifest_sha256":"b"*64}
        with self.assertRaisesRegex(ValueError,"manifest differs"):
            self.loop.load_checkpoint(self.payload)

    def test_missing_loss_definition_is_rejected_before_loading_models(self):
        with self.assertRaisesRegex(ValueError,"loss definition/calibration"):
            self.loop.load_checkpoint(self.payload)

    def test_old_loss_config_and_changed_weights_cannot_resume_silently(self):
        del self.payload["config"]["world_model_loss"]
        with self.assertRaisesRegex(ValueError,"fields differ"):
            self.loop.load_checkpoint(self.payload)
        self.payload["config"] = asdict(self.loop.config)
        self.payload["config"]["world_model_group_weights"] = (2.,1.,1.,1.,1.,1.)
        with self.assertRaisesRegex(ValueError,"configuration differs: world_model_group_weights"):
            self.loop.load_checkpoint(self.payload)

    def test_old_critic_input_configuration_is_rejected_before_parameter_loading(self):
        self.payload["network_config"]["critic_root_error"] = False
        with self.assertRaisesRegex(ValueError,"network configuration differs"):
            self.loop.load_checkpoint(self.payload)
