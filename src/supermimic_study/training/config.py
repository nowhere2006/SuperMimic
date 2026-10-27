"""Small CPU execution profile, with explicit deviations from Table A.5."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 0
    device: str = "cpu"
    num_envs: int = 2
    iterations: int = 2
    warmup_steps: int = 64
    collect_steps: int = 16
    replay_capacity: int = 256
    world_model_window: int = 8
    world_model_batches: int = 2
    world_model_batch_size: int = 8
    world_model_loss: str = "supertrack_l1"
    world_model_group_weights: tuple[float, ...] = (1., 1., 1., 1., 1., 1.)
    world_model_scale_floor: float = 1e-6
    world_model_termination_weight: float = 1.
    imagination_horizon: int = 32
    imagination_batch_size: int = 2
    critic_epochs: int = 5
    critic_minibatches: int = 4
    learning_rate: float = 2e-4
    gamma: float = .99
    gae_lambda: float = .95
    action_noise_std: float = .1
    action_semantics: str = "clean-policy_executed-dynamics-v2"
    sensor_noise: bool = True
    domain_randomization: bool = True
    episode_seconds: float = 2.
    max_seconds: float = 600.
    solver_iterations: int = 10
    solver_ls_iterations: int = 50
    lr_decay_iterations: int = 250000
    minimum_lr_ratio: float = .03
    clip_names: tuple[str, ...] = ("walk1_subject1", "dance1_subject1")

    def __post_init__(self):
        for name in ("num_envs","iterations","warmup_steps","collect_steps","replay_capacity",
                     "world_model_window","world_model_batches","world_model_batch_size",
                     "imagination_horizon","imagination_batch_size","critic_epochs","critic_minibatches",
                     "solver_iterations","solver_ls_iterations","lr_decay_iterations"):
            if getattr(self,name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.replay_capacity < self.world_model_window:
            raise ValueError("replay capacity must hold a world-model window")
        if self.action_semantics != "clean-policy_executed-dynamics-v2":
            raise ValueError("unsupported action semantics")
        if self.world_model_loss not in ("supertrack_l1", "raw_mse"):
            raise ValueError("unknown world-model loss")
        if len(self.world_model_group_weights) != 6 or any(
                not math.isfinite(w) or w <= 0 for w in self.world_model_group_weights):
            raise ValueError("world-model group weights must contain six finite positive values")
        if not math.isfinite(self.world_model_scale_floor) or self.world_model_scale_floor <= 0:
            raise ValueError("world-model scale floor must be finite and positive")
        if not math.isfinite(self.world_model_termination_weight) or self.world_model_termination_weight <= 0:
            raise ValueError("world-model termination weight must be finite and positive")
        if not (0 < self.gamma < 1 and 0 <= self.gae_lambda <= 1):
            raise ValueError("invalid discount or GAE lambda")
        if self.learning_rate <= 0 or self.action_noise_std < 0 or self.max_seconds <= 0:
            raise ValueError("invalid learning rate, noise, or deadline")
        if not 0 < self.minimum_lr_ratio <= 1:
            raise ValueError("minimum learning-rate ratio must be in (0,1]")
