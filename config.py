"""
Configuration settings for Mutual Information Analysis in Backdoor Detection.

This module contains all hyperparameters and configuration settings used in the
mutual information analysis for backdoor detection in neural networks.
"""

from dataclasses import dataclass
from typing import List, Union

@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    batch_size: int = 256
    learning_rate: float = 0.01
    epochs: int = 60
    num_workers: int = 16
    weight_decay: float = 5e-4
    momentum: float = 0.9
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5

@dataclass
class ModelConfig:
    """Model architecture and parameters."""
    model_type: str = 'resnet18'  # or 'vgg16'
    num_classes: int = 10
    noise_std_xt: float = 0.4
    noise_std_ty: float = 0.4
    hidden_dim: int = 128

@dataclass
class MIEstimationConfig:
    """Mutual Information estimation parameters."""
    epochs: int = 350  # for inputs-vs-outputs
    epochs_y: int = 250  # for outputs-vs-Y
    initial_lr: float = 3e-4  # for inputs-vs-outputs
    initial_lr_y: float = 5e-4  # for outputs-vs-Y
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    early_stop_delta: float = 1e-2
    num_negative_samples: int = 512  # for inputs-vs-outputs
    num_negative_samples_y: int = 256  # for outputs-vs-Y

@dataclass
class DataConfig:
    """Data loading and processing parameters."""
    # sampling_datasize: int = 1000
    total_samples: int = 40000
    feature_dim: int = 512
    mi_compute_epochs: List[int] = None

    def __post_init__(self):
        if self.mi_compute_epochs is None:
            self.mi_compute_epochs = [1, 3, 5, 8, 10, 20, 40, 60]

@dataclass
class WandbConfig:
    """Weights & Biases logging configuration."""
    project_name: str = "MI-Analysis-sampleLoader"
    log_metrics: List[str] = None

    def __post_init__(self):
        if self.log_metrics is None:
            self.log_metrics = [
                "train_accuracy",
                "test_accuracy",
                "test_loss",
                "attack_success_rate"
            ]

@dataclass
class Config:
    """Main configuration class combining all sub-configurations."""
    training: TrainingConfig = TrainingConfig()
    model: ModelConfig = ModelConfig()
    mi_estimation: MIEstimationConfig = MIEstimationConfig()
    data: DataConfig = DataConfig()
    wandb: WandbConfig = WandbConfig()
    observe_classes = [0, '0_backdoor', '0_clean', '0_sample', 1, 2, 3]

# Default configuration
default_config = Config() 