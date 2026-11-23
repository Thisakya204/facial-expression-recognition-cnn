"""
Training module for Facial Expression Recognition (FER).
"""
from .trainer import ModelTrainer, EarlyStopping
from .train import train_baseline
from .experiments import RegularizationExperimentSuite

__all__ = [
    "ModelTrainer",
    "EarlyStopping",
    "train_baseline",
    "RegularizationExperimentSuite",
]
