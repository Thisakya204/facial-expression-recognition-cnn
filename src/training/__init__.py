"""
Training module for Facial Expression Recognition (FER).
"""
from .trainer import ModelTrainer, EarlyStopping
from .train import train_baseline
from .experiments import RegularizationExperimentSuite
from .train_final import train_final_model

__all__ = [
    "ModelTrainer",
    "EarlyStopping",
    "train_baseline",
    "RegularizationExperimentSuite",
    "train_final_model",
]
