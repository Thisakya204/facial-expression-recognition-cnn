"""
Training module for Facial Expression Recognition (FER).
"""
from .trainer import ModelTrainer, EarlyStopping
from .train import train_baseline

__all__ = [
    "ModelTrainer",
    "EarlyStopping",
    "train_baseline",
]
