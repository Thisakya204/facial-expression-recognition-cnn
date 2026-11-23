"""
Models module for Facial Expression Recognition (FER).
"""
from .baseline_cnn import BaselineCNN, build_baseline_cnn

__all__ = [
    "BaselineCNN",
    "build_baseline_cnn",
]
