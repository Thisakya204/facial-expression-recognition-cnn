"""
Configuration settings for Facial Expression Recognition (FER) dataset and pipeline.
"""
import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

RESULTS_DIR = PROJECT_ROOT / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"
CONFUSION_MATRIX_DIR = RESULTS_DIR / "confusion_matrix"

# Ensure output directories exist
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
CONFUSION_MATRIX_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Classes
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(CLASSES)

CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(CLASSES)}

# Image Specifications
IMAGE_WIDTH = 48
IMAGE_HEIGHT = 48
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
DEFAULT_CHANNELS = 1  # 1 for grayscale, 3 for RGB

# Training Defaults
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42
