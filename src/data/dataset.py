"""
Dataset Loading Pipeline and Batch Generators for Facial Expression Recognition.
"""
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Generator, Union
import numpy as np
from sklearn.model_selection import train_test_split

from .config import (
    CLASSES,
    CLASS_TO_IDX,
    TRAIN_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    VALIDATION_SPLIT,
    RANDOM_SEED,
)
from .preprocessing import load_and_preprocess_image, DataAugmentation


class FERBatchGenerator:
    """
    Lightweight, high-performance batch generator with on-the-fly augmentation,
    shuffling, and memory optimization.
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        batch_size: int = BATCH_SIZE,
        shuffle: bool = True,
        augment: bool = False,
        grayscale: bool = True,
        normalize: bool = True,
        norm_method: str = "minmax",
        enhance: bool = False,
        random_seed: Optional[int] = RANDOM_SEED,
    ):
        self.image_paths = np.array(image_paths)
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.grayscale = grayscale
        self.normalize = normalize
        self.norm_method = norm_method
        self.enhance = enhance
        self.num_samples = len(self.image_paths)
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        self.indices = np.arange(self.num_samples)

        self.rng = np.random.default_rng(random_seed)
        self.augmenter = DataAugmentation(random_seed=random_seed) if augment else None

        if self.shuffle:
            self.rng.shuffle(self.indices)

    def __len__(self) -> int:
        return self.num_batches

    def on_epoch_end(self) -> None:
        """Called at the end of each epoch to reshuffle indices."""
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def get_batch(self, batch_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Fetches and processes images for a specific batch index."""
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        batch_indices = self.indices[start_idx:end_idx]

        batch_paths = self.image_paths[batch_indices]
        batch_y = self.labels[batch_indices]

        batch_images = []
        for path in batch_paths:
            img = load_and_preprocess_image(
                image_path=path,
                target_size=IMAGE_SIZE,
                grayscale=self.grayscale,
                normalize=self.normalize,
                norm_method=self.norm_method,
                enhance=self.enhance,
            )
            if self.augment and self.augmenter is not None:
                img = self.augmenter.apply(img)
            batch_images.append(img)

        batch_X = np.stack(batch_images, axis=0)
        return batch_X, batch_y

    def __iter__(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        for i in range(self.num_batches):
            yield self.get_batch(i)
        self.on_epoch_end()


class FERDataset:
    """
    Main Dataset manager for FER-2013 facial expression data.
    Provides fast sample discovery, stratified train/validation splitting,
    in-memory array loading, and batch generation.
    """

    def __init__(
        self,
        train_dir: Union[str, Path] = TRAIN_DIR,
        test_dir: Union[str, Path] = TEST_DIR,
        classes: List[str] = CLASSES,
        validation_split: float = VALIDATION_SPLIT,
        random_seed: int = RANDOM_SEED,
    ):
        self.train_dir = Path(train_dir)
        self.test_dir = Path(test_dir)
        self.classes = classes
        self.validation_split = validation_split
        self.random_seed = random_seed

        # Discovered paths and labels
        self.train_paths: List[str] = []
        self.train_labels: List[int] = []
        self.val_paths: List[str] = []
        self.val_labels: List[int] = []
        self.test_paths: List[str] = []
        self.test_labels: List[int] = []

        self._discover_and_split()

    def _discover_split(self, root_dir: Path) -> Tuple[List[str], List[int]]:
        """Discovers all image files and converts directory names to integer labels."""
        paths = []
        labels = []
        for cls_name in self.classes:
            cls_dir = root_dir / cls_name
            if not cls_dir.exists():
                continue
            idx = CLASS_TO_IDX[cls_name]
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    paths.append(str(cls_dir / fname))
                    labels.append(idx)
        return paths, labels

    def _discover_and_split(self) -> None:
        """Scans dataset and applies stratified train/val split."""
        raw_train_paths, raw_train_labels = self._discover_split(self.train_dir)
        self.test_paths, self.test_labels = self._discover_split(self.test_dir)

        if self.validation_split > 0 and len(raw_train_paths) > 0:
            # Stratified split to ensure balanced class distributions in train and val sets
            (
                self.train_paths,
                self.val_paths,
                self.train_labels,
                self.val_labels,
            ) = train_test_split(
                raw_train_paths,
                raw_train_labels,
                test_size=self.validation_split,
                stratify=raw_train_labels,
                random_state=self.random_seed,
            )
        else:
            self.train_paths = raw_train_paths
            self.train_labels = raw_train_labels
            self.val_paths = []
            self.val_labels = []

    def get_split_sizes(self) -> Dict[str, int]:
        """Returns the number of samples in each split."""
        return {
            "train": len(self.train_paths),
            "val": len(self.val_paths),
            "test": len(self.test_paths),
            "total": len(self.train_paths) + len(self.val_paths) + len(self.test_paths),
        }

    def load_split_to_memory(
        self,
        split: str = "train",
        grayscale: bool = True,
        normalize: bool = True,
        norm_method: str = "minmax",
        enhance: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads an entire split directly into NumPy arrays (X, y).
        Because FER-2013 images are 48x48, loading into RAM is fast (~60MB)
        and eliminates I/O bottlenecks during training.

        Args:
            split: 'train', 'val', or 'test'.
            grayscale: 1-channel if True, 3-channel if False.
            normalize: Normalize to [0, 1] if True.
            norm_method: 'minmax', 'centered', 'standard'.
            enhance: Apply histogram equalization if True.

        Returns:
            (X, y): X is array of shape (N, 48, 48, C), y is int array (N,)
        """
        if split == "train":
            paths, labels = self.train_paths, self.train_labels
        elif split == "val":
            paths, labels = self.val_paths, self.val_labels
        elif split == "test":
            paths, labels = self.test_paths, self.test_labels
        else:
            raise ValueError(f"Unknown split: '{split}'. Choose from 'train', 'val', 'test'.")

        num_samples = len(paths)
        channels = 1 if grayscale else 3
        X = np.empty((num_samples, IMAGE_SIZE[1], IMAGE_SIZE[0], channels), dtype=np.float32)
        y = np.array(labels, dtype=np.int32)

        for i, p in enumerate(paths):
            X[i] = load_and_preprocess_image(
                image_path=p,
                target_size=IMAGE_SIZE,
                grayscale=grayscale,
                normalize=normalize,
                norm_method=norm_method,
                enhance=enhance,
            )

        return X, y

    def load_all_to_memory(
        self,
        grayscale: bool = True,
        normalize: bool = True,
        norm_method: str = "minmax",
        enhance: bool = False,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Convenience method to load train, val, and test splits into memory all at once.

        Returns:
            ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        """
        train_data = self.load_split_to_memory("train", grayscale, normalize, norm_method, enhance)
        val_data = self.load_split_to_memory("val", grayscale, normalize, norm_method, enhance)
        test_data = self.load_split_to_memory("test", grayscale, normalize, norm_method, enhance)
        return train_data, val_data, test_data

    def get_generator(
        self,
        split: str = "train",
        batch_size: int = BATCH_SIZE,
        augment: bool = False,
        shuffle: Optional[bool] = None,
        grayscale: bool = True,
        normalize: bool = True,
        norm_method: str = "minmax",
        enhance: bool = False,
    ) -> FERBatchGenerator:
        """
        Returns a batch generator for memory-efficient training/testing.
        """
        if split == "train":
            paths, labels = self.train_paths, self.train_labels
            if shuffle is None:
                shuffle = True
        elif split == "val":
            paths, labels = self.val_paths, self.val_labels
            if shuffle is None:
                shuffle = False
        elif split == "test":
            paths, labels = self.test_paths, self.test_labels
            if shuffle is None:
                shuffle = False
        else:
            raise ValueError(f"Unknown split: '{split}'. Choose from 'train', 'val', 'test'.")

        return FERBatchGenerator(
            image_paths=paths,
            labels=labels,
            batch_size=batch_size,
            shuffle=shuffle,
            augment=augment,
            grayscale=grayscale,
            normalize=normalize,
            norm_method=norm_method,
            enhance=enhance,
            random_seed=self.random_seed,
        )


def load_data(
    in_memory: bool = True,
    validation_split: float = VALIDATION_SPLIT,
    batch_size: int = BATCH_SIZE,
    augment_train: bool = True,
    grayscale: bool = True,
) -> Union[Tuple[Tuple[np.ndarray, np.ndarray], ...], Dict[str, FERBatchGenerator]]:
    """
    High-level entry point to load FER data either as in-memory NumPy arrays
    or as batch generators.

    Args:
        in_memory: If True, returns loaded NumPy arrays; if False, returns batch generators.
        validation_split: Fraction of train data reserved for validation (e.g. 0.2).
        batch_size: Batch size for generators.
        augment_train: Apply augmentation to training set.
        grayscale: Grayscale (1-channel) or RGB (3-channels).

    Returns:
        If in_memory: ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        If not in_memory: dict containing 'train', 'val', 'test' FERBatchGenerator objects.
    """
    dataset = FERDataset(validation_split=validation_split)

    if in_memory:
        return dataset.load_all_to_memory(grayscale=grayscale)
    else:
        return {
            "train": dataset.get_generator("train", batch_size=batch_size, augment=augment_train, grayscale=grayscale),
            "val": dataset.get_generator("val", batch_size=batch_size, augment=False, grayscale=grayscale),
            "test": dataset.get_generator("test", batch_size=batch_size, augment=False, grayscale=grayscale),
        }


if __name__ == "__main__":
    print("Testing FERDataset pipeline...")
    ds = FERDataset(validation_split=0.2)
    print("Split sizes:", ds.get_split_sizes())

    gen = ds.get_generator("train", batch_size=16, augment=True)
    batch_x, batch_y = gen.get_batch(0)
    print(f"Sample Batch X shape: {batch_x.shape}, dtype: {batch_x.dtype}, range: [{batch_x.min():.2f}, {batch_x.max():.2f}]")
    print(f"Sample Batch y shape: {batch_y.shape}, labels: {batch_y}")
    print("FERDataset pipeline verification complete!")
