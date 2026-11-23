"""
Model Training Pipeline and Early Stopping for Facial Expression Recognition.
"""
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..data.config import PROJECT_ROOT, GRAPHS_DIR, RESULTS_DIR


class EarlyStopping:
    """
    Early stops the training if validation metric does not improve after a given patience.
    """

    def __init__(
        self,
        patience: int = 6,
        min_delta: float = 1e-4,
        mode: str = "min",
        verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.best_epoch = 0

    def step(self, current_metric: float, epoch: int) -> bool:
        """
        Returns True if the current model is the best so far, False otherwise.
        Sets self.early_stop to True if patience is exceeded.
        """
        score = -current_metric if self.mode == "min" else current_metric

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return True

        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  --> EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return True


class ModelTrainer:
    """
    Complete Training Pipeline handling:
    - GPU/CPU device management
    - Balanced class weighting in CrossEntropyLoss
    - Learning rate scheduling
    - Early stopping & Best model checkpointing
    - Metric logging & Training history plotting
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        class_weights: Optional[torch.Tensor] = None,
        checkpoint_dir: Optional[Path] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        if criterion is not None:
            self.criterion = criterion
        elif class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optimizer or optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.scheduler = scheduler or optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        )

        self.checkpoint_dir = Path(checkpoint_dir or (PROJECT_ROOT / "models"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rates": [],
            "epoch_times": [],
        }

    def train_epoch(self) -> Tuple[float, float]:
        """Runs a single training epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100.0
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Runs validation without gradients."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        val_loss = running_loss / total
        val_acc = (correct / total) * 100.0
        return val_loss, val_acc

    def fit(
        self,
        epochs: int = 15,
        early_stopping_patience: int = 5,
        model_name: str = "baseline_cnn",
    ) -> Dict[str, List[float]]:
        """
        Executes complete training loop across specified epochs.
        """
        early_stopping = EarlyStopping(patience=early_stopping_patience, mode="min")
        best_model_path = self.checkpoint_dir / f"{model_name}_best.pth"
        last_model_path = self.checkpoint_dir / f"{model_name}_last.pth"

        print("\n" + "=" * 78)
        print(f"  STARTING MODEL TRAINING: {model_name.upper()} ({epochs} Epochs)")
        print(f"  Device: {self.device} | Train Batches: {len(self.train_loader)} | Val Batches: {len(self.val_loader)}")
        print("=" * 78)

        total_start = time.time()
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            current_lr = self.optimizer.param_groups[0]["lr"]
            epoch_duration = time.time() - epoch_start

            # Update scheduler
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()

            # Record history
            self.history["train_loss"].append(round(train_loss, 4))
            self.history["train_acc"].append(round(train_acc, 2))
            self.history["val_loss"].append(round(val_loss, 4))
            self.history["val_acc"].append(round(val_acc, 2))
            self.history["learning_rates"].append(current_lr)
            self.history["epoch_times"].append(round(epoch_duration, 2))

            # Early stopping check and model saving
            is_best = early_stopping.step(val_loss, epoch)
            best_tag = "  --> [BEST MODEL SAVED]" if is_best else ""

            if is_best:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }, best_model_path)

            # Always save latest checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, last_model_path)

            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                f"LR: {current_lr:.6f} ({epoch_duration:.1f}s){best_tag}"
            )

            if early_stopping.early_stop:
                print(f"\n[!] Early stopping triggered at epoch {epoch}! Best epoch was {early_stopping.best_epoch}.")
                break

        total_time = time.time() - total_start
        print("=" * 78)
        print(f"Training completed in {total_time:.1f}s. Best weights saved to: {best_model_path}")
        print("=" * 78 + "\n")

        # Load best weights before returning
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded best checkpoint (Epoch {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']:.2f}%)")

        self.save_history()
        self.plot_history()
        return self.history

    def save_history(self, path: Optional[Path] = None) -> Path:
        """Saves epoch history to JSON."""
        if path is None:
            path = RESULTS_DIR / "training_history.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=4)
        print(f"Saved training history log to: {path}")
        return path

    def plot_history(self, save_path: Optional[Path] = None) -> Path:
        """
        Plots training and validation loss & accuracy curves.
        """
        if save_path is None:
            save_path = GRAPHS_DIR / "training_history.png"

        epochs = range(1, len(self.history["train_loss"]) + 1)
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss curve
        axes[0].plot(epochs, self.history["train_loss"], "o-", label="Train Loss", color="#2980b9", lw=2)
        axes[0].plot(epochs, self.history["val_loss"], "s--", label="Val Loss", color="#e74c3c", lw=2)
        axes[0].set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Epoch", fontsize=10, fontweight="bold")
        axes[0].set_ylabel("Cross Entropy Loss", fontsize=10, fontweight="bold")
        axes[0].legend(frameon=True)

        # Accuracy curve
        axes[1].plot(epochs, self.history["train_acc"], "o-", label="Train Accuracy", color="#27ae60", lw=2)
        axes[1].plot(epochs, self.history["val_acc"], "s--", label="Val Accuracy", color="#f39c12", lw=2)
        axes[1].set_title("Training & Validation Accuracy", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Epoch", fontsize=10, fontweight="bold")
        axes[1].set_ylabel("Accuracy (%)", fontsize=10, fontweight="bold")
        axes[1].legend(frameon=True)

        plt.suptitle("Facial Expression Recognition - Model Training Curves", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved training history curves to: {save_path}")
        return save_path
