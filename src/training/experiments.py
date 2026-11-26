"""
Regularization & Ablation Experiments Suite: Dropout, Batch Normalization, and Data Augmentation.
"""
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import os
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from ..data.config import BATCH_SIZE, GRAPHS_DIR, RESULTS_DIR
from ..data.dataset import FERDataset
from ..data.distribution import ClassDistributionAnalyzer
from ..models.regularized_cnn import (
    StandardCNN,
    DropoutCNN,
    BatchNormCNN,
    FullRegularizedCNN,
)
from .trainer import ModelTrainer
from ..evaluation.evaluate import ModelEvaluator


# Set optimal CPU thread count
num_cores = min(8, os.cpu_count() or 4)
torch.set_num_threads(num_cores)


class FastTensorDataset(Dataset):
    """Memory-resident tensor dataset with fast GPU/CPU transforms."""

    def __init__(self, tensors_x: torch.Tensor, tensors_y: torch.Tensor, transform=None):
        self.tensors_x = tensors_x
        self.tensors_y = tensors_y
        self.transform = transform

    def __len__(self) -> int:
        return self.tensors_x.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.tensors_x[idx]
        y = self.tensors_y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


class RegularizationExperimentSuite:
    """
    Orchestrates comparative regularization experiments:
    1. Baseline (No Regularization: No Dropout, No BatchNorm, No Augmentation)
    2. Dropout Experiment (With Dropout, No BatchNorm, No Augmentation)
    3. Batch Normalization Experiment (With BatchNorm, No Dropout, No Augmentation)
    4. Data Augmentation Experiment (Standard Model, With Data Augmentation)
    5. Full Regularization (BatchNorm + Dropout + Data Augmentation)
    """

    EXPERIMENT_CONFIGS = {
        "Baseline_NoReg": {
            "name": "1. Baseline (No Regularization)",
            "model_builder": lambda: StandardCNN(),
            "use_augmentation": False,
            "description": "No Dropout, No BatchNorm, No Augmentation",
            "color": "#e74c3c",  # Red
        },
        "Dropout_Only": {
            "name": "2. Dropout Only",
            "model_builder": lambda: DropoutCNN(conv_dropout=0.25, dense_dropout=0.5),
            "use_augmentation": False,
            "description": "Dropout (0.25/0.5), No BatchNorm, No Augmentation",
            "color": "#9b59b6",  # Purple
        },
        "BatchNorm_Only": {
            "name": "3. BatchNorm Only",
            "model_builder": lambda: BatchNormCNN(),
            "use_augmentation": False,
            "description": "BatchNorm, No Dropout, No Augmentation",
            "color": "#3498db",  # Blue
        },
        "Augmentation_Only": {
            "name": "4. Augmentation Only",
            "model_builder": lambda: StandardCNN(),
            "use_augmentation": True,
            "description": "Data Augmentation (Flip, Rotate, Zoom), No Dropout/BN",
            "color": "#e67e22",  # Orange
        },
        "Full_Regularization": {
            "name": "5. Full Regularization",
            "model_builder": lambda: FullRegularizedCNN(conv_dropout=0.25, dense_dropout=0.5),
            "use_augmentation": True,
            "description": "BatchNorm + Dropout + Data Augmentation",
            "color": "#27ae60",  # Green
        },
    }

    def __init__(
        self,
        batch_size: int = BATCH_SIZE,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[torch.device] = None,
        use_class_weights: bool = True,
    ):
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.use_class_weights = use_class_weights

        self.results: Dict[str, Any] = {}

        # Preload data into RAM for ultra-fast training (eliminates per-sample disk I/O)
        print("Preloading FER-2013 dataset into system memory for fast execution...")
        t_load = time.time()
        ds = FERDataset(validation_split=0.2)
        (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = ds.load_all_to_memory(grayscale=True)

        train_x = torch.from_numpy(X_tr).permute(0, 3, 1, 2).float()
        train_y = torch.from_numpy(y_tr).long()

        val_x = torch.from_numpy(X_val).permute(0, 3, 1, 2).float()
        val_y = torch.from_numpy(y_val).long()

        test_x = torch.from_numpy(X_te).permute(0, 3, 1, 2).float()
        test_y = torch.from_numpy(y_te).long()

        train_transforms = T.Compose([
            T.RandomCrop(48, padding=4, padding_mode="reflect"),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
        ])

        train_ds_std = FastTensorDataset(train_x, train_y, transform=None)
        train_ds_aug = FastTensorDataset(train_x, train_y, transform=train_transforms)
        val_ds = FastTensorDataset(val_x, val_y, transform=None)
        test_ds = FastTensorDataset(test_x, test_y, transform=None)

        self.standard_loaders = {
            "train": DataLoader(train_ds_std, batch_size=self.batch_size, shuffle=True),
            "val": DataLoader(val_ds, batch_size=self.batch_size, shuffle=False),
            "test": DataLoader(test_ds, batch_size=self.batch_size, shuffle=False),
        }
        self.augmented_loaders = {
            "train": DataLoader(train_ds_aug, batch_size=self.batch_size, shuffle=True),
            "val": DataLoader(val_ds, batch_size=self.batch_size, shuffle=False),
            "test": DataLoader(test_ds, batch_size=self.batch_size, shuffle=False),
        }
        print(f"Dataset cached in RAM in {time.time() - t_load:.1f}s! DataLoaders ready.")

        # Balanced weights tensor
        self.class_weights_tensor = None
        if self.use_class_weights:
            analyzer = ClassDistributionAnalyzer()
            stats = analyzer.analyze()
            raw_weights = [stats["class_weights_by_idx"][i] for i in range(len(stats["class_weights_by_idx"]))]
            self.class_weights_tensor = torch.tensor(raw_weights, dtype=torch.float32).to(self.device)

    def run_single_experiment(
        self,
        exp_key: str,
        epochs: int = 10,
        patience: int = 5,
    ) -> Dict[str, Any]:
        """Runs a single regularization experiment."""
        cfg = self.EXPERIMENT_CONFIGS[exp_key]
        print("\n" + "#" * 78)
        print(f"  RUNNING EXPERIMENT: {cfg['name']}")
        print(f"  Description: {cfg['description']}")
        print("#" * 78)

        # Select data loader based on augmentation flag
        train_loader = (
            self.augmented_loaders["train"]
            if cfg["use_augmentation"]
            else self.standard_loaders["train"]
        )
        val_loader = self.standard_loaders["val"]
        test_loader = self.standard_loaders["test"]

        # Build model
        model = cfg["model_builder"]().to(self.device)
        model.summary()

        # Train
        trainer = ModelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=self.device,
            class_weights=self.class_weights_tensor,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        start_time = time.time()
        history = trainer.fit(
            epochs=epochs,
            early_stopping_patience=patience,
            model_name=f"exp_{exp_key}",
        )
        train_duration = time.time() - start_time

        # Test evaluation
        evaluator = ModelEvaluator(model=model, test_loader=test_loader, device=self.device)
        eval_metrics = evaluator.evaluate()

        final_train_acc = history["train_acc"][-1]
        best_val_acc = max(history["val_acc"])
        final_val_acc = history["val_acc"][-1]
        test_acc = eval_metrics["test_accuracy"]
        overfitting_gap = round(final_train_acc - final_val_acc, 2)

        exp_result = {
            "key": exp_key,
            "name": cfg["name"],
            "description": cfg["description"],
            "color": cfg["color"],
            "epochs_run": len(history["train_loss"]),
            "train_duration_sec": round(train_duration, 1),
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "final_train_acc": final_train_acc,
            "best_val_acc": best_val_acc,
            "final_val_acc": final_val_acc,
            "overfitting_gap": overfitting_gap,
            "test_accuracy": test_acc,
            "test_macro_f1": eval_metrics["macro_f1"],
            "test_weighted_f1": eval_metrics["weighted_f1"],
            "history": history,
        }

        self.results[exp_key] = exp_result
        return exp_result

    def run_all(
        self,
        epochs: int = 10,
        patience: int = 5,
        keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Runs the entire battery of regularization experiments."""
        target_keys = keys or list(self.EXPERIMENT_CONFIGS.keys())

        print("\n" + "=" * 80)
        print("          STARTING REGULARIZATION EXPERIMENTS SUITE")
        print(f"  Target Experiments: {', '.join(target_keys)}")
        print(f"  Epochs per Experiment: {epochs} | Patience: {patience}")
        print("=" * 80)

        for k in target_keys:
            self.run_single_experiment(k, epochs=epochs, patience=patience)

        self.save_summary_json()
        self.plot_all_comparisons()
        self.print_summary_table()
        return self.results

    def get_summary_dataframe(self) -> pd.DataFrame:
        """Returns comparison dataframe of all executed experiments."""
        rows = []
        for k, res in self.results.items():
            rows.append({
                "Experiment": res["name"],
                "Final Train Acc (%)": res["final_train_acc"],
                "Best Val Acc (%)": res["best_val_acc"],
                "Overfitting Gap (%)": res["overfitting_gap"],
                "Test Accuracy (%)": res["test_accuracy"],
                "Test Macro F1 (%)": res["test_macro_f1"],
                "Epochs": res["epochs_run"],
                "Time (s)": res["train_duration_sec"],
            })
        return pd.DataFrame(rows)

    def print_summary_table(self) -> None:
        """Prints formatted summary table to console."""
        df = self.get_summary_dataframe()
        print("\n" + "=" * 85)
        print("           REGULARIZATION EXPERIMENTS - COMPARATIVE SUMMARY")
        print("=" * 85)
        print(df.to_string(index=False))
        print("=" * 85)

    def save_summary_json(self, output_path: Optional[Path] = None) -> Path:
        """Saves experiment results to JSON."""
        if output_path is None:
            output_path = RESULTS_DIR / "regularization_experiments.json"

        # Serialize without numpy types
        serializable = {}
        for k, v in self.results.items():
            serializable[k] = {
                "name": v["name"],
                "description": v["description"],
                "final_train_acc": v["final_train_acc"],
                "best_val_acc": v["best_val_acc"],
                "overfitting_gap": v["overfitting_gap"],
                "test_accuracy": v["test_accuracy"],
                "test_macro_f1": v["test_macro_f1"],
                "epochs_run": v["epochs_run"],
                "train_duration_sec": v["train_duration_sec"],
                "history": v["history"],
            }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=4)

        print(f"Saved regularization experiment results to: {output_path}")
        return output_path

    def plot_all_comparisons(self) -> None:
        """Generates visual comparison figures for loss, accuracy, and overfitting gap."""
        if not self.results:
            return

        sns.set_theme(style="whitegrid")

        # 1. Loss and Accuracy Trajectory Curves Comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for k, res in self.results.items():
            history = res["history"]
            epochs = range(1, len(history["train_loss"]) + 1)
            color = res["color"]
            name = res["name"]

            # Loss plot (Val Loss Solid, Train Loss Dotted)
            axes[0].plot(epochs, history["val_loss"], "-", lw=2, label=f"{name} (Val)", color=color)
            axes[0].plot(epochs, history["train_loss"], ":", lw=1.5, alpha=0.7, color=color)

            # Accuracy plot (Val Acc Solid, Train Acc Dotted)
            axes[1].plot(epochs, history["val_acc"], "-", lw=2, label=f"{name} (Val)", color=color)
            axes[1].plot(epochs, history["train_acc"], ":", lw=1.5, alpha=0.7, color=color)

        axes[0].set_title("Validation Loss Comparison (Dotted = Train Loss)", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Epoch", fontsize=11, fontweight="bold")
        axes[0].set_ylabel("Loss", fontsize=11, fontweight="bold")
        axes[0].legend(fontsize=9, frameon=True)

        axes[1].set_title("Validation Accuracy Comparison (Dotted = Train Acc)", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Epoch", fontsize=11, fontweight="bold")
        axes[1].set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
        axes[1].legend(fontsize=9, frameon=True)

        plt.suptitle("Regularization Experiments: Convergence & Generalization Curves", fontsize=14, fontweight="bold")
        plt.tight_layout()
        curve_path = GRAPHS_DIR / "regularization_convergence_curves.png"
        plt.savefig(curve_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved convergence curves to: {curve_path}")

        # 2. Overfitting Gap & Test Accuracy Comparison Bar Plots
        fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5))
        df = self.get_summary_dataframe()

        # Plot 2A: Overfitting Gap (Train Acc - Val Acc)
        bar1 = sns.barplot(
            data=df,
            x="Experiment",
            y="Overfitting Gap (%)",
            palette="Reds_r",
            ax=axes2[0],
        )
        for p in bar1.patches:
            h = p.get_height()
            axes2[0].annotate(
                f"{h:.1f}%",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                xytext=(0, 3),
                textcoords="offset points",
            )
        axes2[0].set_title("Overfitting Gap: Train Acc - Val Acc (Lower is Better)", fontsize=12, fontweight="bold")
        axes2[0].set_xticklabels(axes2[0].get_xticklabels(), rotation=25, ha="right", fontsize=9)
        axes2[0].set_ylabel("Gap (%)", fontsize=10, fontweight="bold")

        # Plot 2B: Test Accuracy
        bar2 = sns.barplot(
            data=df,
            x="Experiment",
            y="Test Accuracy (%)",
            palette="Greens_d",
            ax=axes2[1],
        )
        for p in bar2.patches:
            h = p.get_height()
            axes2[1].annotate(
                f"{h:.1f}%",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                xytext=(0, 3),
                textcoords="offset points",
            )
        axes2[1].set_title("Final Test Accuracy Comparison (Higher is Better)", fontsize=12, fontweight="bold")
        axes2[1].set_xticklabels(axes2[1].get_xticklabels(), rotation=25, ha="right", fontsize=9)
        axes2[1].set_ylabel("Test Accuracy (%)", fontsize=10, fontweight="bold")

        plt.tight_layout()
        bar_path = GRAPHS_DIR / "regularization_overfitting_comparison.png"
        plt.savefig(bar_path, dpi=200, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved overfitting comparison to: {bar_path}")
