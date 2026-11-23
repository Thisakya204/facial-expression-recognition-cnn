"""
Evaluation Metrics, Classification Report, and Confusion Matrix for Facial Expression Recognition.
"""
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)

from ..data.config import CLASSES, NUM_CLASSES, RESULTS_DIR, CONFUSION_MATRIX_DIR, GRAPHS_DIR


class ModelEvaluator:
    """
    Comprehensive evaluation suite for trained FER models.
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        classes: List[str] = CLASSES,
        device: Optional[torch.device] = None,
        criterion: Optional[nn.Module] = None,
    ):
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.test_loader = test_loader
        self.classes = classes
        self.criterion = criterion or nn.CrossEntropyLoss()

        self.results: Dict[str, Any] = {}

    @torch.no_grad()
    def evaluate(self) -> Dict[str, Any]:
        """
        Runs full evaluation pass on test dataset.
        Computes predictions, loss, accuracy, and detailed classification metrics.
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []

        for images, labels in self.test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=-1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        total = len(all_targets)
        test_loss = running_loss / total if total > 0 else 0.0
        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)

        acc = accuracy_score(y_true, y_pred) * 100.0

        # Detailed classification report
        cls_rep_dict = classification_report(
            y_true,
            y_pred,
            target_names=self.classes,
            output_dict=True,
            zero_division=0,
        )

        cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(self.classes)))

        # Per-class metrics dataframe
        rows = []
        for cls_name in self.classes:
            m = cls_rep_dict.get(cls_name, {})
            rows.append({
                "Emotion": cls_name,
                "Precision": round(m.get("precision", 0.0) * 100, 2),
                "Recall": round(m.get("recall", 0.0) * 100, 2),
                "F1-Score": round(m.get("f1-score", 0.0) * 100, 2),
                "Support": int(m.get("support", 0)),
            })
        metrics_df = pd.DataFrame(rows)

        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        self.results = {
            "test_loss": round(test_loss, 4),
            "test_accuracy": round(acc, 2),
            "macro_precision": round(macro_p * 100, 2),
            "macro_recall": round(macro_r * 100, 2),
            "macro_f1": round(macro_f1 * 100, 2),
            "weighted_precision": round(weighted_p * 100, 2),
            "weighted_recall": round(weighted_r * 100, 2),
            "weighted_f1": round(weighted_f1 * 100, 2),
            "y_true": y_true,
            "y_pred": y_pred,
            "confusion_matrix": cm,
            "per_class_table": metrics_df,
            "full_classification_report": cls_rep_dict,
        }

        return self.results

    def plot_confusion_matrix(
        self,
        normalize: bool = True,
        save_path: Optional[Path] = None,
        cmap: str = "Blues",
    ) -> Path:
        """
        Plots confusion matrix with both raw counts and percentage annotations.
        """
        if not self.results:
            self.evaluate()

        if save_path is None:
            save_path = CONFUSION_MATRIX_DIR / "confusion_matrix.png"

        cm = self.results["confusion_matrix"]
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

        # Create dual-annotation labels (e.g. "82%\n(785)")
        annot_matrix = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                pct = cm_norm[i, j] * 100.0
                count = cm[i, j]
                annot_matrix[i, j] = f"{pct:.1f}%\n({count})"

        plt.figure(figsize=(9, 7))
        sns.heatmap(
            cm_norm if normalize else cm,
            annot=annot_matrix,
            fmt="",
            cmap=cmap,
            xticklabels=self.classes,
            yticklabels=self.classes,
            cbar=True,
            annot_kws={"size": 9, "weight": "semibold"},
        )

        plt.title(
            f"Confusion Matrix (Test Accuracy: {self.results['test_accuracy']}%)",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        plt.xlabel("Predicted Emotion", fontsize=11, fontweight="bold")
        plt.ylabel("True Emotion", fontsize=11, fontweight="bold")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        plt.savefig(save_path, dpi=250, bbox_inches="tight")
        plt.close()

        print(f"Saved confusion matrix plot to: {save_path}")
        return save_path

    def plot_per_class_f1(self, save_path: Optional[Path] = None) -> Path:
        """
        Plots a bar chart comparing F1-scores across each emotion class.
        """
        if not self.results:
            self.evaluate()

        if save_path is None:
            save_path = GRAPHS_DIR / "per_class_f1.png"

        df = self.results["per_class_table"]
        plt.figure(figsize=(9, 5))
        sns.set_theme(style="whitegrid")

        palette = sns.color_palette("mako", len(self.classes))
        ax = sns.barplot(data=df, x="Emotion", y="F1-Score", palette=palette)

        for p in ax.patches:
            height = p.get_height()
            ax.annotate(
                f"{height:.1f}%",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                xytext=(0, 3),
                textcoords="offset points",
            )

        plt.axhline(
            self.results["macro_f1"],
            color="red",
            linestyle="--",
            label=f"Macro Avg F1: {self.results['macro_f1']:.1f}%",
        )
        plt.title("Per-Class F1-Score Performance", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Emotion Class", fontsize=11, fontweight="bold")
        plt.ylabel("F1-Score (%)", fontsize=11, fontweight="bold")
        plt.ylim(0, 100)
        plt.legend(loc="upper right", frameon=True)
        plt.tight_layout()

        plt.savefig(save_path, dpi=250, bbox_inches="tight")
        plt.close()

        print(f"Saved per-class F1 plot to: {save_path}")
        return save_path

    def save_summary_json(self, output_path: Optional[Path] = None) -> Path:
        """Saves evaluation report to JSON file."""
        if not self.results:
            self.evaluate()

        if output_path is None:
            output_path = RESULTS_DIR / "evaluation_report.json"

        serializable = {
            "test_loss": self.results["test_loss"],
            "test_accuracy": self.results["test_accuracy"],
            "macro_precision": self.results["macro_precision"],
            "macro_recall": self.results["macro_recall"],
            "macro_f1": self.results["macro_f1"],
            "weighted_precision": self.results["weighted_precision"],
            "weighted_recall": self.results["weighted_recall"],
            "weighted_f1": self.results["weighted_f1"],
            "per_class_metrics": self.results["per_class_table"].to_dict(orient="records"),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=4)

        print(f"Saved evaluation report to: {output_path}")
        return output_path

    def print_report(self) -> None:
        """Prints a clean, formatted evaluation report to console."""
        if not self.results:
            self.evaluate()

        print("\n" + "=" * 70)
        print("          FACIAL EXPRESSION RECOGNITION - EVALUATION REPORT")
        print("=" * 70)
        print(f" Test Loss            : {self.results['test_loss']:.4f}")
        print(f" Test Accuracy        : {self.results['test_accuracy']:.2f}%")
        print(f" Macro Precision      : {self.results['macro_precision']:.2f}%")
        print(f" Macro Recall         : {self.results['macro_recall']:.2f}%")
        print(f" Macro F1-Score       : {self.results['macro_f1']:.2f}%")
        print(f" Weighted F1-Score    : {self.results['weighted_f1']:.2f}%")
        print("-" * 70)
        print(" Per-Class Breakdown:")
        print(self.results["per_class_table"].to_string(index=False))
        print("=" * 70 + "\n")
