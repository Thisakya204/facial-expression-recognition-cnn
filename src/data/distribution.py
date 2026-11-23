"""
Class Distribution Analysis and Visualization for Facial Expression Recognition.
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight

from .config import (
    CLASSES,
    TRAIN_DIR,
    TEST_DIR,
    RESULTS_DIR,
    GRAPHS_DIR,
)


class ClassDistributionAnalyzer:
    """
    Analyzes class frequencies, imbalance metrics, and class weights for FER dataset.
    """

    def __init__(
        self,
        train_dir: Path = TRAIN_DIR,
        test_dir: Path = TEST_DIR,
        classes: list = CLASSES,
    ):
        self.train_dir = Path(train_dir)
        self.test_dir = Path(test_dir)
        self.classes = classes
        self.train_counts = {}
        self.test_counts = {}
        self.stats = {}

    def analyze(self) -> Dict[str, Any]:
        """
        Scans train and test folders, computes counts, percentages, and class weights.
        """
        for cls in self.classes:
            tr_cls_dir = self.train_dir / cls
            te_cls_dir = self.test_dir / cls
            self.train_counts[cls] = len(os.listdir(tr_cls_dir)) if tr_cls_dir.exists() else 0
            self.test_counts[cls] = len(os.listdir(te_cls_dir)) if te_cls_dir.exists() else 0

        total_train = sum(self.train_counts.values())
        total_test = sum(self.test_counts.values())
        total_all = total_train + total_test

        # Compute balanced class weights for training: N / (n_classes * count)
        train_labels = []
        for idx, cls in enumerate(self.classes):
            train_labels.extend([idx] * self.train_counts[cls])
        
        train_labels_arr = np.array(train_labels)
        unique_classes = np.arange(len(self.classes))
        raw_weights = compute_class_weight(
            class_weight="balanced",
            classes=unique_classes,
            y=train_labels_arr,
        )
        class_weights = {self.classes[i]: float(np.round(raw_weights[i], 4)) for i in range(len(self.classes))}
        class_weights_dict_idx = {i: float(np.round(raw_weights[i], 4)) for i in range(len(self.classes))}

        # Per-class summary dataframe
        rows = []
        for cls in self.classes:
            tr = self.train_counts[cls]
            te = self.test_counts[cls]
            tot = tr + te
            tr_pct = (tr / total_train * 100.0) if total_train > 0 else 0.0
            te_pct = (te / total_test * 100.0) if total_test > 0 else 0.0
            tot_pct = (tot / total_all * 100.0) if total_all > 0 else 0.0
            rows.append({
                "Emotion": cls,
                "Train Count": tr,
                "Train %": round(tr_pct, 2),
                "Test Count": te,
                "Test %": round(te_pct, 2),
                "Total Count": tot,
                "Total %": round(tot_pct, 2),
                "Class Weight": class_weights[cls],
            })

        df = pd.DataFrame(rows)

        # Imbalance ratio
        max_tr = max(self.train_counts.values()) if self.train_counts else 0
        min_tr = min(self.train_counts.values()) if self.train_counts else 0
        imbalance_ratio = round(max_tr / min_tr, 2) if min_tr > 0 else None

        self.stats = {
            "total_train": total_train,
            "total_test": total_test,
            "total_all": total_all,
            "train_counts": self.train_counts,
            "test_counts": self.test_counts,
            "class_weights_by_name": class_weights,
            "class_weights_by_idx": class_weights_dict_idx,
            "imbalance_ratio": imbalance_ratio,
            "summary_table": df,
        }

        return self.stats

    def save_summary_json(self, output_path: Optional[Path] = None) -> Path:
        """Saves distribution summary and class weights to a JSON file."""
        if not self.stats:
            self.analyze()

        if output_path is None:
            output_path = RESULTS_DIR / "class_distribution_summary.json"

        serializable_stats = {
            "total_train": self.stats["total_train"],
            "total_test": self.stats["total_test"],
            "total_all": self.stats["total_all"],
            "imbalance_ratio": self.stats["imbalance_ratio"],
            "train_counts": self.stats["train_counts"],
            "test_counts": self.stats["test_counts"],
            "class_weights_by_name": self.stats["class_weights_by_name"],
            "class_weights_by_idx": self.stats["class_weights_by_idx"],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_stats, f, indent=4)

        print(f"Saved distribution summary to: {output_path}")
        return output_path

    def plot_distribution(
        self,
        save_path: Optional[Path] = None,
        palette: str = "viridis",
    ) -> Path:
        """
        Creates publication-quality visualizations of class distribution and saves plot.
        """
        if not self.stats:
            self.analyze()

        if save_path is None:
            save_path = GRAPHS_DIR / "class_distribution.png"

        df = self.stats["summary_table"]

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Subplot 1: Train vs Test counts grouped bar chart
        df_melt = pd.melt(
            df,
            id_vars=["Emotion"],
            value_vars=["Train Count", "Test Count"],
            var_name="Dataset Split",
            value_name="Number of Images",
        )

        ax1 = axes[0]
        bar = sns.barplot(
            data=df_melt,
            x="Emotion",
            y="Number of Images",
            hue="Dataset Split",
            palette=["#3498db", "#e74c3c"],
            ax=ax1,
        )

        # Add values on top of bars
        for p in bar.patches:
            height = p.get_height()
            if height > 0:
                ax1.annotate(
                    f"{int(height)}",
                    (p.get_x() + p.get_width() / 2.0, height),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    xytext=(0, 3),
                    textcoords="offset points",
                    fontweight="semibold",
                )

        ax1.set_title("Sample Count per Emotion Class (Train vs Test)", fontsize=13, fontweight="bold", pad=12)
        ax1.set_xlabel("Emotion Class", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Number of Samples", fontsize=11, fontweight="bold")
        ax1.legend(title="Split", frameon=True)

        # Subplot 2: Percentage share with class weights indicator
        ax2 = axes[1]
        colors = sns.color_palette(palette, len(self.classes))
        wedges, texts, autotexts = ax2.pie(
            df["Train Count"],
            labels=df["Emotion"],
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            pctdistance=0.75,
            explode=[0.08 if c == "disgust" else 0.02 for c in df["Emotion"]],
            textprops={"fontsize": 10},
        )
        for autotext in autotexts:
            autotext.set_color("black")
            autotext.set_fontsize(9)
            autotext.set_weight("bold")

        ax2.set_title(
            f"Training Set Class Distribution (Imbalance Ratio: {self.stats['imbalance_ratio']}x)",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=250, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved distribution plot to: {save_path}")
        return save_path

    def print_summary(self) -> None:
        """Prints a formatted summary table to console."""
        if not self.stats:
            self.analyze()

        print("=" * 80)
        print("          FACIAL EXPRESSION RECOGNITION - CLASS DISTRIBUTION")
        print("=" * 80)
        print(self.stats["summary_table"].to_string(index=False))
        print("-" * 80)
        print(f"Total Train Samples: {self.stats['total_train']}")
        print(f"Total Test Samples : {self.stats['total_test']}")
        print(f"Total All Samples  : {self.stats['total_all']}")
        print(f"Imbalance Ratio    : {self.stats['imbalance_ratio']}x (Happy vs Disgust)")
        print("\nRecommended Balanced Class Weights (for loss function):")
        for cls, w in self.stats["class_weights_by_name"].items():
            print(f"  - {cls:<10}: {w:.4f}")
        print("=" * 80)


def analyze_class_distribution() -> Dict[str, Any]:
    """Helper function to run full analysis, plot, and return results."""
    analyzer = ClassDistributionAnalyzer()
    stats = analyzer.analyze()
    analyzer.save_summary_json()
    analyzer.plot_distribution()
    analyzer.print_summary()
    return stats


if __name__ == "__main__":
    analyze_class_distribution()
