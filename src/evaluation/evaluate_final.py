"""
Final Model Evaluation Script on FER-2013 Test Partition.
Loads best trained model checkpoint, computes detailed test metrics,
and saves the final Confusion Matrix and Per-Class F1 plots.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)

from src.data.config import CLASSES, NUM_CLASSES, RESULTS_DIR, GRAPHS_DIR, CONFUSION_MATRIX_DIR
from src.data.dataset import FERDataset
from src.models.regularized_cnn import FullRegularizedCNN


def run_final_evaluation(
    checkpoint_path: str = "models/final_model_best.pth",
    batch_size: int = 64,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("        FINAL MODEL EVALUATION ON FER-2013 TEST SET")
    print("=" * 70)
    print(f"Device: {device} | Checkpoint: {checkpoint_path}")

    # 1. Load test data into memory
    print("\n[1/4] Loading test dataset...")
    ds = FERDataset(validation_split=0.2)
    X_te, y_te = ds.load_split_to_memory("test", grayscale=True)
    test_x = torch.from_numpy(X_te).permute(0, 3, 1, 2).float()
    test_y = torch.from_numpy(y_te).long()
    print(f"Loaded {len(test_x)} test samples.")

    # 2. Load model & weights
    print("\n[2/4] Loading FullRegularizedCNN model weights...")
    model = FullRegularizedCNN(in_channels=1, num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    epoch_info = ckpt.get("epoch", "N/A")
    val_acc_info = ckpt.get("val_acc", 0.0)
    print(f"Successfully loaded checkpoint (Epoch: {epoch_info}, Best Val Acc: {val_acc_info:.2f}%)")

    # 3. Test Evaluation
    print("\n[3/4] Evaluating test set predictions...")
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for i in range(0, len(test_x), batch_size):
            bx = test_x[i : i + batch_size].to(device)
            out = model(bx)
            _, preds = torch.max(out, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(test_y[i : i + batch_size].numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    test_acc = accuracy_score(y_true, y_pred) * 100.0

    cls_report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASSES)))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 75)
    print("           FINAL MODEL - TEST SET CLASSIFICATION REPORT")
    print("=" * 75)
    print(f" Final Test Accuracy  : {test_acc:.2f}%")
    print(f" Macro Avg F1-Score   : {macro_f1 * 100:.2f}%")
    print(f" Weighted Avg F1-Score: {weighted_f1 * 100:.2f}%")
    print("-" * 75)
    print(f"{'Emotion':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
    print("-" * 75)
    for c in CLASSES:
        m = cls_report[c]
        print(f"{c:<12} {m['precision']*100:6.2f}%      {m['recall']*100:6.2f}%      {m['f1-score']*100:6.2f}%      {int(m['support']):<8}")
    print("=" * 75)

    # 4. Generate Visual Artifacts
    print("\n[4/4] Saving visual artifacts and evaluation metrics...")
    # Confusion Matrix
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm_norm[i, j]*100:.1f}%\n({cm[i, j]})"

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        cbar=True,
        annot_kws={"size": 9, "weight": "semibold"},
    )
    plt.title(f"Final Model - Confusion Matrix (Test Acc: {test_acc:.2f}%)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Emotion", fontsize=11, fontweight="bold")
    plt.ylabel("True Emotion", fontsize=11, fontweight="bold")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = CONFUSION_MATRIX_DIR / "final_confusion_matrix.png"
    plt.savefig(cm_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved confusion matrix to: {cm_path}")

    # Per-Class F1-Score
    f1_scores = [cls_report[c]["f1-score"] * 100 for c in CLASSES]
    plt.figure(figsize=(9, 5))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x=CLASSES, y=f1_scores, palette="viridis")
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2.0, h), ha="center", va="bottom", fontsize=9, fontweight="bold", xytext=(0, 3), textcoords="offset points")
    plt.axhline(macro_f1 * 100, color="red", linestyle="--", label=f"Macro Avg F1: {macro_f1 * 100:.1f}%")
    plt.title("Final Model - Per-Class F1-Score Performance", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Emotion Class", fontsize=11, fontweight="bold")
    plt.ylabel("F1-Score (%)", fontsize=11, fontweight="bold")
    plt.ylim(0, 100)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    f1_path = GRAPHS_DIR / "final_per_class_f1.png"
    plt.savefig(f1_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved per-class F1 plot to: {f1_path}")

    # Save JSON summary
    summary_data = {
        "model_name": "FullRegularizedCNN",
        "checkpoint": checkpoint_path,
        "best_epoch": epoch_info,
        "best_val_acc": round(val_acc_info, 2),
        "test_accuracy": round(test_acc, 2),
        "macro_precision": round(macro_p * 100, 2),
        "macro_recall": round(macro_r * 100, 2),
        "macro_f1": round(macro_f1 * 100, 2),
        "weighted_f1": round(weighted_f1 * 100, 2),
        "per_class_metrics": {c: {k: round(v, 4) for k, v in cls_report[c].items()} for c in CLASSES},
    }
    json_path = RESULTS_DIR / "final_model_evaluation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)
    print(f"[+] Saved evaluation report to: {json_path}")
    print("\nEvaluation successfully completed!")
    return summary_data


if __name__ == "__main__":
    run_final_evaluation()
