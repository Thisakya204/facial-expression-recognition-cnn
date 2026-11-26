"""
High-Performance Final Model Training and Evaluation Pipeline for Facial Expression Recognition.
Leverages in-memory tensor caching and hardware parallelism for fast execution on CPU.
"""
import os
import sys
import time
import json
import gc
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.config import CLASSES, NUM_CLASSES, RESULTS_DIR, GRAPHS_DIR, CONFUSION_MATRIX_DIR
from src.data.dataset import FERDataset
from src.data.distribution import ClassDistributionAnalyzer
from src.models.regularized_cnn import FullRegularizedCNN


# Set optimal CPU thread count
num_cores = min(4, os.cpu_count() or 4)
torch.set_num_threads(num_cores)


class FastTensorDataset(Dataset):
    """Memory-resident tensor dataset with zero per-sample allocation overhead."""

    def __init__(self, tensors_x: torch.Tensor, tensors_y: torch.Tensor):
        self.tensors_x = tensors_x
        self.tensors_y = tensors_y

    def __len__(self) -> int:
        return self.tensors_x.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.tensors_x[idx], self.tensors_y[idx]


def train_final_model(
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    model_name: str = "final_model",
    fresh_start: bool = True,
) -> Dict[str, Any]:
    """
    Trains the optimal FullRegularizedCNN on FER2013 and produces all evaluation artifacts.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("      FINAL FACIAL EXPRESSION RECOGNITION MODEL TRAINING (FER-2013)")
    print("=" * 78)
    print(f" Device          : {device} (PyTorch CPU Threads: {torch.get_num_threads()})")
    print(f" Target Epochs   : {epochs} | Batch Size: {batch_size} | Base LR: {lr}")
    print(f" Architecture    : FullRegularizedCNN (Conv + BatchNorm + Dropout + Augmentation)")
    print("=" * 78)

    # 1. Load Data into RAM
    t0 = time.time()
    print("\n[1/5] Loading FER-2013 dataset into system memory...")
    ds = FERDataset(validation_split=0.2)
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = ds.load_all_to_memory(grayscale=True)

    # Convert to PyTorch tensors in RAM: (N, H, W, 1) -> (N, 1, 48, 48)
    train_x = torch.from_numpy(X_tr).permute(0, 3, 1, 2).float().contiguous()
    train_y = torch.from_numpy(y_tr).long()

    val_x = torch.from_numpy(X_val).permute(0, 3, 1, 2).float().contiguous()
    val_y = torch.from_numpy(y_val).long()

    test_x = torch.from_numpy(X_te).permute(0, 3, 1, 2).float().contiguous()
    test_y = torch.from_numpy(y_te).long()

    print(f"  --> Train Set : {train_x.shape[0]:,} images")
    print(f"  --> Val Set   : {val_x.shape[0]:,} images")
    print(f"  --> Test Set  : {test_x.shape[0]:,} images")
    print(f"  --> Memory Load Time: {time.time() - t0:.1f}s")

    # 2. Fast Contiguous In-Memory Tensor Loaders
    train_ds = FastTensorDataset(train_x, train_y)
    val_ds = FastTensorDataset(val_x, val_y)
    test_ds = FastTensorDataset(test_x, test_y)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=False)

    # 3. Class Weights for Handling Imbalance
    analyzer = ClassDistributionAnalyzer()
    stats = analyzer.analyze()
    raw_weights = [stats["class_weights_by_idx"][i] for i in range(len(stats["class_weights_by_idx"]))]
    class_weights = torch.tensor(raw_weights, dtype=torch.float32).to(device)
    print(f"[2/5] Applied Balanced Class Weights: {[round(w, 2) for w in raw_weights]}")

    # 4. Instantiate Model
    print("[3/5] Instantiating FullRegularizedCNN...")
    model = FullRegularizedCNN(
        in_channels=1,
        num_classes=NUM_CLASSES,
        conv_dropout=0.25,
        dense_dropout=0.5,
    ).to(device)
    trainable_p, total_p = model.get_num_params()
    print(f"  --> Trainable Parameters: {trainable_p:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)

    # Checkpoint paths
    best_path = models_dir / f"{model_name}_best.pth"
    last_path = models_dir / f"{model_name}_last.pth"

    start_epoch = 1
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    # Resume only if fresh_start is False and last_path exists
    if not fresh_start and last_path.exists():
        try:
            ckpt = torch.load(last_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_loss = ckpt.get("val_loss", float("inf"))
            best_val_acc = ckpt.get("val_acc", 0.0)
            best_epoch = ckpt.get("epoch", 0)
            print(f"Resuming training from Epoch {start_epoch} (Previous Best Val Acc: {best_val_acc:.2f}%)")
        except Exception as e:
            print(f"Starting fresh training: {e}")
    else:
        print("Starting fresh training run from Epoch 1...")

    # 5. Training Loop
    print("\n[4/5] Executing Training Across Epochs...")
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": [], "duration": []
    }

    train_start = time.time()
    for epoch in range(start_epoch, epochs + 1):
        ep_start = time.time()

        # Training pass
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)

            # Vectorized Batch Data Augmentation (Zero heap fragmentation)
            # A. Horizontal Flip (p=0.5)
            if torch.rand(1).item() > 0.5:
                bx = torch.flip(bx, dims=[-1])

            # B. Small spatial translation (+- 2 pixels, p=0.5)
            if torch.rand(1).item() > 0.5:
                shift_x = int(torch.randint(-2, 3, (1,)).item())
                shift_y = int(torch.randint(-2, 3, (1,)).item())
                bx = torch.roll(bx, shifts=(shift_y, shift_x), dims=(-2, -1))

            optimizer.zero_grad(set_to_none=True)
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * bx.size(0)
            _, preds = torch.max(out, 1)
            correct += (preds == by).sum().item()
            total += by.size(0)

        tr_loss = running_loss / total
        tr_acc = (correct / total) * 100.0

        # Validation pass
        model.eval()
        v_loss = 0.0
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                loss = criterion(out, by)
                v_loss += loss.item() * bx.size(0)
                _, preds = torch.max(out, 1)
                v_correct += (preds == by).sum().item()
                v_total += by.size(0)

        val_loss = v_loss / v_total
        val_acc = (v_correct / v_total) * 100.0

        curr_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        ep_time = time.time() - ep_start

        history["train_loss"].append(round(tr_loss, 4))
        history["train_acc"].append(round(tr_acc, 2))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 2))
        history["lr"].append(curr_lr)
        history["duration"].append(round(ep_time, 1))

        # Check for best model
        is_best = val_loss < best_val_loss
        best_tag = ""
        if is_best:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            best_tag = " --> [BEST CHECKPOINT SAVED]"

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "classes": CLASSES,
            }, best_path)
        else:
            patience_counter += 1

        # Always save last checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "classes": CLASSES,
        }, last_path)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"LR: {curr_lr:.6f} ({ep_time:.1f}s){best_tag}"
        )

        gc.collect()

        if patience_counter >= patience:
            print(f"\n[!] Early stopping triggered at epoch {epoch}. Best epoch was Epoch {best_epoch}.")
            break

    total_training_time = time.time() - train_start
    print("=" * 78)
    print(f"Training completed in {total_training_time:.1f}s. Best weights: {best_path}")
    print("=" * 78)

    # 6. Load Best Weights and Evaluate on Test Set
    print("\n[5/5] Evaluating Best Model on Test Partition (7,178 images)...")
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded Best Model from Epoch {ckpt['epoch']} (Val Acc: {ckpt['val_acc']:.2f}%)")

    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            out = model(bx)
            _, preds = torch.max(out, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(by.cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    test_acc = accuracy_score(y_true, y_pred) * 100.0

    cls_report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASSES)))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    # Print Report
    print("\n" + "=" * 75)
    print("           FINAL MODEL - TEST SET CLASSIFICATION REPORT")
    print("=" * 75)
    print(f" Final Test Accuracy  : {test_acc:.2f}%")
    print(f" Macro Avg F1-Score   : {macro_f1 * 100:.2f}%")
    print(f" Weighted Avg F1-Score: {weighted_f1 * 100:.2f}%")
    print("-" * 75)
    rows = []
    for cls_name in CLASSES:
        m = cls_report.get(cls_name, {})
        rows.append({
            "Emotion": cls_name,
            "Precision": f"{m.get('precision', 0.0) * 100:.2f}%",
            "Recall": f"{m.get('recall', 0.0) * 100:.2f}%",
            "F1-Score": f"{m.get('f1-score', 0.0) * 100:.2f}%",
            "Support": int(m.get("support", 0)),
        })
    print(f"{'Emotion':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
    print("-" * 75)
    for r in rows:
        print(f"{r['Emotion']:<12} {r['Precision']:<12} {r['Recall']:<12} {r['F1-Score']:<12} {r['Support']:<8}")
    print("=" * 75)

    # 7. Generate Visual Artifacts
    # Plot A: Training History
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ep_range = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(ep_range, history["train_loss"], "o-", label="Train Loss", color="#2980b9", lw=2)
    axes[0].plot(ep_range, history["val_loss"], "s--", label="Val Loss", color="#e74c3c", lw=2)
    axes[0].set_title("Final Model - Loss Curves", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Loss", fontsize=10, fontweight="bold")
    axes[0].legend(frameon=True)

    axes[1].plot(ep_range, history["train_acc"], "o-", label="Train Acc", color="#27ae60", lw=2)
    axes[1].plot(ep_range, history["val_acc"], "s--", label="Val Acc", color="#f39c12", lw=2)
    axes[1].set_title("Final Model - Accuracy Curves", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Accuracy (%)", fontsize=10, fontweight="bold")
    axes[1].legend(frameon=True)

    plt.suptitle("Final Regularized CNN (FER-2013) - Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    hist_fig_path = GRAPHS_DIR / "final_training_history.png"
    plt.savefig(hist_fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[+] Saved training curves to: {hist_fig_path}")

    # Plot B: Confusion Matrix
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
    cm_fig_path = CONFUSION_MATRIX_DIR / "final_confusion_matrix.png"
    plt.savefig(cm_fig_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved confusion matrix to: {cm_fig_path}")

    # Plot C: Per-Class F1-Score Bar Chart
    f1_scores = [cls_report[c]["f1-score"] * 100 for c in CLASSES]
    plt.figure(figsize=(9, 5))
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
    f1_fig_path = GRAPHS_DIR / "final_per_class_f1.png"
    plt.savefig(f1_fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved per-class F1 plot to: {f1_fig_path}")

    # Save JSON summary
    summary_data = {
        "model_name": model_name,
        "epochs_completed": len(history["train_loss"]),
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_acc": round(best_val_acc, 2),
        "test_accuracy": round(test_acc, 2),
        "macro_precision": round(macro_p * 100, 2),
        "macro_recall": round(macro_r * 100, 2),
        "macro_f1": round(macro_f1 * 100, 2),
        "weighted_f1": round(weighted_f1 * 100, 2),
        "training_time_seconds": round(total_training_time, 1),
        "history": history,
        "per_class_metrics": {c: {k: round(v, 4) for k, v in cls_report[c].items()} for c in CLASSES},
    }
    json_path = RESULTS_DIR / "final_model_evaluation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)
    print(f"[+] Saved evaluation metrics to: {json_path}")

    return summary_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Final Regularized CNN on FER-2013")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Base learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint if exists")
    args = parser.parse_args()

    train_final_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        fresh_start=not args.resume,
    )
