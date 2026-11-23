"""
End-to-End Training & Evaluation Script for Baseline CNN.
"""
import argparse
from pathlib import Path
import torch

from ..data.config import PROJECT_ROOT, BATCH_SIZE
from ..data.dataset import get_torch_dataloaders
from ..data.distribution import ClassDistributionAnalyzer
from ..models.baseline_cnn import build_baseline_cnn
from .trainer import ModelTrainer
from ..evaluation.evaluate import ModelEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Train Baseline CNN for Facial Expression Recognition")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 0.001)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 Regularization weight decay")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (default: 5)")
    parser.add_argument("--use_class_weights", action="store_true", default=True, help="Use balanced class weights in loss")
    parser.add_argument("--model_name", type=str, default="baseline_cnn", help="Model name prefix for checkpoints")
    return parser.parse_args()


def train_baseline(
    epochs: int = 15,
    batch_size: int = BATCH_SIZE,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    use_class_weights: bool = True,
    model_name: str = "baseline_cnn",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # 1. Load DataLoaders
    print("Preparing PyTorch DataLoaders (Train, Val, Test)...")
    loaders = get_torch_dataloaders(
        batch_size=batch_size,
        augment_train=True,
        grayscale=True,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    # 2. Compute Class Weights for CrossEntropyLoss
    class_weights_tensor = None
    if use_class_weights:
        analyzer = ClassDistributionAnalyzer()
        stats = analyzer.analyze()
        raw_weights = [stats["class_weights_by_idx"][i] for i in range(len(stats["class_weights_by_idx"]))]
        class_weights_tensor = torch.tensor(raw_weights, dtype=torch.float32).to(device)
        print(f"Applied balanced class weights to loss function: {raw_weights}")

    # 3. Instantiate Model
    model = build_baseline_cnn(in_channels=1, num_classes=7, device=device)
    model.summary()

    # 4. Train Model
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        class_weights=class_weights_tensor,
        lr=lr,
        weight_decay=weight_decay,
    )

    trainer.fit(
        epochs=epochs,
        early_stopping_patience=patience,
        model_name=model_name,
    )

    # 5. Evaluate Best Model on Test Set
    print("\nRunning Evaluation on Test Set...")
    evaluator = ModelEvaluator(model=model, test_loader=test_loader, device=device)
    evaluator.evaluate()
    evaluator.print_report()
    evaluator.plot_confusion_matrix()
    evaluator.plot_per_class_f1()
    evaluator.save_summary_json()

    print("\nAll artifacts (checkpoints, metrics, curves, confusion matrix) saved successfully!")
    return trainer, evaluator


if __name__ == "__main__":
    args = parse_args()
    train_baseline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        use_class_weights=args.use_class_weights,
        model_name=args.model_name,
    )
