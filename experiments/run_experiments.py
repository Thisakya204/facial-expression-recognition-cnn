"""
CLI Runner for Regularization Experiments (Dropout, Batch Normalization, and Data Augmentation).
"""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.experiments import RegularizationExperimentSuite


def parse_args():
    parser = argparse.ArgumentParser(description="Run Regularization Experiments for Facial Expression Recognition")
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["all", "Baseline_NoReg", "Dropout_Only", "BatchNorm_Only", "Augmentation_Only", "Full_Regularization"],
        help="Experiment to run: 'all' or specific experiment key",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs per experiment (default: 10)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 0.001)")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience (default: 6)")
    return parser.parse_args()


def main():
    args = parse_args()
    suite = RegularizationExperimentSuite(
        batch_size=args.batch_size,
        lr=args.lr,
    )

    if args.experiment == "all":
        suite.run_all(epochs=args.epochs, patience=args.patience)
    else:
        suite.run_single_experiment(args.experiment, epochs=args.epochs, patience=args.patience)
        suite.save_summary_json()
        suite.print_summary_table()


if __name__ == "__main__":
    main()
