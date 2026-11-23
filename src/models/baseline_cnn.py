"""
Baseline Convolutional Neural Network (CNN) Architecture for Facial Expression Recognition.
"""
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.config import NUM_CLASSES


class BaselineCNN(nn.Module):
    """
    Standard baseline CNN architecture for FER-2013 facial expression recognition.
    
    Structure:
    - 3 Convolutional blocks with (Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU -> MaxPool -> Dropout)
    - Fully connected classifier with Dropout regularization
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = NUM_CLASSES,
        conv_dropout: float = 0.25,
        dense_dropout: float = 0.5,
    ):
        super(BaselineCNN, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Conv Block 1: Input (B, C, 48, 48) -> Output (B, 32, 24, 24)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=conv_dropout),
        )

        # Conv Block 2: (B, 32, 24, 24) -> (B, 64, 12, 12)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=conv_dropout),
        )

        # Conv Block 3: (B, 64, 12, 12) -> (B, 128, 6, 6)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=conv_dropout),
        )

        # Classifier Head: Flatten (128 * 6 * 6 = 4608) -> Dense -> 7 classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dense_dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dense_dropout),
            nn.Linear(128, num_classes),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming normal initialization for conv layers, Xavier for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (B, C, 48, 48)
        Returns:
            Logits of shape (B, num_classes)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        logits = self.classifier(x)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax class probabilities."""
        self.eval()
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predicted class indices."""
        self.eval()
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)

    def get_num_params(self) -> Tuple[int, int]:
        """Returns (trainable_parameters, total_parameters)."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total

    def summary(self, input_size: Tuple[int, int, int] = (1, 48, 48)) -> None:
        """Prints architectural summary and parameter counts."""
        trainable, total = self.get_num_params()
        print("=" * 65)
        print("                 BASELINE CNN ARCHITECTURE")
        print("=" * 65)
        print(f" Input Shape       : (Batch, {input_size[0]}, {input_size[1]}, {input_size[2]})")
        print(f" Block 1           : 2x Conv2D(32, 3x3) -> MaxPool -> Dropout")
        print(f" Block 2           : 2x Conv2D(64, 3x3) -> MaxPool -> Dropout")
        print(f" Block 3           : 2x Conv2D(128, 3x3) -> MaxPool -> Dropout")
        print(f" Classifier Head   : Linear(4608 -> 256) -> Linear(256 -> 128) -> Linear(128 -> {self.num_classes})")
        print("-" * 65)
        print(f" Trainable Params  : {trainable:,}")
        print(f" Non-Trainable     : {total - trainable:,}")
        print(f" Total Parameters  : {total:,}")
        print("=" * 65)

    def save(self, path: Path) -> None:
        """Saves model state dictionary."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.state_dict(),
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
        }, path)
        print(f"Model saved to: {path}")

    @classmethod
    def load(cls, path: Path, map_location: str = "cpu") -> "BaselineCNN":
        """Loads model from checkpoint."""
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(
            in_channels=checkpoint.get("in_channels", 1),
            num_classes=checkpoint.get("num_classes", NUM_CLASSES),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model


def build_baseline_cnn(
    in_channels: int = 1,
    num_classes: int = NUM_CLASSES,
    device: Optional[torch.device] = None,
) -> BaselineCNN:
    """Helper function to instantiate and place model onto device."""
    model = BaselineCNN(in_channels=in_channels, num_classes=num_classes)
    if device is not None:
        model.to(device)
    return model


if __name__ == "__main__":
    model = build_baseline_cnn()
    model.summary()
    dummy = torch.randn(4, 1, 48, 48)
    out = model(dummy)
    print(f"Forward test output shape: {out.shape} (Expected: (4, 7))")
