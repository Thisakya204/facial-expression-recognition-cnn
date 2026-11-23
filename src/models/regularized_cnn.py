"""
Configurable CNN Models for Regularization Experiments (Dropout, Batch Normalization, and Ablation Studies).
"""
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.config import NUM_CLASSES


class RegularizedCNN(nn.Module):
    """
    Ablation-friendly CNN Architecture for FER-2013.
    Allows isolated evaluation of:
    - Batch Normalization (on/off)
    - Dropout Regularization (on/off, configurable rates)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = NUM_CLASSES,
        use_batch_norm: bool = True,
        use_dropout: bool = True,
        conv_dropout: float = 0.25,
        dense_dropout: float = 0.5,
        model_name: str = "RegularizedCNN",
    ):
        super(RegularizedCNN, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.conv_dropout = conv_dropout if use_dropout else 0.0
        self.dense_dropout = dense_dropout if use_dropout else 0.0
        self.model_name = model_name

        # Block 1: (B, in_channels, 48, 48) -> (B, 32, 24, 24)
        self.conv1_1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32) if use_batch_norm else nn.Identity()
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32) if use_batch_norm else nn.Identity()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(p=self.conv_dropout) if use_dropout else nn.Identity()

        # Block 2: (B, 32, 24, 24) -> (B, 64, 12, 12)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64) if use_batch_norm else nn.Identity()
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64) if use_batch_norm else nn.Identity()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(p=self.conv_dropout) if use_dropout else nn.Identity()

        # Block 3: (B, 64, 12, 12) -> (B, 128, 6, 6)
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128) if use_batch_norm else nn.Identity()
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128) if use_batch_norm else nn.Identity()
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout2d(p=self.conv_dropout) if use_dropout else nn.Identity()

        # Classifier Head: (B, 128 * 6 * 6 = 4608) -> Dense -> Logits (B, num_classes)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.bn_fc1 = nn.BatchNorm1d(256) if use_batch_norm else nn.Identity()
        self.drop_fc1 = nn.Dropout(p=self.dense_dropout) if use_dropout else nn.Identity()

        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128) if use_batch_norm else nn.Identity()
        self.drop_fc2 = nn.Dropout(p=self.dense_dropout) if use_dropout else nn.Identity()

        self.out = nn.Linear(128, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
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
        # Block 1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        # Block 3
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)
        x = self.drop3(x)

        # Classifier
        x = self.flatten(x)
        x = self.drop_fc1(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.drop_fc2(F.relu(self.bn_fc2(self.fc2(x))))
        logits = self.out(x)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return F.softmax(self.forward(x), dim=-1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.argmax(self.forward(x), dim=-1)

    def get_num_params(self) -> Tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total

    def summary(self) -> None:
        trainable, total = self.get_num_params()
        print("=" * 70)
        print(f"       MODEL CONFIGURATION: {self.model_name.upper()}")
        print("=" * 70)
        print(f" Batch Normalization : {'Enabled' if self.use_batch_norm else 'Disabled'}")
        print(f" Dropout             : {'Enabled' if self.use_dropout else 'Disabled'} (Conv: {self.conv_dropout}, Dense: {self.dense_dropout})")
        print(f" Trainable Params    : {trainable:,}")
        print(f" Total Params        : {total:,}")
        print("=" * 70)


# Dedicated experiment model constructors
def StandardCNN(in_channels: int = 1, num_classes: int = NUM_CLASSES) -> RegularizedCNN:
    """Baseline CNN with NO Regularization (No Dropout, No BatchNorm)."""
    return RegularizedCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        use_batch_norm=False,
        use_dropout=False,
        model_name="Standard_NoReg_CNN",
    )


def DropoutCNN(
    in_channels: int = 1,
    num_classes: int = NUM_CLASSES,
    conv_dropout: float = 0.25,
    dense_dropout: float = 0.5,
) -> RegularizedCNN:
    """CNN with Dropout Regularization only (No BatchNorm)."""
    return RegularizedCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        use_batch_norm=False,
        use_dropout=True,
        conv_dropout=conv_dropout,
        dense_dropout=dense_dropout,
        model_name="Dropout_CNN",
    )


def BatchNormCNN(in_channels: int = 1, num_classes: int = NUM_CLASSES) -> RegularizedCNN:
    """CNN with Batch Normalization only (No Dropout)."""
    return RegularizedCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        use_batch_norm=True,
        use_dropout=False,
        model_name="BatchNorm_CNN",
    )


def FullRegularizedCNN(
    in_channels: int = 1,
    num_classes: int = NUM_CLASSES,
    conv_dropout: float = 0.25,
    dense_dropout: float = 0.5,
) -> RegularizedCNN:
    """CNN with BOTH Batch Normalization and Dropout Regularization."""
    return RegularizedCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        use_batch_norm=True,
        use_dropout=True,
        conv_dropout=conv_dropout,
        dense_dropout=dense_dropout,
        model_name="Full_Regularized_CNN",
    )
