"""
CNN Inference Engine and Temporal Probability Smoothing for Real-Time Emotion Classification.
"""
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from ..data.config import CLASSES, NUM_CLASSES, PROJECT_ROOT
from ..models.baseline_cnn import BaselineCNN
from ..models.regularized_cnn import RegularizedCNN, DropoutCNN, StandardCNN


class EmotionClassifier:
    """
    Handles CNN model loading, forward inference, softmax probabilities,
    and temporal smoothing to prevent flicker in live video.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        smoothing_alpha: float = 0.35,
        classes: List[str] = CLASSES,
    ):
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.classes = classes
        self.num_classes = len(classes)
        self.smoothing_alpha = smoothing_alpha

        # EMA smoothed probabilities for stability
        self.smoothed_probs: Optional[np.ndarray] = None

        # Load model
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

    def _find_default_checkpoint(self) -> Optional[Path]:
        """Scans the models directory for the best available checkpoint."""
        models_dir = PROJECT_ROOT / "models"
        candidates = [
            models_dir / "exp_Dropout_Only_best.pth",
            models_dir / "baseline_cnn_best.pth",
            models_dir / "exp_Baseline_NoReg_best.pth",
            models_dir / "exp_Dropout_Only_last.pth",
            models_dir / "exp_Baseline_NoReg_last.pth",
        ]
        for c in candidates:
            if c.exists():
                return c

        # Any .pth in models/
        other_pths = list(models_dir.glob("*.pth"))
        if other_pths:
            return other_pths[0]

        return None

    def _load_model(self, model_path: Optional[str]) -> torch.nn.Module:
        """Instantiates CNN architecture and loads trained weights."""
        resolved_path = Path(model_path) if model_path else self._find_default_checkpoint()

        if resolved_path and resolved_path.exists():
            print(f"Loading emotion recognition model weights from: {resolved_path.name}")
            checkpoint = torch.load(resolved_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint.get("model_state_dict", checkpoint)

            # Detect architecture type from keys
            if any(k.startswith("conv1_1") for k in state_dict.keys()):
                has_bn = any("bn1_1" in k for k in state_dict.keys())
                model = RegularizedCNN(
                    in_channels=1,
                    num_classes=self.num_classes,
                    use_batch_norm=has_bn,
                    use_dropout=True,
                )
            else:
                model = BaselineCNN(in_channels=1, num_classes=self.num_classes)

            model.load_state_dict(state_dict)
            print(f"Successfully loaded weights into {model.__class__.__name__}")
        else:
            model = BaselineCNN(in_channels=1, num_classes=self.num_classes)
            print("[!] Warning: No checkpoint found. Initializing model with default weights.")

        return model

    @torch.no_grad()
    def predict(self, face_roi: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Runs CNN inference on a preprocessed face ROI.

        Args:
            face_roi: (48, 48, 1) normalized float32 numpy array.

        Returns:
            top_emotion: Name of highest probability emotion (e.g. 'happy').
            confidence: Highest probability as a percentage (e.g. 91.4).
            prob_dict: Dictionary mapping each emotion class to its confidence percentage.
        """
        # Convert (H, W, C) -> (B, C, H, W)
        tensor_in = torch.from_numpy(face_roi).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        logits = self.model(tensor_in)
        raw_probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        # Apply Exponential Moving Average (EMA) smoothing for stability
        if self.smoothed_probs is None:
            self.smoothed_probs = raw_probs
        else:
            self.smoothed_probs = (
                self.smoothing_alpha * raw_probs
                + (1.0 - self.smoothing_alpha) * self.smoothed_probs
            )

        top_idx = int(np.argmax(self.smoothed_probs))
        top_emotion = self.classes[top_idx]
        confidence = float(self.smoothed_probs[top_idx] * 100.0)

        prob_dict = {
            cls_name: float(self.smoothed_probs[i] * 100.0)
            for i, cls_name in enumerate(self.classes)
        }

        return top_emotion, confidence, prob_dict

    def reset_smoothing(self) -> None:
        """Resets temporal smoothing history (e.g., when a face leaves the frame)."""
        self.smoothed_probs = None
