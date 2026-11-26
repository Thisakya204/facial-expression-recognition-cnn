"""
Face Detection and Region-of-Interest (ROI) Preprocessing for Real-Time FER.
"""
from typing import List, Tuple, Optional
from pathlib import Path
import os
import urllib.request
import cv2
import numpy as np

CASCADE_FILENAME = "haarcascade_frontalface_default.xml"
CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"


class FaceDetector:
    """
    OpenCV Haar Cascade Face Detector with automatic model resolution and ROI extraction.
    """

    def __init__(
        self,
        cascade_path: Optional[str] = None,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (40, 40),
    ):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.cascade_path = self._resolve_cascade_path(cascade_path)

        self.classifier = cv2.CascadeClassifier(str(self.cascade_path))
        if self.classifier.empty():
            raise RuntimeError(f"Failed to load Haar Cascade from: {self.cascade_path}")

    def _resolve_cascade_path(self, custom_path: Optional[str]) -> Path:
        """Finds cascade XML locally, in cv2 data, or downloads it if absent."""
        if custom_path and Path(custom_path).exists():
            return Path(custom_path)

        # Check local realtime directory
        local_dir = Path(__file__).resolve().parent
        local_cascade = local_dir / CASCADE_FILENAME
        if local_cascade.exists():
            return local_cascade

        # Check cv2 built-in data
        if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            cv_cascade = Path(cv2.data.haarcascades) / CASCADE_FILENAME
            if cv_cascade.exists():
                return cv_cascade

        # Download if needed
        print(f"Downloading Haar cascade XML to {local_cascade}...")
        local_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CASCADE_URL, str(local_cascade))
        return local_cascade

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detects faces in BGR or grayscale image frame.

        Args:
            frame: Numpy BGR or grayscale image array.

        Returns:
            List of bounding boxes: [(x, y, w, h), ...]
        """
        if frame.ndim == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        faces = self.classifier.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # Convert numpy array of rects to standard Python tuples
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    def extract_face_roi(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        target_size: Tuple[int, int] = (48, 48),
        margin_pct: float = 0.08,
    ) -> np.ndarray:
        """
        Extracts, pads with margin, crops, resizes, and normalizes the face region for CNN input.

        Args:
            frame: Input frame (BGR or grayscale).
            bbox: (x, y, w, h) tuple.
            target_size: (width, height) expected by CNN (48, 48).
            margin_pct: Context margin added around bounding box.

        Returns:
            Preprocessed numpy array of shape (target_size[1], target_size[0], 1) in [0, 1].
        """
        x, y, w, h = bbox
        img_h, img_w = frame.shape[:2]

        # Add slight margin around face
        dx = int(w * margin_pct)
        dy = int(h * margin_pct)
        x1 = max(0, x - dx)
        y1 = max(0, y - dy)
        x2 = min(img_w, x + w + dx)
        y2 = min(img_h, y + h + dy)

        # Crop face
        face_crop = frame[y1:y2, x1:x2]

        # Convert to grayscale if BGR
        if face_crop.ndim == 3 and face_crop.shape[2] == 3:
            gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray_crop = face_crop

        # Resize to CNN target dimensions
        resized = cv2.resize(gray_crop, target_size, interpolation=cv2.INTER_AREA)

        # Histogram equalization for contrast clarity under varying lighting
        equalized = cv2.equalizeHist(resized)

        # Normalize pixel values to [0, 1]
        normalized = equalized.astype(np.float32) / 255.0

        # Expand channel dimension: (H, W) -> (H, W, 1)
        return np.expand_dims(normalized, axis=-1)
