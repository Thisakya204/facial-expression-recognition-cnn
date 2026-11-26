"""
Real-Time HUD, Bounding Box, and Confidence Score Visualizer for Facial Expression Recognition.
"""
from typing import Dict, Tuple, List, Optional
import cv2
import numpy as np

from ..data.config import CLASSES


# BGR Color palette for distinct visual distinction per emotion
EMOTION_COLORS = {
    "happy": (46, 204, 113),      # Emerald Green
    "surprise": (241, 196, 15),   # Golden Yellow
    "neutral": (189, 195, 199),   # Silver Gray
    "sad": (52, 152, 219),        # Dodger Blue
    "fear": (155, 89, 182),       # Amethyst Purple
    "angry": (231, 76, 60),       # Alizarin Red
    "disgust": (230, 126, 34),    # Carrot Orange
}


class RealtimeVisualizer:
    """
    Renders styled bounding boxes, confidence badges, live probability distribution HUD,
    and performance stats onto camera frames.
    """

    def __init__(
        self,
        classes: List[str] = CLASSES,
        show_hud: bool = True,
        hud_width: int = 240,
    ):
        self.classes = classes
        self.show_hud = show_hud
        self.hud_width = hud_width
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_face_box(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        emotion: str,
        confidence: float,
    ) -> np.ndarray:
        """
        Draws bounding box with stylized corner accents and top badge displaying emotion and confidence.
        """
        x, y, w, h = bbox
        color = EMOTION_COLORS.get(emotion.lower(), (0, 255, 0))

        # Main bounding rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)

        # Stylish corner accents
        line_len = max(12, int(w * 0.15))
        thick = 4
        # Top-left
        cv2.line(frame, (x, y), (x + line_len, y), color, thick)
        cv2.line(frame, (x, y), (x, y + line_len), color, thick)
        # Top-right
        cv2.line(frame, (x + w, y), (x + w - line_len, y), color, thick)
        cv2.line(frame, (x + w, y), (x + w, y + line_len), color, thick)
        # Bottom-left
        cv2.line(frame, (x, y + h), (x + line_len, y + h), color, thick)
        cv2.line(frame, (x, y + h), (x, y + h - line_len), color, thick)
        # Bottom-right
        cv2.line(frame, (x + w, y + h), (x + w - line_len, y + h), color, thick)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - line_len), color, thick)

        # Confidence Badge above face
        label_text = f"{emotion.capitalize()} ({confidence:.1f}%)"
        font_scale = 0.55
        thickness = 1
        (label_w, label_h), baseline = cv2.getTextSize(label_text, self.font, font_scale, thickness)

        badge_y1 = max(0, y - label_h - 12)
        badge_y2 = y
        badge_x1 = x
        badge_x2 = x + label_w + 16

        # Draw filled badge
        cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), color, cv2.FILLED)
        # Dark text on light color, white text on dark
        text_color = (0, 0, 0) if emotion.lower() in ["surprise", "neutral", "happy"] else (255, 255, 255)
        cv2.putText(
            frame,
            label_text,
            (badge_x1 + 8, badge_y2 - 6),
            self.font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

        return frame

    def draw_confidence_hud(
        self,
        frame: np.ndarray,
        probabilities: Dict[str, float],
        active_emotion: Optional[str] = None,
    ) -> np.ndarray:
        """
        Draws semi-transparent horizontal bar HUD displaying confidence across all 7 emotions.
        """
        if not self.show_hud or not probabilities:
            return frame

        frame_h, frame_w = frame.shape[:2]
        hud_w = min(self.hud_width, frame_w - 20)
        hud_h = 25 + len(self.classes) * 26 + 10
        margin_x = frame_w - hud_w - 15
        margin_y = 15

        # Create semi-transparent background overlay
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (margin_x, margin_y),
            (margin_x + hud_w, margin_y + hud_h),
            (25, 25, 25),
            cv2.FILLED,
        )
        # Border
        cv2.rectangle(
            overlay,
            (margin_x, margin_y),
            (margin_x + hud_w, margin_y + hud_h),
            (100, 100, 100),
            1,
        )
        alpha = 0.78
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # HUD Title
        cv2.putText(
            frame,
            "EMOTION CONFIDENCE",
            (margin_x + 12, margin_y + 20),
            self.font,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        # Draw probability bars for each emotion
        bar_start_x = margin_x + 72
        bar_max_width = hud_w - 120

        for idx, cls_name in enumerate(self.classes):
            prob = probabilities.get(cls_name, 0.0)
            item_y = margin_y + 44 + idx * 26
            is_active = (active_emotion and cls_name.lower() == active_emotion.lower())

            # Text label
            label_col = (255, 255, 255) if is_active else (180, 180, 180)
            cv2.putText(
                frame,
                f"{cls_name.capitalize()[:7]:<7}",
                (margin_x + 10, item_y),
                self.font,
                0.40,
                label_col,
                1,
                cv2.LINE_AA,
            )

            # Background bar slot
            bar_y1 = item_y - 10
            bar_y2 = item_y + 2
            cv2.rectangle(
                frame,
                (bar_start_x, bar_y1),
                (bar_start_x + bar_max_width, bar_y2),
                (55, 55, 55),
                cv2.FILLED,
            )

            # Filled bar
            filled_w = int((prob / 100.0) * bar_max_width)
            fill_color = EMOTION_COLORS.get(cls_name.lower(), (150, 150, 150))
            if filled_w > 0:
                cv2.rectangle(
                    frame,
                    (bar_start_x, bar_y1),
                    (bar_start_x + filled_w, bar_y2),
                    fill_color,
                    cv2.FILLED,
                )

            # Percentage text
            pct_text = f"{prob:4.1f}%"
            cv2.putText(
                frame,
                pct_text,
                (bar_start_x + bar_max_width + 6, item_y),
                self.font,
                0.38,
                (240, 240, 240) if is_active else (150, 150, 150),
                1,
                cv2.LINE_AA,
            )

        return frame

    def draw_status_header(
        self,
        frame: np.ndarray,
        fps: float,
        num_faces: int,
    ) -> np.ndarray:
        """Draws top status bar with FPS, face count, and control hints."""
        # Top-left status badge
        fps_text = f"FPS: {fps:.1f} | Faces: {num_faces}"
        cv2.putText(
            frame,
            fps_text,
            (16, 26),
            self.font,
            0.55,
            (0, 255, 128),
            2,
            cv2.LINE_AA,
        )

        # Bottom help footer
        footer_text = "[q]: Quit  |  [h]: Toggle HUD  |  [s]: Screenshot"
        frame_h = frame.shape[0]
        cv2.putText(
            frame,
            footer_text,
            (16, frame_h - 14),
            self.font,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        return frame
