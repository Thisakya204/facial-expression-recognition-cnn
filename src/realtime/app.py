"""
Real-Time Facial Expression Recognition Application using OpenCV and PyTorch CNN.
"""
from typing import Optional
from pathlib import Path
import argparse
import time
from datetime import datetime
import cv2
import numpy as np

from .face_detector import FaceDetector
from .emotion_classifier import EmotionClassifier
from .visualizer import RealtimeVisualizer
from ..data.config import PROJECT_ROOT, RESULTS_DIR


class RealtimeFERApp:
    """
    Main Real-Time Facial Expression Recognition Application.
    Orchestrates:
    - OpenCV Webcam Capture
    - Face Detection
    - Deep Learning CNN Inference
    - Real-Time Confidence Score HUD and Visualizations
    """

    def __init__(
        self,
        camera_id: int = 0,
        model_path: Optional[str] = None,
        source: Optional[str] = None,
        show_hud: bool = True,
        window_name: str = "Facial Expression Recognition (FER) - Member 4",
    ):
        self.camera_id = camera_id
        self.source = source
        self.window_name = window_name

        print("=" * 70)
        print("    INITIALIZING REAL-TIME FACIAL EXPRESSION RECOGNITION")
        print("=" * 70)

        # 1. Initialize Face Detector
        print("[1/3] Loading Face Detector...")
        self.detector = FaceDetector()

        # 2. Initialize CNN Classifier
        print("[2/3] Loading CNN Emotion Classifier...")
        self.classifier = EmotionClassifier(model_path=model_path)

        # 3. Initialize Visualizer
        print("[3/3] Setting up Visualizer and HUD...")
        self.visualizer = RealtimeVisualizer(show_hud=show_hud)

        # Screenshots directory
        self.screenshots_dir = RESULTS_DIR / "screenshots"
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

        print("Initialization complete! Ready to start video feed.\n")

    def run(self) -> None:
        """Starts live video loop and handles UI interactions."""
        # Open video source (camera index, video file, or image)
        if self.source and Path(self.source).exists():
            input_source = self.source
            print(f"Opening video/image file source: {input_source}")
        else:
            input_source = self.camera_id
            print(f"Connecting to Webcam (Device ID: {self.camera_id})...")

        cap = cv2.VideoCapture(input_source)

        if not cap.isOpened():
            print(f"\n[!] Error: Unable to access video source ({input_source}).")
            print("Troubleshooting Tips:")
            print("  1. Verify your webcam is plugged in and permissions are granted.")
            print("  2. Try passing a different camera index: --camera 1")
            print("  3. Or test with an image or video file: --source path/to/image.jpg")
            return

        # Attempt to set camera resolution (720p preferred for optimal balance of speed and clarity)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("\n" + "=" * 70)
        print("  LIVE STREAM RUNNING - KEYBOARD CONTROLS:")
        print("  - Press 'q' or 'ESC' to Exit")
        print("  - Press 'h' to Toggle Confidence HUD on/off")
        print("  - Press 's' to Save a Screenshot")
        print("=" * 70 + "\n")

        fps = 0.0
        frame_count = 0
        fps_timer = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    # If reading video file, loop back to start
                    if self.source:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    print("[!] Video feed ended or disconnected.")
                    break

                # Mirror frame horizontally for natural webcam interaction
                if not self.source:
                    frame = cv2.flip(frame, 1)

                display_frame = frame.copy()

                # 1. Detect faces
                faces = self.detector.detect_faces(frame)

                top_face_probs = None
                top_face_emotion = None

                # 2. Process detected faces
                if len(faces) > 0:
                    for i, bbox in enumerate(faces):
                        # Extract & preprocess face ROI
                        face_roi = self.detector.extract_face_roi(frame, bbox)

                        # CNN Inference
                        emotion, conf, probs = self.classifier.predict(face_roi)

                        # Save primary face probabilities for side HUD
                        if i == 0:
                            top_face_probs = probs
                            top_face_emotion = emotion

                        # Draw bounding box and confidence badge
                        self.visualizer.draw_face_box(
                            frame=display_frame,
                            bbox=bbox,
                            emotion=emotion,
                            confidence=conf,
                        )
                else:
                    self.classifier.reset_smoothing()

                # 3. Draw Confidence HUD
                if top_face_probs:
                    self.visualizer.draw_confidence_hud(
                        frame=display_frame,
                        probabilities=top_face_probs,
                        active_emotion=top_face_emotion,
                    )

                # 4. Compute and display FPS
                frame_count += 1
                if frame_count >= 10:
                    now = time.time()
                    elapsed = now - fps_timer
                    fps = frame_count / elapsed if elapsed > 0 else 0.0
                    frame_count = 0
                    fps_timer = now

                self.visualizer.draw_status_header(
                    frame=display_frame,
                    fps=fps,
                    num_faces=len(faces),
                )

                # 5. Display window
                cv2.imshow(self.window_name, display_frame)

                # 6. Key bindings
                key = cv2.waitKey(1) & 0xFF
                if key in [ord("q"), ord("Q"), 27]:  # 'q' or ESC
                    print("Exiting application...")
                    break
                elif key in [ord("h"), ord("H")]:    # 'h'
                    self.visualizer.show_hud = not self.visualizer.show_hud
                    print(f"HUD visibility: {'ON' if self.visualizer.show_hud else 'OFF'}")
                elif key in [ord("s"), ord("S")]:    # 's'
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_file = self.screenshots_dir / f"fer_snapshot_{timestamp}.png"
                    cv2.imwrite(str(out_file), display_frame)
                    print(f"[+] Saved snapshot to: {out_file}")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Camera released and windows closed. Goodbye!")


def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time Facial Expression Recognition with OpenCV and PyTorch CNN")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--model", type=str, default=None, help="Path to trained PyTorch .pth checkpoint")
    parser.add_argument("--source", type=str, default=None, help="Optional image or video file path for testing")
    parser.add_argument("--no_hud", action="store_true", help="Disable confidence HUD overlay")
    return parser.parse_args()


def main():
    args = parse_args()
    app = RealtimeFERApp(
        camera_id=args.camera,
        model_path=args.model,
        source=args.source,
        show_hud=not args.no_hud,
    )
    app.run()


if __name__ == "__main__":
    main()
