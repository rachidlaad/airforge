from __future__ import annotations

import argparse
import time
import urllib.request
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .generator import GenerationError, generate_page
from .gestures import GestureState, HandGestureDetector
from .sketch import detect_shapes, interpret_layout, sample_canvas


DRAW_COLOR = (255, 255, 255)
ERASE_RADIUS = 28
WINDOW_NAME = "AirForge"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output).resolve()

    if args.sample:
        run_sample(output_dir, args.export)
        return

    run_webcam(output_dir, args.export, args.camera)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw website wireframes in the air and generate landing pages.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--output", default="generated", help="Directory for generated files.")
    parser.add_argument("--export", choices=["html", "react"], default="html", help="Generated output format.")
    parser.add_argument("--sample", action="store_true", help="Generate from a synthetic sketch without a webcam.")
    return parser.parse_args()


def run_sample(output_dir: Path, export: str) -> None:
    canvas = sample_canvas()
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "sketch.png"), canvas)
    files = generate_from_canvas(canvas, output_dir, export)
    print(f"Generated {files.html_path}")
    if files.react_path:
        print(f"Generated {files.react_path}")


def run_webcam(output_dir: Path, export: str, camera_index: int) -> None:
    try:
        hand_tracker = _create_hand_tracker()
    except ImportError as exc:
        raise SystemExit("MediaPipe is required for webcam tracking. Run: pip install -r requirements.txt") from exc

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {camera_index}.")

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Camera opened, but no frame was received.")

    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    detector = HandGestureDetector()
    previous_point: tuple[int, int] | None = None
    last_generate_time = 0.0
    last_clear_time = 0.0
    last_status = "Show your index finger to draw."

    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                last_status = "No camera frame received."
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hand_tracker.process(rgb)

            gesture = GestureState("idle", (0, 0))
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                gesture = detector.classify(hand_landmarks, width, height)
                previous_point, last_status, last_generate_time, last_clear_time = _apply_gesture(
                    gesture,
                    canvas,
                    previous_point,
                    output_dir,
                    export,
                    last_generate_time,
                    last_clear_time,
                )
            else:
                previous_point = None
                last_status = "No hand detected."

            preview = _compose_preview(frame, canvas, gesture, last_status)
            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("c"):
                canvas[:] = 0
                previous_point = None
                last_status = "Canvas cleared."
            if key == ord("g"):
                try:
                    generate_from_canvas(canvas, output_dir, export)
                    last_status = f"Generated {output_dir / 'index.html'}"
                except GenerationError as exc:
                    _print_generation_error(exc, output_dir)
                    last_status = _short_error(exc)
            if key == ord("s"):
                cv2.imwrite(str(output_dir / "sketch.png"), canvas)
                last_status = f"Saved {output_dir / 'sketch.png'}"
    finally:
        hand_tracker.close()
        cap.release()
        cv2.destroyAllWindows()


def generate_from_canvas(canvas: np.ndarray, output_dir: Path, export: str):
    shapes = detect_shapes(canvas)
    layout = interpret_layout(shapes, canvas.shape[1], canvas.shape[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    sketch_path = output_dir / "sketch.png"
    cv2.imwrite(str(sketch_path), canvas)
    files = generate_page(layout, output_dir, export, sketch_path)
    webbrowser.open(files.html_path.as_uri())
    return files


def _apply_gesture(
    gesture: GestureState,
    canvas: np.ndarray,
    previous_point: tuple[int, int] | None,
    output_dir: Path,
    export: str,
    last_generate_time: float,
    last_clear_time: float,
) -> tuple[tuple[int, int] | None, str, float, float]:
    now = time.monotonic()
    status = f"Gesture: {gesture.name}"

    if gesture.name == "draw":
        if previous_point is not None:
            cv2.line(canvas, previous_point, gesture.index_tip, DRAW_COLOR, 7, cv2.LINE_AA)
        cv2.circle(canvas, gesture.index_tip, 5, DRAW_COLOR, -1, cv2.LINE_AA)
        return gesture.index_tip, "Drawing.", last_generate_time, last_clear_time

    if gesture.name == "erase" and gesture.erase_point:
        cv2.circle(canvas, gesture.erase_point, ERASE_RADIUS, (0, 0, 0), -1, cv2.LINE_AA)
        return None, "Pinch erase.", last_generate_time, last_clear_time

    if gesture.name == "clear":
        if now - last_clear_time > 1.2:
            canvas[:] = 0
            last_clear_time = now
            status = "Open palm cleared the canvas."
        else:
            status = "Clear gesture cooling down."
        return None, status, last_generate_time, last_clear_time

    if gesture.name == "generate":
        if now - last_generate_time > 2.5:
            try:
                generate_from_canvas(canvas, output_dir, export)
                last_generate_time = now
                status = f"Generated {output_dir / 'index.html'}"
            except GenerationError as exc:
                _print_generation_error(exc, output_dir)
                status = _short_error(exc)
        else:
            status = "Generation cooling down."
        return None, status, last_generate_time, last_clear_time

    return None, status, last_generate_time, last_clear_time


def _compose_preview(frame: np.ndarray, canvas: np.ndarray, gesture: GestureState, status: str) -> np.ndarray:
    overlay = cv2.addWeighted(frame, 0.72, canvas, 0.88, 0)
    panel = overlay.copy()
    cv2.rectangle(panel, (16, 16), (min(690, overlay.shape[1] - 16), 116), (20, 28, 34), -1)
    overlay = cv2.addWeighted(panel, 0.78, overlay, 0.22, 0)
    lines = [
        "AirForge: index draws | pinch erases | palm clears | thumbs-up generates",
        status,
        "Keys: g generate | c clear | s save sketch | q quit",
    ]
    for idx, line in enumerate(lines):
        cv2.putText(
            overlay,
            line,
            (30, 46 + idx * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (245, 248, 250),
            2,
            cv2.LINE_AA,
        )

    if gesture.name in {"draw", "erase", "idle"} and gesture.index_tip != (0, 0):
        color = (46, 204, 113) if gesture.name == "draw" else (70, 170, 255)
        cv2.circle(overlay, gesture.index_tip, 10, color, 2, cv2.LINE_AA)
    if gesture.erase_point:
        cv2.circle(overlay, gesture.erase_point, ERASE_RADIUS, (70, 170, 255), 2, cv2.LINE_AA)
    return overlay


@dataclass
class _HandResults:
    multi_hand_landmarks: list


class _SolutionsHandTracker:
    def __init__(self, hands_module) -> None:
        self._hands = hands_module.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.55,
        )

    def process(self, rgb: np.ndarray):
        return self._hands.process(rgb)

    def close(self) -> None:
        self._hands.close()


class _TasksHandTracker:
    def __init__(self) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        model_path = _ensure_hand_model()
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.65,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self._mp = mp
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def process(self, rgb: np.ndarray) -> _HandResults:
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )
        result = self._landmarker.detect_for_video(image, int(time.monotonic() * 1000))
        return _HandResults(multi_hand_landmarks=result.hand_landmarks)

    def close(self) -> None:
        self._landmarker.close()


def _create_hand_tracker():
    import mediapipe as mp

    solutions = getattr(mp, "solutions", None)
    if solutions is not None and hasattr(solutions, "hands"):
        return _SolutionsHandTracker(solutions.hands)

    return _TasksHandTracker()


def _ensure_hand_model() -> Path:
    model_dir = Path.home() / ".airforge"
    model_path = model_dir / "hand_landmarker.task"
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(HAND_MODEL_URL, model_path)
    except Exception as exc:
        raise SystemExit(
            "MediaPipe Tasks needs the hand landmarker model. "
            f"Could not download it from {HAND_MODEL_URL}: {exc}"
        ) from exc
    return model_path


def _short_error(exc: Exception) -> str:
    first_line = str(exc).splitlines()[0]
    if len(first_line) > 92:
        return first_line[:89] + "..."
    return first_line


def _print_generation_error(exc: Exception, output_dir: Path) -> None:
    print("\nAirForge generation failed:")
    print(str(exc))
    print(f"Check {output_dir / 'codex-error.txt'} and {output_dir / 'codex-result.txt'} for details.\n")
