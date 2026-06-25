"""Real-time Sign Language Translator app using webcam and MediaPipe Hands."""

from __future__ import annotations

import argparse
import os
import time
from collections import Counter, deque
from pathlib import Path
from typing import Deque, Optional

import cv2
import mediapipe as mp
import pandas as pd

from predict import DEFAULT_MODEL_PATH, GesturePredictor, PredictionResult, resolve_project_path


LOG_PATH = Path("logs") / "gesture_history.csv"
SPACE_TOKEN = "SPACE"


class PredictionSmoother:
    """Reduce jitter by requiring the same prediction across recent frames."""

    def __init__(self, window_size: int = 8, min_votes: int = 5) -> None:
        self.window: Deque[str] = deque(maxlen=window_size)
        self.min_votes = min_votes

    def update(self, result: PredictionResult) -> Optional[str]:
        if not result.accepted:
            self.window.append("Unknown")
            return None

        self.window.append(result.label)
        label, count = Counter(self.window).most_common(1)[0]
        if label != "Unknown" and count >= self.min_votes:
            return label
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run real-time sign language translator.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index. Try 1 if 0 fails.")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH), help="Model path.")
    parser.add_argument("--log", type=str, default=str(LOG_PATH), help="Prediction log CSV path.")
    parser.add_argument("--confidence", type=float, default=0.65, help="Prediction confidence threshold.")
    parser.add_argument("--window", type=int, default=8, help="Smoothing window size.")
    parser.add_argument("--votes", type=int, default=5, help="Minimum votes for stable prediction.")
    parser.add_argument("--camera-width", type=int, default=1280, help="Requested camera capture width.")
    parser.add_argument("--camera-height", type=int, default=720, help="Requested camera capture height.")
    parser.add_argument("--display-width", type=int, default=1280, help="Displayed window width.")
    parser.add_argument("--display-height", type=int, default=720, help="Displayed window height.")
    parser.add_argument(
        "--display-fit",
        choices=("cover", "contain", "stretch"),
        default="cover",
        help="How the camera image fits the display area.",
    )
    parser.add_argument(
        "--fullscreen",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open the camera window in fullscreen mode. Use --no-fullscreen for a normal window.",
    )
    parser.add_argument(
        "--auto-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically add stable gesture predictions to the translated text.",
    )
    parser.add_argument(
        "--auto-text-cooldown",
        type=float,
        default=1.2,
        help="Minimum seconds before auto-text can add the same gesture again after a hand reset.",
    )
    return parser


def open_camera(camera_index: int) -> cv2.VideoCapture:
    if os.name == "nt":
        return cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    return cv2.VideoCapture(camera_index)


def append_prediction_log(log_path: Path, label: str, confidence: float, text_snapshot: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "label": label,
        "confidence": round(confidence, 4),
        "text_snapshot": text_snapshot,
    }
    pd.DataFrame([row]).to_csv(
        log_path,
        mode="a",
        header=not log_path.exists(),
        index=False,
    )


def resize_for_display(frame, target_width: int, target_height: int, fit_mode: str = "cover"):
    """Resize the camera view for display without distorting it by default."""

    height, width = frame.shape[:2]
    if target_width <= 0 or target_height <= 0:
        return frame

    if fit_mode == "stretch":
        return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    scale_func = max if fit_mode == "cover" else min
    scale = scale_func(target_width / width, target_height / height)
    if abs(scale - 1.0) < 0.01 and fit_mode == "contain":
        return frame

    new_size = (int(width * scale), int(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, new_size, interpolation=interpolation)

    if fit_mode != "cover":
        return resized

    resized_height, resized_width = resized.shape[:2]
    left = max(0, (resized_width - target_width) // 2)
    top = max(0, (resized_height - target_height) // 2)
    return resized[top : top + target_height, left : left + target_width]


def get_text_width(text: str, scale: float, thickness: int = 2) -> int:
    return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]


def split_long_word(word: str, max_width: int, scale: float, thickness: int = 2) -> list[str]:
    chunks: list[str] = []
    current = ""

    for char in word:
        candidate = f"{current}{char}"
        if not current or get_text_width(candidate, scale, thickness) <= max_width:
            current = candidate
        else:
            chunks.append(current)
            current = char

    if current:
        chunks.append(current)

    return chunks


def wrap_text(text: str, max_width: int, scale: float, thickness: int = 2) -> list[str]:
    """Wrap text into multiple lines instead of cutting it with ellipses."""

    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = ""

    for word in words:
        chunks = (
            [word]
            if get_text_width(word, scale, thickness) <= max_width
            else split_long_word(word, max_width, scale, thickness)
        )

        for chunk in chunks:
            candidate = chunk if not current else f"{current} {chunk}"
            if get_text_width(candidate, scale, thickness) <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = chunk

    if current:
        lines.append(current)

    return lines


def draw_text(
    frame,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.65,
    color=(245, 245, 245),
    thickness: int = 1,
) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def format_translated_text(tokens: list[str]) -> str:
    """Join alphabet labels as words while keeping phrase labels readable."""

    chunks: list[str] = []
    letter_buffer: list[str] = []

    def flush_letters() -> None:
        if letter_buffer:
            chunks.append("".join(letter_buffer))
            letter_buffer.clear()

    for token in tokens:
        clean_token = token.strip()
        if not clean_token:
            continue

        if clean_token.upper() == SPACE_TOKEN:
            flush_letters()
            continue

        if len(clean_token) == 1 and clean_token.isalpha():
            letter_buffer.append(clean_token.upper())
        else:
            flush_letters()
            chunks.append(clean_token)

    flush_letters()
    return " ".join(chunks)


def draw_center_output(frame, text: str) -> None:
    """Draw the translated output as the main readable result in the camera view."""

    if not text:
        return

    height, width = frame.shape[:2]
    max_text_width = int(width * 0.86)
    max_text_height = int(height * 0.30)
    scale = 4.0
    line_gap = 24
    lines = [text]
    sizes = []

    while scale >= 0.9:
        thickness = max(2, int(scale * 2))
        lines = wrap_text(text, max_text_width, scale, thickness)
        sizes = [
            cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
            for line in lines
        ]
        line_gap = max(12, int(18 * scale))
        total_height = sum(size[1] for size in sizes) + line_gap * max(0, len(lines) - 1)
        widest_line = max((size[0] for size in sizes), default=0)
        if widest_line <= max_text_width and total_height <= max_text_height:
            break
        scale -= 0.15

    thickness = max(2, int(scale * 2))
    outline_thickness = thickness + 6
    total_height = sum(size[1] for size in sizes) + line_gap * max(0, len(lines) - 1)
    widest_line = max((size[0] for size in sizes), default=0)

    box_padding_x = 42
    box_padding_y = 34
    box_left = max(16, (width - widest_line) // 2 - box_padding_x)
    box_right = min(width - 16, (width + widest_line) // 2 + box_padding_x)
    box_top = max(16, (height - total_height) // 2 - box_padding_y)
    box_bottom = min(height - 16, (height + total_height) // 2 + box_padding_y)

    overlay = frame.copy()
    cv2.rectangle(overlay, (box_left, box_top), (box_right, box_bottom), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.34, frame, 0.66, 0, frame)

    y = (height - total_height) // 2
    for line, size in zip(lines, sizes):
        text_width, text_height = size
        x = (width - text_width) // 2
        y += text_height
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            outline_thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 245, 120),
            thickness,
            cv2.LINE_AA,
        )
        y += line_gap


def get_model_status(predictor: GesturePredictor) -> str:
    if predictor.is_ready:
        label_count = len(predictor.labels)
        if predictor.metadata.get("single_label_demo") or label_count == 1:
            return "Demo model loaded (1 gesture). Tambahkan label lain agar deteksi lebih akurat."
        return f"Model loaded ({label_count} gesture labels)"
    return "Model belum dilatih. Kumpulkan dataset lalu jalankan train_model.py."


def get_handedness_label(result) -> str:
    if not result.multi_handedness:
        return "-"

    classification = result.multi_handedness[0].classification[0]
    label_map = {"Left": "Kiri", "Right": "Kanan"}
    label = label_map.get(classification.label, classification.label)
    return f"{label} ({classification.score:.0%})"


def draw_panel(
    frame,
    prediction: str,
    confidence: float,
    stable_label: Optional[str],
    translated_text: str,
    model_status: str,
    handedness: str,
    action_message: str,
    auto_text: bool,
) -> None:
    stable_text = stable_label if stable_label else "-"
    text_preview = translated_text if translated_text else "-"

    height, width = frame.shape[:2]
    panel_left = 16
    panel_bottom = height - 16
    panel_width = min(width - 32, 560)
    panel_right = panel_left + panel_width
    padding_x = 14
    padding_y = 12
    max_text_width = max(260, panel_width - (padding_x * 2))

    content = [
        ("Sign Language Translator", 0.58, (255, 255, 255), 7, 2),
        (f"Prediction: {prediction} ({confidence:.0%}) | Stable: {stable_text}", 0.45, (225, 255, 225), 4, 1),
        (f"Text: {text_preview}", 0.45, (255, 235, 180), 4, 1),
        (f"Hand: {handedness}", 0.43, (205, 235, 255), 4, 1),
        (model_status, 0.42, (215, 215, 215), 4, 1),
        (action_message, 0.42, (230, 220, 255), 5, 1),
        (
            "AUTO text | ENTER space | SPACE add | BACKSPACE delete | C clear | Q quit"
            if auto_text
            else "ENTER space | SPACE add | BACKSPACE delete | C clear | Q quit",
            0.40,
            (220, 220, 220),
            0,
            1,
        ),
    ]

    line_items: list[tuple[str, float, tuple[int, int, int], int, int, int]] = []
    for text, scale, color, after_gap, thickness in content:
        wrapped_lines = wrap_text(text, max_text_width, scale, thickness)
        for index, line in enumerate(wrapped_lines):
            gap = after_gap if index == len(wrapped_lines) - 1 else 4
            text_height = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][1]
            line_items.append((line, scale, color, gap, text_height, thickness))

    panel_height = padding_y * 2 + sum(text_height + gap for _, _, _, gap, text_height, _ in line_items)
    panel_top = max(16, panel_bottom - panel_height)

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_left, panel_top), (panel_right, panel_bottom), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)

    y = panel_top + padding_y
    for line, scale, color, gap, text_height, thickness in line_items:
        y += text_height
        if y < panel_bottom - 6:
            draw_text(frame, line, (panel_left + padding_x, y), scale, color, thickness)
        y += gap


def main() -> int:
    args = build_parser().parse_args()
    model_path = resolve_project_path(args.model)
    log_path = resolve_project_path(args.log)

    predictor = GesturePredictor(model_path=model_path, confidence_threshold=args.confidence)
    smoother = PredictionSmoother(window_size=args.window, min_votes=args.votes)

    cap = open_camera(args.camera)
    if not cap.isOpened():
        print(
            "Error: Kamera tidak terdeteksi. "
            "Coba tutup aplikasi lain yang memakai kamera atau jalankan: python app.py --camera 1"
        )
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    tokens: list[str] = []
    stable_label: Optional[str] = None
    current_prediction = "No hand"
    current_confidence = 0.0
    current_handedness = "-"
    action_message = (
        "SPACE aktif setelah Stable menampilkan label."
        if predictor.is_ready
        else "Model belum dilatih. Gunakan collect_dataset.py lalu train_model.py."
    )
    last_logged_label: Optional[str] = None
    last_logged_at = 0.0
    last_auto_added_label: Optional[str] = None
    last_auto_added_at = 0.0

    model_status = get_model_status(predictor)
    window_name = "Sign Language Translator"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(window_name, args.display_width, args.display_height)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.70,
        min_tracking_confidence=0.50,
    ) as hands:
        while True:
            success, frame = cap.read()
            if not success:
                print("Error: Frame kamera tidak bisa dibaca.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                current_handedness = get_handedness_label(result)
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

                if predictor.is_ready:
                    prediction = predictor.predict_landmarks(hand_landmarks)
                    current_prediction = prediction.label
                    current_confidence = prediction.confidence
                    stable_prediction = smoother.update(prediction)
                    if stable_prediction:
                        stable_label = stable_prediction

                        now = time.monotonic()
                        if last_auto_added_label is None:
                            can_auto_add = now - last_auto_added_at >= args.auto_text_cooldown
                        else:
                            can_auto_add = stable_prediction != last_auto_added_label
                        if args.auto_text and can_auto_add:
                            tokens.append(stable_prediction)
                            last_auto_added_label = stable_prediction
                            last_auto_added_at = now
                            action_message = f"Teks otomatis: {stable_prediction}"

                        should_log = (
                            stable_label != last_logged_label
                            or now - last_logged_at >= 2.0
                        )
                        if should_log:
                            append_prediction_log(
                                log_path,
                                stable_label,
                                current_confidence,
                                format_translated_text(tokens),
                            )
                            last_logged_label = stable_label
                            last_logged_at = now
                else:
                    current_prediction = "Train model first"
                    current_confidence = 0.0
            else:
                current_prediction = "No hand"
                current_confidence = 0.0
                current_handedness = "-"
                stable_label = None
                last_auto_added_label = None
                smoother.update(PredictionResult("Unknown", 0.0, False, {}))

            frame = resize_for_display(frame, args.display_width, args.display_height, args.display_fit)
            translated_text = format_translated_text(tokens)
            draw_center_output(frame, translated_text)
            draw_panel(
                frame,
                current_prediction,
                current_confidence,
                stable_label,
                translated_text,
                model_status,
                current_handedness,
                action_message,
                args.auto_text,
            )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue

            if key in (ord("q"), 27):
                break
            if key in (13, 10):
                if tokens and tokens[-1] != SPACE_TOKEN:
                    tokens.append(SPACE_TOKEN)
                    action_message = "Spasi ditambahkan."
                else:
                    action_message = "Spasi belum ditambahkan: teks masih kosong atau sudah ada spasi."
                continue
            if key == ord("c"):
                tokens.clear()
                stable_label = None
                last_auto_added_label = None
                last_auto_added_at = 0.0
                action_message = "Teks dibersihkan."
            if key in (8, 127) and tokens:
                tokens.pop()
                action_message = "Kata terakhir dihapus."
            if key == 32 and stable_label:
                tokens.append(stable_label)
                action_message = f"Ditambahkan ke teks: {stable_label}"
            elif key == 32 and not predictor.is_ready:
                action_message = "SPACE belum bisa: model belum dilatih."
            elif key == 32:
                action_message = "SPACE belum bisa: tunggu sampai Stable bukan '-'."

    cap.release()
    cv2.destroyAllWindows()
    print(f"Log prediksi tersimpan di: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
