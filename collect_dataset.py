"""Collect hand gesture landmark samples from a webcam.

Usage examples:
    python collect_dataset.py --label A --samples 100
    python collect_dataset.py --label "I Love You" --samples 120 --auto
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd

from predict import FEATURE_NAMES, SUPPORTED_GESTURES, feature_row_from_landmarks, resolve_project_path


DATASET_PATH = Path("dataset") / "gesture_dataset.csv"
NOTIFICATION_DURATION = 1.6


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect sign language gesture samples.")
    parser.add_argument("--label", type=str, help="Gesture label, for example A or Hello.")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to collect.")
    parser.add_argument("--output", type=str, default=str(DATASET_PATH), help="Output CSV path.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index. Try 1 if 0 fails.")
    parser.add_argument("--auto", action="store_true", help="Capture samples automatically.")
    parser.add_argument(
        "--interval",
        type=float,
        default=0.25,
        help="Seconds between automatic captures when --auto is active.",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.70,
        help="Minimum MediaPipe hand detection confidence.",
    )
    parser.add_argument("--camera-width", type=int, default=640, help="Requested camera capture width.")
    parser.add_argument("--camera-height", type=int, default=480, help="Requested camera capture height.")
    parser.add_argument("--display-width", type=int, default=1280, help="Maximum displayed window width.")
    parser.add_argument("--display-height", type=int, default=720, help="Maximum displayed window height.")
    return parser


def open_camera(camera_index: int) -> cv2.VideoCapture:
    if os.name == "nt":
        return cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    return cv2.VideoCapture(camera_index)


def save_sample(output_path: Path, row: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_columns = ["label", *FEATURE_NAMES]
    pd.DataFrame([row], columns=ordered_columns).to_csv(
        output_path,
        mode="a",
        header=not output_path.exists(),
        index=False,
    )


def resize_for_display(frame, max_width: int, max_height: int):
    """Resize the camera view to a large display size without stretching it."""

    height, width = frame.shape[:2]
    if max_width <= 0 or max_height <= 0:
        return frame

    scale = min(max_width / width, max_height / height)
    if abs(scale - 1.0) < 0.01:
        return frame

    new_size = (int(width * scale), int(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(frame, new_size, interpolation=interpolation)


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


def draw_panel(frame, lines: list[str]) -> None:
    height, width = frame.shape[:2]
    panel_left = 10
    panel_top = 10
    panel_right = width - 10
    padding_x = 18
    padding_y = 16
    max_text_width = max(260, panel_right - panel_left - (padding_x * 2))

    line_items: list[tuple[str, int]] = []
    for text in lines:
        wrapped_lines = wrap_text(text, max_text_width, 0.68)
        for index, line in enumerate(wrapped_lines):
            gap = 8 if index == len(wrapped_lines) - 1 else 4
            line_items.append((line, gap))

    panel_height = padding_y * 2
    for line, gap in line_items:
        text_height = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.68, 2)[0][1]
        panel_height += text_height + gap

    panel_bottom = min(panel_top + panel_height, height - 10)
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_left, panel_top), (panel_right, panel_bottom), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    y = panel_top + padding_y
    for text, gap in line_items:
        text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.68, 2)[0][1]
        y += text_height
        if y >= panel_bottom - 6:
            break
        cv2.putText(
            frame,
            text,
            (panel_left + padding_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
        y += gap


def draw_notification(frame, message: str, success: bool = True) -> None:
    """Show a short feedback message after saving or failing to save a sample."""

    if not message:
        return

    height, width = frame.shape[:2]
    panel_left = 10
    panel_right = width - 10
    panel_bottom = height - 14
    panel_top = max(10, panel_bottom - 58)
    color = (35, 120, 55) if success else (45, 45, 160)

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_left, panel_top), (panel_right, panel_bottom), color, -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    max_text_width = max(260, panel_right - panel_left - 36)
    lines = wrap_text(message, max_text_width, 0.68)
    y = panel_top + 23
    for line in lines[:2]:
        cv2.putText(
            frame,
            line,
            (panel_left + 18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 24


def get_handedness_label(result) -> str:
    if not result.multi_handedness:
        return "-"

    classification = result.multi_handedness[0].classification[0]
    label_map = {"Left": "Kiri", "Right": "Kanan"}
    label = label_map.get(classification.label, classification.label)
    return f"{label} ({classification.score:.0%})"


def get_label(label_arg: str | None) -> str:
    if label_arg:
        return label_arg.strip()

    print("Gesture awal yang disarankan:")
    for gesture in SUPPORTED_GESTURES:
        print(f"- {gesture}")

    label = input("Masukkan label gesture yang ingin dikumpulkan: ").strip()
    if not label:
        raise ValueError("Label tidak boleh kosong.")
    return label


def main() -> int:
    args = build_parser().parse_args()
    output_path = resolve_project_path(args.output)

    try:
        label = get_label(args.label)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    if args.samples <= 0:
        print("Error: --samples harus lebih dari 0.")
        return 1

    cap = open_camera(args.camera)
    if not cap.isOpened():
        print(
            "Error: Kamera tidak terdeteksi. "
            "Coba tutup aplikasi lain yang memakai kamera atau gunakan --camera 1."
        )
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    saved_count = 0
    last_auto_capture = 0.0
    notification_message = "Siap. Tampilkan satu tangan, lalu tekan SPACE untuk menyimpan sample."
    notification_success = True
    notification_until = time.monotonic() + NOTIFICATION_DURATION

    print("Kamera aktif.")
    print("Tekan SPACE untuk menyimpan sample, atau Q untuk keluar.")
    window_name = "Collect Dataset - Sign Language Translator"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=0.50,
    ) as hands:
        while saved_count < args.samples:
            success, frame = cap.read()
            if not success:
                print("Error: Frame kamera tidak bisa dibaca.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            hand_landmarks = None
            handedness = "-"
            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                handedness = get_handedness_label(result)
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

            if args.auto and hand_landmarks is not None:
                now = time.monotonic()
                if now - last_auto_capture >= args.interval:
                    row = feature_row_from_landmarks(label, hand_landmarks)
                    save_sample(output_path, row)
                    saved_count += 1
                    last_auto_capture = now
                    notification_message = (
                        f"Sample otomatis tersimpan untuk '{label}': {saved_count}/{args.samples}"
                    )
                    notification_success = True
                    notification_until = now + NOTIFICATION_DURATION

            status = (
                f"Hand detected: {handedness}"
                if hand_landmarks is not None
                else "Show one hand to camera"
            )
            mode = "AUTO" if args.auto else "MANUAL"
            frame = resize_for_display(frame, args.display_width, args.display_height)
            draw_panel(
                frame,
                [
                    f"Collect Dataset - {mode}",
                    f"Label: {label}",
                    f"Saved: {saved_count}/{args.samples}",
                    status,
                    "SPACE: save sample | Q: quit",
                ],
            )
            if time.monotonic() <= notification_until:
                draw_notification(frame, notification_message, notification_success)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == 27:
                break
            if key == 32 and hand_landmarks is not None:
                row = feature_row_from_landmarks(label, hand_landmarks)
                save_sample(output_path, row)
                saved_count += 1
                notification_message = f"Sample tersimpan untuk '{label}': {saved_count}/{args.samples}"
                notification_success = True
                notification_until = time.monotonic() + NOTIFICATION_DURATION
                print(notification_message)
                if saved_count >= args.samples:
                    draw_notification(frame, notification_message, notification_success)
                    cv2.imshow(window_name, frame)
                    cv2.waitKey(900)
                    break
            elif key == 32:
                notification_message = "Sample belum disimpan: tangan belum terdeteksi kamera."
                notification_success = False
                notification_until = time.monotonic() + NOTIFICATION_DURATION

    cap.release()
    cv2.destroyAllWindows()
    print(f"Selesai. {saved_count} sample tersimpan di {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
