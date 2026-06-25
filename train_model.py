"""Train a gesture classifier from collected MediaPipe landmark data."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from predict import FEATURE_NAMES, SUPPORTED_GESTURES, X_FEATURE_INDICES, resolve_project_path


DATASET_PATH = Path("dataset") / "gesture_dataset.csv"
MODEL_PATH = Path("model") / "sign_language_model.pkl"
REPORT_PATH = Path("model") / "training_report.txt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train sign language gesture classifier.")
    parser.add_argument("--data", type=str, default=str(DATASET_PATH), help="Dataset CSV path.")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH), help="Output model path.")
    parser.add_argument("--report", type=str, default=str(REPORT_PATH), help="Training report path.")
    parser.add_argument("--test-size", type=float, default=0.20, help="Validation split ratio.")
    parser.add_argument("--trees", type=int, default=250, help="Number of Random Forest trees.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--no-mirror-augment",
        action="store_true",
        help="Disable automatic left/right hand mirror augmentation.",
    )
    return parser


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan di {dataset_path}. Jalankan collect_dataset.py terlebih dahulu."
        )

    dataset = pd.read_csv(dataset_path)
    required_columns = {"label", *FEATURE_NAMES}
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset tidak lengkap. Kolom hilang: {missing}")

    dataset = dataset[["label", *FEATURE_NAMES]].dropna()
    if dataset.empty:
        raise ValueError("Dataset kosong setelah membersihkan nilai kosong.")

    return dataset


def can_use_stratified_split(label_counts: pd.Series, sample_count: int, test_size: float) -> bool:
    test_count = int(np.ceil(sample_count * test_size))
    train_count = sample_count - test_count
    return (
        label_counts.min() >= 2
        and test_count >= len(label_counts)
        and train_count >= len(label_counts)
    )


def train_classifier(x_train: np.ndarray, y_train: pd.Series, trees: int, seed: int) -> RandomForestClassifier:
    classifier = RandomForestClassifier(
        n_estimators=trees,
        random_state=seed,
        class_weight="balanced",
        n_jobs=-1,
    )
    classifier.fit(x_train, y_train)
    return classifier


def mirror_feature_matrix(features: np.ndarray) -> np.ndarray:
    """Mirror normalized hand landmark features across the horizontal axis."""

    mirrored = np.asarray(features, dtype=np.float32).copy()
    mirrored[:, list(X_FEATURE_INDICES)] *= -1
    return mirrored


def add_mirror_augmentation(
    features: np.ndarray,
    labels: pd.Series,
) -> tuple[np.ndarray, pd.Series]:
    """Duplicate samples with mirrored x coordinates for right/left hand support."""

    mirrored_features = mirror_feature_matrix(features)
    augmented_features = np.vstack([features, mirrored_features])
    augmented_labels = pd.concat(
        [labels.reset_index(drop=True), labels.reset_index(drop=True)],
        ignore_index=True,
    )
    return augmented_features, augmented_labels


def main() -> int:
    args = build_parser().parse_args()
    dataset_path = resolve_project_path(args.data)
    model_path = resolve_project_path(args.model)
    report_path = resolve_project_path(args.report)

    try:
        dataset = load_dataset(dataset_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    raw_label_counts = dataset["label"].value_counts().sort_index()
    single_label_demo = len(raw_label_counts) == 1
    if single_label_demo:
        only_label = str(raw_label_counts.index[0])
        print(
            "Peringatan: dataset baru memiliki 1 label gesture. "
            f"Model demo akan selalu memprediksi '{only_label}' untuk tangan yang terdeteksi. "
            "Tambahkan label lain agar translator bisa membedakan gesture."
        )

    x = dataset[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = dataset["label"]

    test_size = min(max(args.test_size, 0.10), 0.40)
    use_validation = can_use_stratified_split(raw_label_counts, len(dataset), test_size)

    if use_validation:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=args.seed,
            stratify=y,
        )
        validation_note = "Validation menggunakan stratified train-test split."
    else:
        x_train, y_train = x, y
        x_test, y_test = x, y
        validation_note = (
            "Dataset masih kecil, evaluasi memakai data training. "
            "Tambahkan sample per label untuk validasi yang lebih akurat."
        )

    if single_label_demo:
        validation_note = (
            "Mode demo 1 label: model hanya bisa mengenali satu gesture. "
            "Tambahkan minimal 1 label lain untuk klasifikasi yang bermakna."
        )

    mirror_augmentation = not args.no_mirror_augment
    if mirror_augmentation:
        x_train, y_train = add_mirror_augmentation(x_train, y_train)
        x_test, y_test = add_mirror_augmentation(x_test, y_test)

    classifier = train_classifier(x_train, y_train, args.trees, args.seed)
    predictions = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": classifier,
        "labels": [str(label) for label in classifier.classes_],
        "feature_names": FEATURE_NAMES,
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "dataset_path": str(dataset_path),
            "raw_sample_count": int(len(dataset)),
            "training_sample_count": int(len(x_train)),
            "mirror_augmentation": mirror_augmentation,
            "label_counts": {str(label): int(count) for label, count in raw_label_counts.items()},
            "supported_gesture_targets": list(SUPPORTED_GESTURES),
            "model_type": "RandomForestClassifier",
            "trees": int(args.trees),
            "accuracy": float(accuracy),
            "validation_note": validation_note,
            "single_label_demo": bool(single_label_demo),
        },
    }
    joblib.dump(artifact, model_path)

    report_text = "\n".join(
        [
            "Sign Language Translator - Training Report",
            "=" * 48,
            f"Created at: {artifact['metadata']['created_at']}",
            f"Dataset: {dataset_path}",
            f"Raw samples: {len(dataset)}",
            f"Training samples: {len(x_train)}",
            f"Mirror augmentation: {'enabled' if mirror_augmentation else 'disabled'}",
            "Label counts:",
            raw_label_counts.to_string(),
            "",
            validation_note,
            f"Accuracy: {accuracy:.4f}",
            "",
            report,
        ]
    )
    report_path.write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"Model tersimpan di: {model_path}")
    print(f"Report tersimpan di: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
