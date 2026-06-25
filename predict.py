"""Prediction helpers for the Sign Language Translator project.

This module keeps feature extraction and model inference in one place so the
dataset collector, trainer, and real-time app always use the same input format.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import joblib
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "model" / "sign_language_model.pkl"

NUM_HAND_LANDMARKS = 21
LANDMARK_AXES = ("x", "y", "z")
FEATURE_NAMES = [
    f"{axis}_{landmark_index}"
    for landmark_index in range(NUM_HAND_LANDMARKS)
    for axis in LANDMARK_AXES
]
X_FEATURE_INDICES = tuple(
    index for index, feature_name in enumerate(FEATURE_NAMES) if feature_name.startswith("x_")
)

SUPPORTED_GESTURES = (
    "A",
    "B",
    "C",
    "I Love You",
    "Hello",
    "Thank You",
    "Yes",
    "No",
)


@dataclass(frozen=True)
class PredictionResult:
    """Container returned by GesturePredictor."""

    label: str
    confidence: float
    accepted: bool
    probabilities: Dict[str, float]
    orientation: str = "original"


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve a path relative to the project folder when needed."""

    path = Path(path_value)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def landmarks_to_array(hand_landmarks) -> np.ndarray:
    """Convert MediaPipe hand landmarks into a (21, 3) numpy array."""

    landmarks = hand_landmarks.landmark
    if len(landmarks) != NUM_HAND_LANDMARKS:
        raise ValueError(f"Expected {NUM_HAND_LANDMARKS} hand landmarks, got {len(landmarks)}")

    return np.array([[point.x, point.y, point.z] for point in landmarks], dtype=np.float32)


def normalize_landmarks(landmarks: Sequence[Sequence[float]]) -> np.ndarray:
    """Normalize landmarks so the model focuses on hand pose, not location.

    The wrist landmark becomes the origin. The hand is scaled by its largest
    2D distance from the wrist so samples remain comparable across distances
    from the camera.
    """

    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (NUM_HAND_LANDMARKS, len(LANDMARK_AXES)):
        raise ValueError(
            f"Expected landmark shape {(NUM_HAND_LANDMARKS, len(LANDMARK_AXES))}, got {points.shape}"
        )

    wrist = points[0].copy()
    normalized = points - wrist
    scale = np.linalg.norm(normalized[:, :2], axis=1).max()
    if scale < 1e-6:
        scale = 1.0

    normalized = normalized / scale
    return normalized.astype(np.float32)


def extract_features(hand_landmarks) -> np.ndarray:
    """Create the flat feature vector used by the classifier."""

    raw_landmarks = landmarks_to_array(hand_landmarks)
    normalized_landmarks = normalize_landmarks(raw_landmarks)
    return normalized_landmarks.flatten()


def mirror_features(features: Sequence[float]) -> np.ndarray:
    """Mirror a feature vector horizontally to support left and right hands."""

    mirrored = np.asarray(features, dtype=np.float32).reshape(-1).copy()
    if mirrored.shape[0] != len(FEATURE_NAMES):
        raise ValueError(f"Expected {len(FEATURE_NAMES)} features, got {mirrored.shape[0]}")

    mirrored[list(X_FEATURE_INDICES)] *= -1
    return mirrored


def feature_row_from_landmarks(label: str, hand_landmarks) -> Dict[str, float | str]:
    """Build one CSV-ready dataset row from MediaPipe landmarks."""

    features = extract_features(hand_landmarks)
    row: Dict[str, float | str] = {"label": label}
    row.update({name: float(value) for name, value in zip(FEATURE_NAMES, features)})
    return row


class GesturePredictor:
    """Load a trained model and run gesture prediction on landmark features."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        confidence_threshold: float = 0.65,
        use_mirror_prediction: bool = True,
    ) -> None:
        self.model_path = resolve_project_path(model_path)
        self.confidence_threshold = confidence_threshold
        self.use_mirror_prediction = use_mirror_prediction
        self.model = None
        self.labels: list[str] = []
        self.feature_names = FEATURE_NAMES.copy()
        self.metadata: dict = {}
        self.error: Optional[str] = None
        self.load()

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        """Load model artifact from disk."""

        if not self.model_path.exists():
            self.error = (
                f"Model belum ditemukan di {self.model_path}. "
                "Jalankan train_model.py setelah mengumpulkan dataset."
            )
            return

        try:
            artifact = joblib.load(self.model_path)
        except Exception as exc:  # pragma: no cover - depends on external file state
            self.error = f"Gagal membuka model: {exc}"
            return

        if isinstance(artifact, dict) and "model" in artifact:
            self.model = artifact["model"]
            self.labels = [str(label) for label in artifact.get("labels", [])]
            self.feature_names = artifact.get("feature_names", FEATURE_NAMES.copy())
            self.metadata = artifact.get("metadata", {})
        else:
            self.model = artifact
            if hasattr(self.model, "classes_"):
                self.labels = [str(label) for label in self.model.classes_]

        self.error = None

    def _predict_candidate(self, feature_array: np.ndarray, orientation: str) -> PredictionResult:
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(feature_array)[0]
            classes = [str(label) for label in getattr(self.model, "classes_", self.labels)]
            probability_map = {
                label: float(probability)
                for label, probability in zip(classes, probabilities)
            }
            best_label = max(probability_map, key=probability_map.get)
            confidence = probability_map[best_label]
        else:
            best_label = str(self.model.predict(feature_array)[0])
            confidence = 1.0
            probability_map = {best_label: confidence}

        accepted = confidence >= self.confidence_threshold
        label = best_label if accepted else "Unknown"
        return PredictionResult(label, confidence, accepted, probability_map, orientation)

    def predict_features(self, features: Sequence[float]) -> PredictionResult:
        """Predict one gesture from a flat feature vector.

        The mirrored candidate helps the same model recognize left-hand and
        right-hand versions of a gesture.
        """

        if not self.is_ready:
            return PredictionResult("Model not loaded", 0.0, False, {})

        feature_array = np.asarray(features, dtype=np.float32).reshape(1, -1)
        if feature_array.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, got {feature_array.shape[1]}"
            )

        candidates = [("original", feature_array)]
        if self.use_mirror_prediction:
            mirrored_array = mirror_features(feature_array.ravel()).reshape(1, -1)
            candidates.append(("mirrored", mirrored_array))

        results = [
            self._predict_candidate(candidate_array, orientation)
            for orientation, candidate_array in candidates
        ]
        return max(results, key=lambda result: result.confidence)

    def predict_landmarks(self, hand_landmarks) -> PredictionResult:
        """Extract features from landmarks and predict the gesture label."""

        features = extract_features(hand_landmarks)
        return self.predict_features(features)
