"""
ML classifier for AI-likelihood.

Pipeline:
    raw text --> [word TF-IDF | char TF-IDF | stylometric features] --> Logistic Regression --> calibrated probability

This is deliberately a lightweight, fast, fully local model (no GPU,
no downloads). It is built so a future version can swap in a
transformer-based encoder without touching the rest of the app: any
class implementing `predict_proba(texts) -> List[float]` and
`is_trained` satisfies the interface the Detector expects.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.config import settings
from app.detector.features import FEATURE_NAMES, extract_features

MODEL_FILENAME = "ai_detector_model.joblib"
METADATA_FILENAME = "ai_detector_metadata.json"


def _style_matrix(texts: Iterable[str]) -> np.ndarray:
    rows = []
    for t in texts:
        bundle = extract_features(t)
        rows.append(bundle.as_vector(FEATURE_NAMES))
    return np.array(rows, dtype=float)


@dataclass
class TrainResult:
    n_samples: int
    n_features: int
    classes: List[str]


class AIDetectorModel:
    """Wraps vectorizers + a Logistic Regression classifier.

    Not tightly coupled to the UI or API: it only knows about text in,
    probabilities out. Swappable with a stronger model later as long as
    the same interface (`predict_proba`, `is_trained`, `save`, `load`)
    is implemented.
    """

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = Path(model_dir or settings.model_dir)
        self.word_vectorizer: Optional[TfidfVectorizer] = None
        self.char_vectorizer: Optional[TfidfVectorizer] = None
        self.classifier: Optional[LogisticRegression] = None
        self.classes_: Optional[List[str]] = None
        self.metadata: dict = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, texts: List[str], labels: List[str]) -> TrainResult:
        self.word_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
        )

        X_word = self.word_vectorizer.fit_transform(texts)
        X_char = self.char_vectorizer.fit_transform(texts)
        X_style = _style_matrix(texts)

        # Normalize stylometric features to a comparable scale to the
        # (already bounded) TF-IDF weights, then combine everything.
        style_max = np.abs(X_style).max(axis=0)
        style_max[style_max == 0] = 1.0
        X_style_norm = X_style / style_max
        self._style_scale = style_max

        X = hstack([X_word, X_char, csr_matrix(X_style_norm)]).tocsr()

        self.classifier = LogisticRegression(max_iter=2000, class_weight="balanced")
        self.classifier.fit(X, labels)
        self.classes_ = list(self.classifier.classes_)

        result = TrainResult(
            n_samples=len(texts),
            n_features=X.shape[1],
            classes=self.classes_,
        )
        self.metadata = {
            "n_samples": result.n_samples,
            "n_features": result.n_features,
            "classes": result.classes,
        }
        return result

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @property
    def is_trained(self) -> bool:
        return self.classifier is not None and self.word_vectorizer is not None

    def _transform(self, texts: List[str]) -> csr_matrix:
        X_word = self.word_vectorizer.transform(texts)
        X_char = self.char_vectorizer.transform(texts)
        X_style = _style_matrix(texts) / self._style_scale
        return hstack([X_word, X_char, csr_matrix(X_style)]).tocsr()

    def predict_proba(self, texts: List[str]) -> List[float]:
        """Returns P(label == 'ai') for each text."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Run scripts/train_model.py first.")
        X = self._transform(texts)
        proba = self.classifier.predict_proba(X)
        ai_index = list(self.classifier.classes_).index("ai")
        return [float(p[ai_index]) for p in proba]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "word_vectorizer": self.word_vectorizer,
                "char_vectorizer": self.char_vectorizer,
                "classifier": self.classifier,
                "style_scale": self._style_scale,
            },
            self.model_dir / MODEL_FILENAME,
        )
        (self.model_dir / METADATA_FILENAME).write_text(json.dumps(self.metadata, indent=2))

    def load(self) -> bool:
        model_path = self.model_dir / MODEL_FILENAME
        if not model_path.exists():
            return False
        data = joblib.load(model_path)
        self.word_vectorizer = data["word_vectorizer"]
        self.char_vectorizer = data["char_vectorizer"]
        self.classifier = data["classifier"]
        self._style_scale = data["style_scale"]
        self.classes_ = list(self.classifier.classes_)
        meta_path = self.model_dir / METADATA_FILENAME
        if meta_path.exists():
            self.metadata = json.loads(meta_path.read_text())
        return True


_singleton: Optional[AIDetectorModel] = None


def get_model() -> AIDetectorModel:
    """Process-wide singleton so the (potentially sizeable) TF-IDF
    vocabulary is loaded from disk only once."""
    global _singleton
    if _singleton is None:
        _singleton = AIDetectorModel()
        _singleton.load()
    return _singleton
