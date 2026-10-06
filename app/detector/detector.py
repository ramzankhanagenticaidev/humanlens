"""
Detector orchestrator.

    text --> FeatureExtractor --> Model --> Calibration --> Explanation --> DetectionResult

This is the single entry point the API/frontend should call. It never
claims certainty, always reports confidence + limitations, and falls
back to a transparent heuristic mode if no trained model is present
(rather than crashing or pretending to have ML it doesn't).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from app.config import settings
from app.detector.calibration import (
    apply_sensitivity,
    shrink_for_short_text,
    verdict_label,
)
from app.detector.classifier import AIDetectorModel, get_model
from app.detector.explain import Signal, build_signals, confidence_label
from app.detector.features import extract_features
from app.detector.formulaic import formulaic_rate


@dataclass
class ParagraphResult:
    index: int
    text: str
    ai_probability: float
    verdict: str


@dataclass
class DetectionResult:
    ai_probability: float
    verdict: str
    confidence: str
    num_words: int
    num_sentences: int
    vocabulary_diversity: float
    supporting_signals: List[Signal]
    counter_signals: List[Signal]
    formulaic_matches: List[dict]
    paragraphs: List[ParagraphResult]
    model_used: str  # "ml_model" or "heuristic_fallback"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ai_probability": round(self.ai_probability, 4),
            "verdict": self.verdict,
            "confidence": self.confidence,
            "num_words": self.num_words,
            "num_sentences": self.num_sentences,
            "vocabulary_diversity": round(self.vocabulary_diversity, 4),
            "supporting_signals": [s.__dict__ for s in self.supporting_signals],
            "counter_signals": [s.__dict__ for s in self.counter_signals],
            "formulaic_matches": self.formulaic_matches,
            "paragraphs": [p.__dict__ for p in self.paragraphs],
            "model_used": self.model_used,
            "warnings": self.warnings,
        }


def _heuristic_probability(features_values: dict, f_rate: float) -> float:
    """Transparent, non-ML fallback used only if no trained model file
    exists yet. Weighted sum of normalized signals -> clearly documented,
    never described as "machine learning" anywhere in the UI."""
    score = 0.5
    cv = features_values.get("sentence_len_cv", 0.45)
    score += (0.45 - cv) * 0.4  # low variance -> more AI-leaning

    mattr = features_values.get("moving_avg_ttr", 0.65)
    score += (0.65 - mattr) * 0.5

    score += min(f_rate, 0.5) * 0.5

    bigram = features_values.get("bigram_repetition_rate", 0.0)
    score += min(bigram, 0.3) * 0.3

    contraction = features_values.get("contraction_rate", 0.0)
    score -= min(contraction, 0.5) * 0.2

    return min(max(score, 0.0), 1.0)


class Detector:
    def __init__(self, model: Optional[AIDetectorModel] = None):
        self.model = model or get_model()

    def _score_text(self, text: str) -> float:
        """Raw (pre-calibration) probability that text is AI-leaning."""
        if self.model.is_trained:
            return self.model.predict_proba([text])[0]
        bundle = extract_features(text)
        rate, _ = formulaic_rate(bundle.sentences)
        return _heuristic_probability(bundle.values, rate)

    def analyze(self, text: str, sensitivity: Optional[float] = None) -> DetectionResult:
        sensitivity = settings.detector_sensitivity if sensitivity is None else sensitivity
        warnings: List[str] = []

        bundle = extract_features(text)
        num_words = int(bundle.values.get("num_words", 0))
        num_sentences = int(bundle.values.get("num_sentences", 0))

        if num_words < settings.min_words_for_analysis:
            warnings.append(
                f"Text is shorter than the recommended minimum of "
                f"{settings.min_words_for_analysis} words. Results may be unreliable."
            )

        f_rate, matches = formulaic_rate(bundle.sentences)

        raw_prob = self._score_text(text)
        shrunk = shrink_for_short_text(raw_prob, num_words)
        calibrated = apply_sensitivity(shrunk, sensitivity)

        supporting, counter = build_signals(bundle, f_rate)
        confidence = confidence_label(calibrated, num_words)

        paragraphs: List[ParagraphResult] = []
        for i, para in enumerate(bundle.paragraphs):
            p_words = len(para.split())
            if p_words < 15:
                # too short to score meaningfully on its own
                paragraphs.append(
                    ParagraphResult(index=i, text=para, ai_probability=0.5, verdict="Too short to assess")
                )
                continue
            p_raw = self._score_text(para)
            p_shrunk = shrink_for_short_text(p_raw, p_words, floor_words=80)
            p_calibrated = apply_sensitivity(p_shrunk, sensitivity)
            paragraphs.append(
                ParagraphResult(
                    index=i,
                    text=para,
                    ai_probability=round(p_calibrated, 4),
                    verdict=verdict_label(p_calibrated),
                )
            )

        return DetectionResult(
            ai_probability=calibrated,
            verdict=verdict_label(calibrated),
            confidence=confidence,
            num_words=num_words,
            num_sentences=num_sentences,
            vocabulary_diversity=bundle.values.get("moving_avg_ttr", 0.0),
            supporting_signals=supporting,
            counter_signals=counter,
            formulaic_matches=[m.__dict__ for m in matches[:15]],
            paragraphs=paragraphs,
            model_used="ml_model" if self.model.is_trained else "heuristic_fallback",
            warnings=warnings,
        )


_detector_singleton: Optional[Detector] = None


def get_detector() -> Detector:
    global _detector_singleton
    if _detector_singleton is None:
        _detector_singleton = Detector()
    return _detector_singleton
