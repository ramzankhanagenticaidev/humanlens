"""
Calibration.

Two responsibilities:
1. Apply the user-configurable "sensitivity" setting as a threshold
   shift (NOT a probability rewrite — we never lie about the raw
   score, we only shift where "flagged" begins).
2. Provide a short-text penalty: stylometric signal is noisy on very
   short passages, so we shrink extreme probabilities toward 0.5 when
   there isn't much text to go on, rather than reporting false
   confidence.
"""
from __future__ import annotations


def shrink_for_short_text(probability: float, num_words: int, floor_words: int = 150) -> float:
    """Pulls extreme probabilities toward 0.5 as text gets shorter than
    `floor_words`. At 0 words -> fully shrunk to 0.5. At >= floor_words
    -> unchanged."""
    if num_words >= floor_words:
        return probability
    weight = max(0.0, num_words / floor_words)
    return 0.5 + (probability - 0.5) * weight


def apply_sensitivity(probability: float, sensitivity: float) -> float:
    """`sensitivity` in [0,1], default 0.5 = neutral.
    Higher sensitivity => lower effective threshold => more text gets
    labeled as "AI-leaning" for a given probability. This is done by
    remapping probability through the requested threshold so that
    `sensitivity` acts as the new decision midpoint, without distorting
    the extremes."""
    sensitivity = min(max(sensitivity, 0.05), 0.95)
    if probability <= sensitivity:
        return 0.5 * (probability / sensitivity) if sensitivity > 0 else probability
    else:
        span = 1 - sensitivity
        return 0.5 + 0.5 * ((probability - sensitivity) / span if span > 0 else 0)


def verdict_label(probability: float) -> str:
    if probability >= 0.75:
        return "Likely AI-influenced"
    if probability >= 0.55:
        return "Possibly AI-influenced"
    if probability >= 0.45:
        return "Unclear / mixed signals"
    if probability >= 0.25:
        return "Possibly human-written"
    return "Likely human-written"
