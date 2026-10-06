"""
Explainability layer.

Converts numeric features + model output into plain-language supporting
signals and counter-signals. Deliberately conservative in wording:
signals describe *patterns*, never authorship claims.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from app.detector.features import FeatureBundle


@dataclass
class Signal:
    label: str
    detail: str
    strength: float  # 0..1, roughly how unusual/notable this reading is


# Each rule: (feature getter, is_ai_leaning_if_true, label, detail template)
def _rule_high_sentence_uniformity(f: FeatureBundle) -> Signal | None:
    cv = f.values.get("sentence_len_cv", 1.0)
    if cv < 0.35:
        strength = max(0.0, min(1.0, (0.35 - cv) / 0.35))
        return Signal(
            "Unusually consistent sentence lengths",
            f"Sentence-length variation (coefficient of variation) is {cv:.2f}, "
            "lower than typical human writing, which tends to mix short and long sentences.",
            strength,
        )
    return None


def _rule_varied_sentence_lengths(f: FeatureBundle) -> Signal | None:
    cv = f.values.get("sentence_len_cv", 0.0)
    if cv > 0.55:
        strength = max(0.0, min(1.0, (cv - 0.55) / 0.6))
        return Signal(
            "Natural sentence-length variation",
            f"Sentence lengths vary considerably (CV = {cv:.2f}), which is more typical of human writing.",
            strength,
        )
    return None


def _rule_formulaic_density(rate: float) -> Signal | None:
    if rate > 0.15:
        strength = max(0.0, min(1.0, rate / 0.5))
        return Signal(
            "High formulaic phrase frequency",
            f"About {rate:.2f} formulaic/generic phrases were found per sentence on average "
            "(e.g. transition clichés, hedging phrases such as \"it is important to note that\").",
            strength,
        )
    return None


def _rule_low_formulaic_density(rate: float) -> Signal | None:
    if rate < 0.03:
        return Signal(
            "Low formulaic phrase usage",
            "Very few generic/formulaic phrases were detected.",
            0.3,
        )
    return None


def _rule_repetition(f: FeatureBundle) -> Signal | None:
    bigram = f.values.get("bigram_repetition_rate", 0.0)
    if bigram > 0.12:
        strength = max(0.0, min(1.0, bigram / 0.3))
        return Signal(
            "Repeated phrase structures",
            f"{bigram*100:.0f}% of word pairs (bigrams) recur elsewhere in the text, "
            "suggesting repetitive phrasing.",
            strength,
        )
    return None


def _rule_vocab_diversity_high(f: FeatureBundle) -> Signal | None:
    mattr = f.values.get("moving_avg_ttr", 0.0)
    if mattr > 0.72:
        strength = max(0.0, min(1.0, (mattr - 0.72) / 0.2))
        return Signal(
            "High vocabulary diversity",
            f"Moving-average type-token ratio is {mattr:.2f}, indicating varied word choice.",
            strength,
        )
    return None


def _rule_vocab_diversity_low(f: FeatureBundle) -> Signal | None:
    mattr = f.values.get("moving_avg_ttr", 1.0)
    if mattr < 0.55:
        strength = max(0.0, min(1.0, (0.55 - mattr) / 0.25))
        return Signal(
            "Low structural/vocabulary variation",
            f"Moving-average type-token ratio is {mattr:.2f}, lower than typical, "
            "indicating relatively repetitive word choice.",
            strength,
        )
    return None


def _rule_contractions(f: FeatureBundle) -> Signal | None:
    rate = f.values.get("contraction_rate", 0.0)
    if rate > 0.15:
        return Signal(
            "Frequent contraction usage",
            f"Contractions (e.g. \"don't\", \"it's\") appear roughly {rate:.2f} times per sentence, "
            "a pattern more common in informal human writing.",
            min(1.0, rate),
        )
    return None


def _rule_no_contractions(f: FeatureBundle) -> Signal | None:
    rate = f.values.get("contraction_rate", 1.0)
    words = f.values.get("num_words", 0)
    if rate == 0.0 and words > 80:
        return Signal(
            "No contractions used",
            "The text avoids contractions entirely despite its length, "
            "a pattern sometimes associated with more formal or generated text.",
            0.35,
        )
    return None


def _rule_uniform_openings(f: FeatureBundle) -> Signal | None:
    rate = f.values.get("unique_sentence_opening_rate", 1.0)
    n_sent = f.values.get("num_sentences", 0)
    if rate < 0.6 and n_sent >= 5:
        strength = max(0.0, min(1.0, (0.6 - rate) / 0.6))
        return Signal(
            "Repetitive sentence openings",
            f"Only {rate*100:.0f}% of sentences start with a unique first word, "
            "suggesting repeated sentence-opening patterns.",
            strength,
        )
    return None


def _rule_transition_density(f: FeatureBundle) -> Signal | None:
    rate = f.values.get("transition_word_rate", 0.0)
    if rate > 0.3:
        strength = max(0.0, min(1.0, rate / 0.6))
        return Signal(
            "Frequent transition words",
            f"Transition words (e.g. \"furthermore\", \"moreover\") appear at a rate of "
            f"{rate:.2f} per sentence, notably above typical usage. "
            "Note: transition words alone are weak evidence and common in formal human writing too.",
            strength * 0.6,  # deliberately down-weighted per product philosophy
        )
    return None


SUPPORTING_RULES = [
    _rule_high_sentence_uniformity,
    _rule_repetition,
    _rule_vocab_diversity_low,
    _rule_uniform_openings,
    _rule_no_contractions,
    _rule_transition_density,
]

COUNTER_RULES = [
    _rule_varied_sentence_lengths,
    _rule_vocab_diversity_high,
    _rule_contractions,
    _rule_low_formulaic_density,
]


def build_signals(
    features: FeatureBundle, formulaic_rate: float
) -> Tuple[List[Signal], List[Signal]]:
    """Returns (supporting_signals, counter_signals), sorted by strength."""
    supporting: List[Signal] = []
    counter: List[Signal] = []

    for rule in SUPPORTING_RULES:
        sig = rule(features)
        if sig:
            supporting.append(sig)

    formulaic_sig = _rule_formulaic_density(formulaic_rate)
    if formulaic_sig:
        supporting.append(formulaic_sig)

    for rule in COUNTER_RULES:
        sig = rule(features) if rule is not _rule_low_formulaic_density else rule(formulaic_rate)
        if sig:
            counter.append(sig)

    supporting.sort(key=lambda s: s.strength, reverse=True)
    counter.sort(key=lambda s: s.strength, reverse=True)
    return supporting, counter


def confidence_label(probability: float, num_words: int) -> str:
    """Confidence reflects both how far from 0.5 the score is AND how
    much text we had to work with — short texts get capped confidence
    regardless of how extreme the score looks, because short-text
    stylometry is noisy."""
    distance_from_mid = abs(probability - 0.5) * 2  # 0..1

    if num_words < 60:
        length_cap = "Low"
    elif num_words < 150:
        length_cap = "Moderate"
    else:
        length_cap = "High"

    if distance_from_mid < 0.25:
        score_level = "Low"
    elif distance_from_mid < 0.55:
        score_level = "Moderate"
    else:
        score_level = "High"

    order = {"Low": 0, "Moderate": 1, "High": 2}
    return score_level if order[score_level] <= order[length_cap] else length_cap
