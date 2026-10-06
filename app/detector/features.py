"""
Statistical + stylometric feature extraction.

This module turns raw text into a fixed-size vector of numeric,
human-interpretable features. These features are used both:
  1. as direct inputs to the ML classifier (alongside TF-IDF), and
  2. to generate human-readable "supporting signals" / "counter-signals"
     for the explainability layer.

No feature here is a black box — every value can be traced back to a
concrete, well-known stylometric measurement.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List

import nltk

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:  # pragma: no cover - first-run convenience
    nltk.download("punkt_tab", quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize

CONTRACTIONS = re.compile(
    r"\b(\w+)'(t|re|ve|ll|d|s|m)\b", re.IGNORECASE
)

TRANSITION_WORDS = {
    "additionally", "moreover", "furthermore", "however", "therefore",
    "consequently", "nevertheless", "nonetheless", "meanwhile", "thus",
    "hence", "accordingly", "subsequently", "conversely", "similarly",
    "likewise", "in addition", "on the other hand", "as a result",
    "in conclusion", "in summary", "overall", "ultimately",
}

FUNCTION_WORDS = {
    "the", "a", "an", "of", "to", "in", "on", "at", "for", "with", "by",
    "and", "or", "but", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "as", "if", "than",
    "so", "not", "no", "which", "who", "whom", "what", "when", "where",
}


def _split_paragraphs(text: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    if not parts and text.strip():
        parts = [text.strip()]
    return parts


def _words(text: str) -> List[str]:
    tokens = word_tokenize(text)
    return [t for t in tokens if re.search(r"[A-Za-z0-9]", t)]


def _sentences(text: str) -> List[str]:
    try:
        return [s.strip() for s in sent_tokenize(text) if s.strip()]
    except Exception:
        # fall back to a naive splitter if punkt data is unavailable
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


@dataclass
class FeatureBundle:
    """A fixed, named set of numeric features plus the raw components
    needed to build human-readable explanations later."""

    values: Dict[str, float] = field(default_factory=dict)
    sentences: List[str] = field(default_factory=list)
    paragraphs: List[str] = field(default_factory=list)
    words: List[str] = field(default_factory=list)

    def as_vector(self, ordered_keys: List[str]) -> List[float]:
        return [self.values.get(k, 0.0) for k in ordered_keys]


# The canonical, ordered list of numeric feature names produced below.
# Keeping this explicit (rather than relying on dict ordering) keeps the
# ML feature vector stable across runs/versions.
FEATURE_NAMES: List[str] = [
    "num_words",
    "num_sentences",
    "num_paragraphs",
    "avg_sentence_len",
    "sentence_len_stdev",
    "sentence_len_cv",  # coefficient of variation = stdev / mean
    "avg_paragraph_len_sentences",
    "type_token_ratio",
    "moving_avg_ttr",
    "avg_word_len",
    "word_len_stdev",
    "punctuation_rate",
    "punctuation_diversity",
    "contraction_rate",
    "transition_word_rate",
    "function_word_rate",
    "repeated_word_rate",
    "bigram_repetition_rate",
    "trigram_repetition_rate",
    "unique_sentence_opening_rate",
    "long_word_rate",  # words with 7+ chars
    "comma_per_sentence",
    "flesch_reading_ease",
]


def _safe_std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def _moving_average_ttr(words: List[str], window: int = 50) -> float:
    """MATTR: reduces the bias plain TTR has toward short texts."""
    words_lower = [w.lower() for w in words]
    n = len(words_lower)
    if n == 0:
        return 0.0
    if n <= window:
        return len(set(words_lower)) / n
    ratios = []
    for i in range(0, n - window + 1):
        chunk = words_lower[i : i + window]
        ratios.append(len(set(chunk)) / window)
    return sum(ratios) / len(ratios)


def _ngram_repetition_rate(words: List[str], n: int) -> float:
    words_lower = [w.lower() for w in words]
    if len(words_lower) < n + 1:
        return 0.0
    grams = [tuple(words_lower[i : i + n]) for i in range(len(words_lower) - n + 1)]
    counts = Counter(grams)
    repeated = sum(c for c in counts.values() if c > 1)
    return repeated / len(grams)


def _flesch_reading_ease(text: str) -> float:
    try:
        import textstat

        return float(textstat.flesch_reading_ease(text))
    except Exception:
        return 0.0


def extract_features(text: str) -> FeatureBundle:
    """Compute the full statistical + stylometric feature set for a text."""
    text = text or ""
    sentences = _sentences(text)
    paragraphs = _split_paragraphs(text)
    words = _words(text)
    words_lower = [w.lower() for w in words]

    num_words = len(words)
    num_sentences = max(len(sentences), 1)
    num_paragraphs = max(len(paragraphs), 1)

    sentence_lengths = [len(_words(s)) for s in sentences] or [0]
    avg_sentence_len = sum(sentence_lengths) / len(sentence_lengths)
    sentence_len_stdev = _safe_std([float(x) for x in sentence_lengths])
    sentence_len_cv = (sentence_len_stdev / avg_sentence_len) if avg_sentence_len else 0.0

    para_sentence_counts = [len(_sentences(p)) for p in paragraphs] or [0]
    avg_paragraph_len_sentences = sum(para_sentence_counts) / len(para_sentence_counts)

    unique_words = set(words_lower)
    type_token_ratio = (len(unique_words) / num_words) if num_words else 0.0
    moving_avg_ttr = _moving_average_ttr(words)

    word_lengths = [len(w) for w in words] or [0]
    avg_word_len = sum(word_lengths) / len(word_lengths)
    word_len_stdev = _safe_std([float(x) for x in word_lengths])
    long_word_rate = (sum(1 for w in words if len(w) >= 7) / num_words) if num_words else 0.0

    punctuation_chars = re.findall(r"[.,;:!?\-\u2014()\"']", text)
    punctuation_rate = (len(punctuation_chars) / num_words) if num_words else 0.0
    punctuation_diversity = (
        len(set(punctuation_chars)) / len(punctuation_chars) if punctuation_chars else 0.0
    )
    comma_per_sentence = text.count(",") / num_sentences

    contractions_found = CONTRACTIONS.findall(text)
    contraction_rate = (len(contractions_found) / num_sentences) if num_sentences else 0.0

    transition_hits = 0
    lower_text = text.lower()
    for phrase in TRANSITION_WORDS:
        transition_hits += len(re.findall(rf"\b{re.escape(phrase)}\b", lower_text))
    transition_word_rate = transition_hits / num_sentences

    function_hits = sum(1 for w in words_lower if w in FUNCTION_WORDS)
    function_word_rate = (function_hits / num_words) if num_words else 0.0

    word_counts = Counter(words_lower)
    repeated_words = sum(c for w, c in word_counts.items() if c > 1 and len(w) > 3)
    repeated_word_rate = (repeated_words / num_words) if num_words else 0.0

    bigram_repetition_rate = _ngram_repetition_rate(words, 2)
    trigram_repetition_rate = _ngram_repetition_rate(words, 3)

    openings = []
    for s in sentences:
        w = _words(s)
        if w:
            openings.append(w[0].lower())
    unique_sentence_opening_rate = (
        len(set(openings)) / len(openings) if openings else 1.0
    )

    flesch = _flesch_reading_ease(text)

    values = {
        "num_words": float(num_words),
        "num_sentences": float(num_sentences),
        "num_paragraphs": float(num_paragraphs),
        "avg_sentence_len": avg_sentence_len,
        "sentence_len_stdev": sentence_len_stdev,
        "sentence_len_cv": sentence_len_cv,
        "avg_paragraph_len_sentences": avg_paragraph_len_sentences,
        "type_token_ratio": type_token_ratio,
        "moving_avg_ttr": moving_avg_ttr,
        "avg_word_len": avg_word_len,
        "word_len_stdev": word_len_stdev,
        "punctuation_rate": punctuation_rate,
        "punctuation_diversity": punctuation_diversity,
        "contraction_rate": contraction_rate,
        "transition_word_rate": transition_word_rate,
        "function_word_rate": function_word_rate,
        "repeated_word_rate": repeated_word_rate,
        "bigram_repetition_rate": bigram_repetition_rate,
        "trigram_repetition_rate": trigram_repetition_rate,
        "unique_sentence_opening_rate": unique_sentence_opening_rate,
        "long_word_rate": long_word_rate,
        "comma_per_sentence": comma_per_sentence,
        "flesch_reading_ease": flesch,
    }

    return FeatureBundle(values=values, sentences=sentences, paragraphs=paragraphs, words=words)
