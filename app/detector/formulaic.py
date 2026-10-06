"""
Formulaic language detection.

Important design note: presence of these phrases is a WEAK signal.
Human writers use "furthermore" and "in conclusion" too. This module
never classifies text as AI on its own — it only produces a rate and a
list of matched phrases that the detector combines with everything else
in features.py, and which calibration.py learns a (small) weight for.

The phrase list is intentionally stored as plain data so it can be
edited/extended without touching any logic.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

DEFAULT_PHRASES_PATH = Path(__file__).resolve().parent / "formulaic_phrases.json"

DEFAULT_PHRASES = [
    "it is important to note that",
    "it is worth noting that",
    "in today's world",
    "in today's society",
    "plays a crucial role",
    "plays a vital role",
    "delve into",
    "a wide range of",
    "in conclusion",
    "to summarize",
    "multifaceted",
    "ever-evolving",
    "seamlessly",
    "leverage",
    "robust",
    "comprehensive",
    "furthermore",
    "moreover",
    "additionally",
    "it is essential to",
    "unlock the potential",
    "navigate the complexities",
    "in the realm of",
    "at the end of the day",
    "when it comes to",
    "shed light on",
    "the importance of",
    "cannot be overstated",
    "underscores the importance",
    "fosters a sense of",
    "paves the way for",
    "stands as a testament",
    "in an era where",
    "as we navigate",
    "holistic approach",
    "cutting-edge",
    "game-changer",
    "synergy",
    "testament to",
]


def _ensure_phrase_file() -> Path:
    if not DEFAULT_PHRASES_PATH.exists():
        DEFAULT_PHRASES_PATH.write_text(json.dumps(DEFAULT_PHRASES, indent=2))
    return DEFAULT_PHRASES_PATH


def load_phrases() -> List[str]:
    path = _ensure_phrase_file()
    try:
        data = json.loads(path.read_text())
        if isinstance(data, list) and data:
            return [str(p).lower() for p in data]
    except Exception:
        pass
    return DEFAULT_PHRASES


@dataclass
class FormulaicMatch:
    phrase: str
    sentence: str
    sentence_index: int


def find_formulaic_matches(sentences: List[str], phrases: List[str] | None = None) -> List[FormulaicMatch]:
    phrases = phrases or load_phrases()
    matches: List[FormulaicMatch] = []
    for idx, sentence in enumerate(sentences):
        lower = sentence.lower()
        for phrase in phrases:
            if re.search(rf"\b{re.escape(phrase)}\b", lower):
                matches.append(FormulaicMatch(phrase=phrase, sentence=sentence, sentence_index=idx))
    return matches


def formulaic_rate(sentences: List[str], phrases: List[str] | None = None) -> Tuple[float, List[FormulaicMatch]]:
    """Returns (matches per sentence, match objects)."""
    if not sentences:
        return 0.0, []
    matches = find_formulaic_matches(sentences, phrases)
    return len(matches) / len(sentences), matches
