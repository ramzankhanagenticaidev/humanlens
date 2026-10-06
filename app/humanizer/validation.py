"""
Meaning-preservation validation.

After rewriting, we check that the revised text didn't silently drop or
alter facts. This is intentionally conservative and rule-based (no
paid embedding API required): it extracts concrete, checkable elements
— numbers, dates, URLs, quoted material, capitalized entities — from
both versions and flags anything present in the original but missing
from the rewrite.

Semantic similarity is estimated locally using TF-IDF cosine similarity
(no external embedding API needed). This is a coarse signal, not proof
of meaning equivalence, and is presented to the user as such.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Set

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

URL_RE = re.compile(r"https?://\S+|www\.\S+")
NUMBER_RE = re.compile(r"\b\d[\d,]*\.?\d*%?\b")
DATE_RE = re.compile(
    r"\b(?:\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s*\d{0,4})\b",
    re.IGNORECASE,
)
QUOTE_RE = re.compile(r'"([^"]+)"|\u201c([^\u201d]+)\u201d')
# Simple capitalized-phrase entity heuristic (no spaCy dependency required
# for the base install; good enough to catch names/orgs/places).
ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")

COMMON_SENTENCE_STARTERS = {
    "The", "A", "An", "This", "That", "These", "Those", "It", "In", "On",
    "For", "With", "As", "If", "When", "While", "After", "Before", "But",
    "And", "So", "However", "Furthermore", "Moreover", "Additionally",
}


@dataclass
class ValidationIssue:
    category: str  # "number" | "date" | "url" | "quote" | "entity"
    value: str
    detail: str


@dataclass
class ValidationReport:
    semantic_similarity: float
    issues: List[ValidationIssue] = field(default_factory=list)
    is_safe: bool = True

    def to_dict(self) -> dict:
        return {
            "semantic_similarity": round(self.semantic_similarity, 4),
            "issues": [i.__dict__ for i in self.issues],
            "is_safe": self.is_safe,
        }


def _extract_entities(text: str) -> Set[str]:
    candidates = set(ENTITY_RE.findall(text))
    # Drop single-word matches that are just common sentence starters
    filtered = set()
    for c in candidates:
        if c in COMMON_SENTENCE_STARTERS:
            continue
        filtered.add(c)
    return filtered


def _extract_quotes(text: str) -> Set[str]:
    quotes = set()
    for m in QUOTE_RE.finditer(text):
        val = m.group(1) or m.group(2)
        if val:
            quotes.add(val.strip())
    return quotes


def _semantic_similarity(original: str, revised: str) -> float:
    if not original.strip() or not revised.strip():
        return 0.0
    try:
        vect = TfidfVectorizer(stop_words="english").fit([original, revised])
        matrix = vect.transform([original, revised])
        sim = cosine_similarity(matrix[0], matrix[1])[0][0]
        return float(sim)
    except Exception:
        return 0.0


def validate(original: str, revised: str) -> ValidationReport:
    issues: List[ValidationIssue] = []

    orig_numbers = set(NUMBER_RE.findall(original))
    rev_numbers = set(NUMBER_RE.findall(revised))
    for missing in orig_numbers - rev_numbers:
        issues.append(ValidationIssue("number", missing, "A number from the original text is missing in the rewrite."))

    orig_dates = set(DATE_RE.findall(original))
    rev_dates = set(DATE_RE.findall(revised))
    for missing in orig_dates - rev_dates:
        issues.append(ValidationIssue("date", missing, "A date from the original text is missing in the rewrite."))

    orig_urls = set(URL_RE.findall(original))
    rev_urls = set(URL_RE.findall(revised))
    for missing in orig_urls - rev_urls:
        issues.append(ValidationIssue("url", missing, "A URL from the original text is missing in the rewrite."))

    orig_quotes = _extract_quotes(original)
    rev_quotes = _extract_quotes(revised)
    for missing in orig_quotes - rev_quotes:
        issues.append(ValidationIssue("quote", missing, "Quoted material from the original is missing in the rewrite."))

    orig_entities = _extract_entities(original)
    rev_entities = _extract_entities(revised)
    for missing in orig_entities - rev_entities:
        issues.append(ValidationIssue("entity", missing, "A named entity from the original may be missing in the rewrite."))

    similarity = _semantic_similarity(original, revised)

    # "Safe" heuristic: no hard-fact losses (numbers/dates/urls/quotes),
    # and similarity isn't suspiciously low. Entity-only diffs are
    # flagged but don't block, since our entity heuristic has false positives.
    hard_issue = any(i.category in {"number", "date", "url", "quote"} for i in issues)
    is_safe = (not hard_issue) and similarity > 0.35

    return ValidationReport(semantic_similarity=similarity, issues=issues, is_safe=is_safe)
