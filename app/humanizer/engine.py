"""
Humanizer engine orchestrator.

    text --> [Ollama rewrite (optional)  OR  deterministic rule pipeline] --> validation --> HumanizeResult

Design goals (see product philosophy in the spec):
  - never silently invent facts
  - always show a real diff of what changed and why
  - always run meaning-preservation validation, even when using an LLM
  - degrade gracefully if Ollama is unavailable
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from app.detector.features import _sentences  # reuse the same tokenizer
from app.humanizer.ollama_client import get_ollama_client
from app.humanizer.rules import (
    Change,
    flag_long_sentences,
    remove_redundant_qualifiers,
    simplify_verbosity,
    strip_throat_clearers,
    vary_repeated_openers,
)
from app.humanizer.validation import ValidationReport, validate


@dataclass
class HumanizeResult:
    original_text: str
    revised_text: str
    changes: List[Change]
    validation: ValidationReport
    engine_used: str  # "ollama" or "rule_based"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "original_text": self.original_text,
            "revised_text": self.revised_text,
            "changes": [c.__dict__ for c in self.changes],
            "validation": self.validation.to_dict(),
            "engine_used": self.engine_used,
            "warnings": self.warnings,
        }


def _rule_based_rewrite(text: str) -> tuple[str, List[Change]]:
    all_changes: List[Change] = []

    step1, c1 = strip_throat_clearers(text)
    all_changes += c1

    step2, c2 = simplify_verbosity(step1)
    all_changes += c2

    step3, c3 = remove_redundant_qualifiers(step2)
    all_changes += c3

    sentences = _sentences(step3)
    varied_sentences, c4 = vary_repeated_openers(sentences)
    all_changes += c4
    step4 = " ".join(varied_sentences) if varied_sentences else step3

    c5 = flag_long_sentences(sentences)
    all_changes += c5  # flags only, don't change text

    # Final cleanup: our substitutions can lowercase what was a
    # sentence-initial word (e.g. "Utilizing" -> "using"). Re-capitalize
    # the start of the text and the start of every sentence.
    final_text = re.sub(
        r"(^\s*|[.!?]\s+)([a-z])",
        lambda m: m.group(1) + m.group(2).upper(),
        step4,
    )
    final_text = re.sub(r"\s{2,}", " ", final_text).strip()

    return final_text, all_changes


class HumanizerEngine:
    def __init__(self, use_ollama: Optional[bool] = None):
        self.use_ollama = use_ollama

    def humanize(self, text: str, use_ollama: Optional[bool] = None, ollama_model: Optional[str] = None) -> HumanizeResult:
        warnings: List[str] = []
        want_ollama = self.use_ollama if use_ollama is None else use_ollama

        revised_text: Optional[str] = None
        engine_used = "rule_based"
        changes: List[Change] = []

        if want_ollama:
            client = get_ollama_client()
            result = client.rewrite(text, model=ollama_model)
            if result.ok:
                revised_text = result.output
                engine_used = "ollama"
            else:
                warnings.append(f"{result.error} Falling back to rule-based rewriting.")

        if revised_text is None:
            revised_text, changes = _rule_based_rewrite(text)

        report = validate(text, revised_text)
        if not report.is_safe:
            warnings.append(
                "Meaning-preservation check flagged possible content loss or low similarity. "
                "Review the rewrite carefully before using it."
            )

        return HumanizeResult(
            original_text=text,
            revised_text=revised_text,
            changes=changes,
            validation=report,
            engine_used=engine_used,
            warnings=warnings,
        )


_engine_singleton: Optional[HumanizerEngine] = None


def get_humanizer_engine() -> HumanizerEngine:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = HumanizerEngine()
    return _engine_singleton
