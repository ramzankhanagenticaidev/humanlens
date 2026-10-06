"""
Paraphrasing modes.

Each mode is a distinct rewriting *goal*, not just a different synonym
aggressiveness slider. Modelled loosely on QuillBot's mode set. Each
mode defines:
  - an Ollama prompt (used when a local LLM is available/selected)
  - rule-based fallback parameters (used when it isn't)

so behavior is consistent in spirit regardless of which backend runs.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParaphraseMode:
    key: str
    label: str
    description: str
    ollama_instruction: str
    # Rule-based fallback knobs:
    synonym_rate: float       # 0..1, rough fraction of eligible content words to consider swapping
    prefer_shorter_synonyms: bool
    allow_contractions: bool
    target_formality: str     # "casual" | "neutral" | "formal"
    length_bias: float        # <1.0 shortens, 1.0 neutral, >1.0 tends to expand (informs sentence-combining/splitting)


MODES = {
    "standard": ParaphraseMode(
        key="standard",
        label="Standard",
        description="Balanced rewrite: clearer wording, varied sentence structure, same meaning and length.",
        ollama_instruction=(
            "Rewrite the following text in a balanced, natural way. Vary sentence structure and word "
            "choice where it improves clarity, but keep the tone, meaning, and approximate length the same."
        ),
        synonym_rate=0.35,
        prefer_shorter_synonyms=False,
        allow_contractions=True,
        target_formality="neutral",
        length_bias=1.0,
    ),
    "fluency": ParaphraseMode(
        key="fluency",
        label="Fluency",
        description="Light touch: fixes awkward phrasing and grammar without changing much else.",
        ollama_instruction=(
            "Lightly polish the following text: fix awkward phrasing, grammar, and word choice issues "
            "only. Make the smallest changes necessary. Do not rewrite sentences that are already clear."
        ),
        synonym_rate=0.12,
        prefer_shorter_synonyms=False,
        allow_contractions=True,
        target_formality="neutral",
        length_bias=1.0,
    ),
    "formal": ParaphraseMode(
        key="formal",
        label="Formal",
        description="More professional register: fewer contractions, more precise vocabulary.",
        ollama_instruction=(
            "Rewrite the following text in a more formal, professional register. Avoid contractions and "
            "casual phrasing, use precise vocabulary, but do not make it stiff or add unnecessary jargon. "
            "Keep the same meaning and approximate length."
        ),
        synonym_rate=0.4,
        prefer_shorter_synonyms=False,
        allow_contractions=False,
        target_formality="formal",
        length_bias=1.0,
    ),
    "simple": ParaphraseMode(
        key="simple",
        label="Simple",
        description="Plainer, shorter words and sentences — easier to read.",
        ollama_instruction=(
            "Rewrite the following text using simpler words and shorter sentences, so it's easier to read "
            "(roughly a plain-English / 8th-grade reading level). Keep the same meaning. Do not remove any facts."
        ),
        synonym_rate=0.3,
        prefer_shorter_synonyms=True,
        allow_contractions=True,
        target_formality="casual",
        length_bias=0.9,
    ),
    "creative": ParaphraseMode(
        key="creative",
        label="Creative",
        description="More varied phrasing and sentence rhythm, further from the original wording.",
        ollama_instruction=(
            "Rewrite the following text with more creative, varied phrasing and sentence rhythm — feel free "
            "to restructure sentences and use more expressive word choices. Keep the same core meaning and facts."
        ),
        synonym_rate=0.55,
        prefer_shorter_synonyms=False,
        allow_contractions=True,
        target_formality="neutral",
        length_bias=1.05,
    ),
    "shorten": ParaphraseMode(
        key="shorten",
        label="Shorten",
        description="Condenses the text — cuts redundancy while preserving key facts.",
        ollama_instruction=(
            "Condense the following text to be noticeably shorter (aim for roughly 60-75% of the original "
            "length) by cutting redundancy and filler, while preserving every fact, number, and name."
        ),
        synonym_rate=0.25,
        prefer_shorter_synonyms=True,
        allow_contractions=True,
        target_formality="neutral",
        length_bias=0.7,
    ),
    "expand": ParaphraseMode(
        key="expand",
        label="Expand",
        description="Adds clarifying detail and elaboration without inventing new facts.",
        ollama_instruction=(
            "Expand the following text with clarifying detail, transitions, and elaboration on points already "
            "present, making it more thorough — but do not invent new facts, statistics, or claims not implied "
            "by the original."
        ),
        synonym_rate=0.3,
        prefer_shorter_synonyms=False,
        allow_contractions=True,
        target_formality="neutral",
        length_bias=1.3,
    ),
}

DEFAULT_MODE = "standard"


def get_mode(key: str) -> ParaphraseMode:
    return MODES.get(key, MODES[DEFAULT_MODE])
