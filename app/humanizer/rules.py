"""
Deterministic, rule-based rewriting rules.

These are NOT synonym-spinning. Each rule targets a specific, well-known
category of stilted/formulaic writing and applies a structural fix:
  - strip empty throat-clearing phrases
  - convert common formulaic openers into direct statements
  - break up long, comma-heavy sentences
  - flag (not silently fix) excessive passive voice

Every transformation is logged as a `Change` so the UI can show a real
diff instead of a black-box "improved" blob of text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

from app.detector.formulaic import load_phrases


@dataclass
class Change:
    rule: str
    before: str
    after: str
    reason: str


# --- Rule 1: strip empty throat-clearing / hedging openers -----------------
THROAT_CLEARERS = [
    (r"\bIt is important to note that\s+", ""),
    (r"\bIt is worth noting that\s+", ""),
    (r"\bIt should be noted that\s+", ""),
    (r"\bIt is essential to (note|mention|recognize) that\s+", ""),
    (r"\bOne must consider that\s+", ""),
    (r"\bAs previously mentioned,\s+", ""),
]


def strip_throat_clearers(text: str) -> Tuple[str, List[Change]]:
    changes = []
    result = text
    for pattern, replacement in THROAT_CLEARERS:
        for m in re.finditer(pattern, result, flags=re.IGNORECASE):
            before = m.group(0)
            changes.append(
                Change(
                    rule="throat_clearer_removed",
                    before=before.strip(),
                    after="(removed)",
                    reason="Empty hedging phrase that adds no meaning.",
                )
            )
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)
        # Capitalize the new sentence start if we just removed a leading phrase
        result = re.sub(
            r"(^|[.!?]\s+)([a-z])",
            lambda m: m.group(1) + m.group(2).upper(),
            result,
        )
    return result, changes


# --- Rule 2: simplify verbose corporate-speak phrases -----------------------
VERBOSE_REPLACEMENTS = [
    (r"\butilize\b", "use"),
    (r"\butilizing\b", "using"),
    (r"\bfacilitate the optimization of\b", "improve"),
    (r"\bplays a crucial role in\b", "helps with"),
    (r"\bplays a vital role in\b", "helps with"),
    (r"\ba wide range of\b", "many"),
    (r"\bin order to\b", "to"),
    (r"\bdue to the fact that\b", "because"),
    (r"\bat this point in time\b", "now"),
    (r"\bin the event that\b", "if"),
    (r"\bfor the purpose of\b", "to"),
    (r"\bwith regard to\b", "about"),
    (r"\bin the process of\b", ""),
    (r"\bhas the ability to\b", "can"),
    (r"\bis able to\b", "can"),
    (r"\bcannot be overstated\b", "matters a lot"),
    (r"\bcomprehensive\b", "thorough"),
    (r"\bmultifaceted\b", "complex"),
    (r"\bseamlessly\b", "smoothly"),
    (r"\bleverage\b", "use"),
    (r"\bleveraging\b", "using"),
    (r"\brobust\b", "strong"),
]


def simplify_verbosity(text: str) -> Tuple[str, List[Change]]:
    changes = []
    result = text
    for pattern, replacement in VERBOSE_REPLACEMENTS:
        for m in re.finditer(pattern, result, flags=re.IGNORECASE):
            changes.append(
                Change(
                    rule="verbosity_simplified",
                    before=m.group(0),
                    after=replacement if replacement else "(removed)",
                    reason="Replaced verbose/corporate phrasing with a plainer equivalent.",
                )
            )
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    result = re.sub(r"\s{2,}", " ", result)
    return result, changes


# --- Rule 3: vary repeated sentence openings --------------------------------
REPEATED_OPENERS = ["Furthermore,", "Moreover,", "Additionally,", "In conclusion,", "Overall,"]
OPENER_ALTERNATIVES = {
    "Furthermore,": ["Also,", "On top of that,", "What's more,"],
    "Moreover,": ["On top of that,", "And", "Beyond that,"],
    "Additionally,": ["Also,", "On top of that,"],
    "In conclusion,": ["In short,", "Overall,", "All told,"],
    "Overall,": ["In short,", "On the whole,"],
}


def vary_repeated_openers(sentences: List[str]) -> Tuple[List[str], List[Change]]:
    changes = []
    seen_openers = {}
    result = []
    for s in sentences:
        replaced = s
        for opener in REPEATED_OPENERS:
            if s.startswith(opener):
                seen_openers[opener] = seen_openers.get(opener, 0) + 1
                if seen_openers[opener] > 1:
                    alternatives = OPENER_ALTERNATIVES.get(opener, [])
                    if alternatives:
                        replacement = alternatives[(seen_openers[opener] - 2) % len(alternatives)]
                        replaced = replacement + s[len(opener):]
                        changes.append(
                            Change(
                                rule="varied_sentence_opener",
                                before=opener,
                                after=replacement,
                                reason=f'"{opener}" was already used to open another sentence nearby.',
                            )
                        )
                break
        result.append(replaced)
    return result, changes


# --- Rule 4: flag (don't silently rewrite) long, comma-heavy sentences -----
def flag_long_sentences(sentences: List[str], max_words: int = 35, max_commas: int = 4) -> List[Change]:
    changes = []
    for s in sentences:
        word_count = len(s.split())
        comma_count = s.count(",")
        if word_count > max_words and comma_count >= max_commas:
            changes.append(
                Change(
                    rule="long_sentence_flagged",
                    before=s,
                    after=s,  # not auto-rewritten — flagged for the user
                    reason=(
                        f"This sentence is {word_count} words with {comma_count} commas. "
                        "Consider splitting it into two sentences for clarity."
                    ),
                )
            )
    return changes


# --- Rule 5: reduce redundant qualifier stacking ----------------------------
QUALIFIER_STACKS = [
    r"\bvery unique\b",
    r"\bcompletely essential\b",
    r"\bactual fact\b",
    r"\bfuture plans\b",
    r"\bpast history\b",
    r"\beach and every\b",
    r"\bfirst and foremost\b",
    r"\bvarious different\b",
    r"\bend result\b",
    r"\bbasic fundamentals\b",
]

QUALIFIER_REPLACEMENTS = {
    r"\bvery unique\b": "unique",
    r"\bcompletely essential\b": "essential",
    r"\bactual fact\b": "fact",
    r"\bfuture plans\b": "plans",
    r"\bpast history\b": "history",
    r"\beach and every\b": "every",
    r"\bfirst and foremost\b": "first",
    r"\bvarious different\b": "various",
    r"\bend result\b": "result",
    r"\bbasic fundamentals\b": "fundamentals",
}


def remove_redundant_qualifiers(text: str) -> Tuple[str, List[Change]]:
    changes = []
    result = text
    for pattern in QUALIFIER_STACKS:
        replacement = QUALIFIER_REPLACEMENTS[pattern]
        for m in re.finditer(pattern, result, flags=re.IGNORECASE):
            changes.append(
                Change(
                    rule="redundant_qualifier_removed",
                    before=m.group(0),
                    after=replacement,
                    reason="Redundant phrase where one word already implies the other.",
                )
            )
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result, changes
