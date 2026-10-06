"""
Adversarial / robustness testing.

Runs the detector against categories of text specifically chosen to be
hard: very short text, heavily formal human writing, informal human
writing, mixed human/AI text, paraphrased text, different genres. The
goal is to find where the detector is unreliable and report it plainly
— not to make the detector look good.

Usage:
    python evaluation/adversarial.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.detector.detector import get_detector  # noqa: E402

REPORTS_DIR = Path(__file__).resolve().parent / "reports"

# Each case: (category, expected_label, text)
# expected_label is what a careful human reader would judge the text to
# be, used only to report whether the detector's verdict direction
# matched expectation — this is a stress test, not a labeled training set.
CASES = [
    (
        "short_text",
        "human",
        "lol ok that meeting ran way too long, someone please tell Greg we get it.",
    ),
    (
        "short_text",
        "ai",
        "Effective communication is essential for organizational success.",
    ),
    (
        "long_text_human",
        "human",
        "I've been putting off writing about this for weeks because I wasn't sure how to say it without "
        "sounding dramatic, but here goes. My dad was diagnosed in March, and since then everything about "
        "our family's routine has shifted in small, exhausting ways — who drives him to appointments, who "
        "calls the insurance company, who sits with him when the house feels too quiet. Some days it's fine, "
        "genuinely fine, we make jokes about the hospital coffee and argue about what to watch. Other days I "
        "sit in my car in the parking garage for ten minutes before I can make myself go inside. There isn't "
        "really a neat lesson here. I just wanted to write it down somewhere.",
    ),
    (
        "long_text_formal_human",
        "human",
        "The committee convened on the fourteenth to review the proposed amendments to the bylaws. "
        "After considerable debate, particularly regarding Section 4(b), the members present voted seven "
        "to two in favor of adoption, with two abstentions. Concerns were raised about the enforcement "
        "mechanism, and it was agreed that the treasurer would circulate a revised draft prior to the next "
        "session. Minutes were recorded by the secretary and will be distributed within the week.",
    ),
    (
        "highly_formulaic_ai",
        "ai",
        "In today's ever-evolving business landscape, it is important to note that organizations must "
        "leverage cutting-edge technology to remain competitive. Furthermore, a comprehensive and holistic "
        "approach to digital transformation plays a crucial role in driving sustainable growth. Moreover, "
        "businesses that navigate the complexities of this multifaceted environment are better positioned "
        "to unlock the potential of emerging opportunities. In conclusion, embracing innovation is essential "
        "for long-term success.",
    ),
    (
        "edited_ai_lightly_humanized",
        "ai",
        "AI is changing a lot of industries right now. It helps companies work faster and make fewer "
        "mistakes. A lot of businesses are trying to use it more, especially for things like customer "
        "service and data analysis. It's not perfect, but it's getting better, and most experts think this "
        "trend continues for the next several years.",
    ),
    (
        "mixed_human_ai",
        "mixed",
        "So I asked an AI tool to help draft this intro and honestly it wasn't bad. In today's competitive "
        "job market, it is essential to craft a resume that effectively highlights your core strengths. "
        "Anyway I tweaked the rest myself — figured I'd add a bit about the internship I did last summer "
        "at the clinic, since that's honestly the part I'm most proud of.",
    ),
    (
        "technical_writing",
        "human",
        "To reproduce: run `make build` on a clean checkout, then `./bin/server --port 8080`. If the health "
        "check times out, check that port 8080 isn't already bound — `lsof -i :8080` will tell you. This bit "
        "me twice this week, so I'm adding it to the runbook.",
    ),
    (
        "email_genre",
        "human",
        "Hey — can you push the 1:1 to Thursday? Something came up with the Miller account and I need to "
        "sort it before end of day. Sorry for the short notice, I owe you a coffee.",
    ),
    (
        "blog_post_genre_ai_leaning",
        "ai",
        "Are you looking to boost your productivity in today's fast-paced world? In this comprehensive "
        "guide, we'll delve into a wide range of strategies that can seamlessly transform your daily "
        "routine. From time-blocking to mindfulness, these robust techniques play a crucial role in "
        "helping you unlock your full potential.",
    ),
    (
        "heavily_paraphrased_ai",
        "ai",
        "Companies today face constant pressure to keep pace with change. Adopting modern tools well is "
        "not optional anymore if a business wants to stay relevant. Taking a broad, joined-up view of how "
        "technology gets rolled out tends to support steadier growth over time. Firms that handle this kind "
        "of many-sided shift tend to end up ahead when new chances come along. Ultimately, staying open to "
        "change tends to pay off in the long run.",
    ),
]


def run_suite() -> dict:
    detector = get_detector()
    results = []
    for category, expected, text in CASES:
        r = detector.analyze(text)
        results.append(
            {
                "category": category,
                "expected": expected,
                "ai_probability": round(r.ai_probability, 4),
                "verdict": r.verdict,
                "confidence": r.confidence,
                "num_words": r.num_words,
                "text_preview": text[:100] + ("…" if len(text) > 100 else ""),
            }
        )

    # Simple pass/fail heuristic for reporting purposes only:
    # "ai" expected -> probability should lean > 0.5
    # "human" expected -> probability should lean < 0.5
    # "mixed" -> no strict expectation, informational only
    scored = []
    for r in results:
        if r["expected"] == "ai":
            passed = r["ai_probability"] > 0.5
        elif r["expected"] == "human":
            passed = r["ai_probability"] < 0.5
        else:
            passed = None
        scored.append({**r, "matched_expectation": passed})

    n_scored = [r for r in scored if r["matched_expectation"] is not None]
    n_passed = sum(1 for r in n_scored if r["matched_expectation"])

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_cases": len(scored),
        "scored_cases": len(n_scored),
        "passed": n_passed,
        "pass_rate": round(n_passed / len(n_scored), 4) if n_scored else None,
        "results": scored,
        "note": (
            "This is a small, hand-written stress-test set (not a benchmark). 'matched_expectation' "
            "reflects a human reviewer's rough judgment of the text, not ground truth. Failures here "
            "are expected and informative — they show where stylometric detection genuinely struggles "
            "(short text, lightly-edited AI text, mixed authorship, heavy paraphrasing)."
        ),
    }


def main() -> None:
    report = run_suite()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "adversarial_report.json"
    out_path.write_text(json.dumps(report, indent=2))

    print(f"Ran {report['total_cases']} adversarial cases "
          f"({report['passed']}/{report['scored_cases']} matched expected direction)\n")
    for r in report["results"]:
        flag = "✓" if r["matched_expectation"] else ("✗" if r["matched_expectation"] is False else "·")
        print(f"[{flag}] {r['category']:32s} expected={r['expected']:6s} "
              f"prob={r['ai_probability']:.2f} verdict={r['verdict']}")
    print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    main()
