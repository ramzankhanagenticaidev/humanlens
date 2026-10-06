"""
Merges multiple text,label CSV datasets (e.g. the hand-written seed set
and an imported RAID sample) into one, deduplicating exact-text matches
and reporting the resulting class balance so an imbalance doesn't
silently skew training.

Usage:
    python scripts/merge_datasets.py datasets/combined_sample.csv datasets/raid_sample.csv \
        --out datasets/combined_sample.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "text" not in (reader.fieldnames or []) or "label" not in (reader.fieldnames or []):
            print(f"Skipping {path}: missing required 'text'/'label' columns.")
            return []
        return list(reader)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple text,label dataset CSVs")
    parser.add_argument("inputs", nargs="+", help="Input CSV files to merge")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    seen_texts: set[str] = set()
    merged: list[dict] = []
    dupes = 0

    for path_str in args.inputs:
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: {path} does not exist, skipping.")
            continue
        rows = load_rows(path)
        added = 0
        for row in rows:
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip().lower()
            if not text or label not in {"human", "ai"}:
                continue
            key = text.lower()
            if key in seen_texts:
                dupes += 1
                continue
            seen_texts.add(key)
            merged.append({"text": text, "label": label})
            added += 1
        print(f"{path}: added {added} rows")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(merged)

    n_human = sum(1 for r in merged if r["label"] == "human")
    n_ai = sum(1 for r in merged if r["label"] == "ai")

    print(f"\nWrote {len(merged)} rows to {out_path} ({dupes} exact-duplicate rows skipped)")
    print(f"  human: {n_human}")
    print(f"  ai:    {n_ai}")

    if n_human == 0 or n_ai == 0:
        print("ERROR: at least one class has zero examples. Training will fail — check your inputs.")
        sys.exit(1)

    ratio = max(n_human, n_ai) / max(1, min(n_human, n_ai))
    if ratio > 1.5:
        print(
            f"Warning: class imbalance ratio is {ratio:.2f}x (larger class / smaller class). "
            f"The classifier uses class_weight='balanced' to compensate, but a closer-to-even "
            f"dataset generally still gives more reliable results."
        )


if __name__ == "__main__":
    main()
