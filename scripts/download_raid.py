"""
Imports a stratified sample from the RAID benchmark
(https://github.com/liamdugan/raid, MIT licensed, ACL 2024) and converts
it into this project's text,label CSV schema.

RAID contains 10M+ documents: real human-written text plus generations
from 11 different AI models (ChatGPT, GPT-4, GPT-3, GPT-2, Llama-2-Chat,
Mistral, Mistral-Chat, MPT, MPT-Chat, Cohere, Cohere-Chat) across 11
genres (news, reviews, reddit, wiki, books, abstracts, recipes, poetry,
code, czech, german). This is exactly the kind of larger, multi-model,
multi-genre dataset the README's "Limitations" section says you need to
get a trustworthy accuracy number — the 80-row seed dataset is not it.

WHY STREAMING: the full RAID train split is ~11GB. This script uses
Hugging Face `datasets` in streaming mode so it pulls only as many rows
as needed for your sample size, without downloading the whole file.

USAGE
    pip install datasets   # not in the base requirements.txt — optional
    python scripts/download_raid.py --per-class 300 --domains news reviews reddit wiki

    # Then merge with (or replace) the seed dataset:
    python scripts/merge_datasets.py datasets/combined_sample.csv datasets/raid_sample.csv \
        --out datasets/combined_sample.csv

    # Then retrain:
    python scripts/train_model.py --dataset datasets/combined_sample.csv

NOTES / HONESTY
- This script was written and its filtering/sampling logic was verified
  against a mock CSV matching RAID's exact column schema. The actual
  network download to Hugging Face was NOT exercised in the environment
  this project was built in (that sandbox blocks huggingface.co). Run it
  yourself with a normal internet connection — if the RAID schema has
  changed since this was written, the column names below are the place
  to check first.
- Default filters keep only `decoding=greedy`, `repetition_penalty=no`,
  `attack=none` — i.e. "clean" generations, not adversarially-attacked
  ones. That's the right starting point for a basic classifier. RAID's
  whole point is also robustness testing against attacks (paraphrasing,
  homoglyphs, etc.) — see --include-adversarial if you want that harder,
  more realistic mix once the basic pipeline is working well.
- RAID is MIT licensed, so commercial use is fine — see THIRD_PARTY_LICENSES.md.
  Still worth a quick look at the dataset card yourself before shipping
  a paid product trained on it: https://huggingface.co/datasets/liamdugan/raid
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator

OUT_DEFAULT = Path(__file__).resolve().parent.parent / "datasets" / "raid_sample.csv"

# All 11 AI model labels RAID currently generates from (excludes human text,
# which RAID marks separately — see NOTE below).
AI_MODELS = [
    "chatgpt", "gpt4", "gpt3", "gpt2", "llama-chat",
    "mistral", "mistral-chat", "mpt", "mpt-chat", "cohere", "cohere-chat",
]

# English-language genres relevant to publishing/SEO content. RAID also
# has "code", "czech", "german", "poetry" — excluded by default since
# they're off-topic for typical article writing, but you can pass
# --domains to include them.
DEFAULT_DOMAINS = ["news", "reviews", "reddit", "wiki", "books", "abstracts", "recipes"]

MIN_WORDS = 30  # skip near-empty generations that wouldn't pass this app's own min-word analysis


def _word_count(text: str) -> int:
    return len(text.split())


def stream_raid_rows(domains: list[str], include_adversarial: bool) -> Iterator[dict]:
    """Yields raw RAID rows one at a time via HF streaming (no full download)."""
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "The 'datasets' package isn't installed. Run:\n"
            "    pip install datasets\n"
            "(It's optional — only needed for this import script, not the base app.)"
        )
        sys.exit(1)

    ds = load_dataset("liamdugan/raid", split="train", streaming=True)

    for row in ds:
        if row.get("domain") not in domains:
            continue
        if not include_adversarial:
            # Keep only "clean" generations: no adversarial attack, greedy
            # decoding, no repetition penalty. Human rows may not carry
            # these fields the same way — accept them regardless.
            is_human = _is_human_row(row)
            if not is_human:
                if row.get("attack") not in (None, "none", ""):
                    continue
                if row.get("decoding") not in (None, "greedy"):
                    continue
                if row.get("repetition_penalty") not in (None, "no"):
                    continue
        yield row


def _is_human_row(row: dict) -> bool:
    """RAID's human-written rows are identified by model == 'human'
    (source_id == id and no generating model). If the schema differs when
    you actually pull it, adjust this one function — everything else in
    the script is schema-agnostic."""
    model = (row.get("model") or "").lower()
    if model == "human":
        return True
    # Fallback heuristic in case the 'human' literal isn't used: a source
    # document referencing itself (no separate generation) is the human original.
    return row.get("id") == row.get("source_id") and not model


def build_sample(
    domains: list[str],
    per_class: int,
    include_adversarial: bool,
    seed: int = 42,
) -> list[dict]:
    """Collects up to `per_class` human rows and `per_class` AI rows,
    balancing across the 11 AI models on a best-effort basis (a soft
    per-model ceiling, not a hard cap) so the total can still reach
    `per_class` even if the stream happens to contain fewer distinct
    models than expected — stopping as soon as both quotas are met so
    we don't pull more of the stream than necessary."""
    rng = random.Random(seed)

    human_rows: list[dict] = []
    ai_buckets: dict[str, list[dict]] = defaultdict(list)
    # Soft ceiling per model: generous enough that a shortfall in one
    # model's diversity doesn't prevent reaching the overall per_class
    # total from the remaining models.
    soft_ceiling_per_model = max(3, (per_class // len(AI_MODELS)) * 3)

    for row in stream_raid_rows(domains, include_adversarial):
        text = (row.get("generation") or "").strip()
        if _word_count(text) < MIN_WORDS:
            continue

        if _is_human_row(row):
            if len(human_rows) < per_class:
                human_rows.append(row)
        else:
            model = (row.get("model") or "unknown").lower()
            ai_total_so_far = sum(len(b) for b in ai_buckets.values())
            bucket = ai_buckets[model]
            # Accept into this model's bucket if it hasn't hit its own
            # soft ceiling yet, OR the overall AI total still needs more
            # rows than the other buckets can currently supply.
            if len(bucket) < soft_ceiling_per_model or ai_total_so_far < per_class:
                bucket.append(row)

        ai_total = sum(len(b) for b in ai_buckets.values())
        if len(human_rows) >= per_class and ai_total >= per_class:
            break

    ai_rows: list[dict] = []
    for bucket in ai_buckets.values():
        ai_rows.extend(bucket)
    rng.shuffle(ai_rows)
    ai_rows = ai_rows[:per_class]

    return human_rows, ai_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a stratified sample from the RAID benchmark")
    parser.add_argument("--per-class", type=int, default=300, help="Target rows per label (human / ai)")
    parser.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS, help="RAID domains to include")
    parser.add_argument("--include-adversarial", action="store_true",
                         help="Include adversarially-attacked AI generations (harder, more realistic, noisier)")
    parser.add_argument("--out", default=str(OUT_DEFAULT))
    args = parser.parse_args()

    print(f"Streaming RAID (domains={args.domains}, per_class={args.per_class}, "
          f"include_adversarial={args.include_adversarial}) ...")
    print("This pulls only as many rows as needed via streaming, not the full ~11GB dataset.")

    human_rows, ai_rows = build_sample(args.domains, args.per_class, args.include_adversarial)

    if len(human_rows) < args.per_class or len(ai_rows) < args.per_class:
        print(
            f"Warning: only found {len(human_rows)} human / {len(ai_rows)} ai rows matching "
            f"your filters (wanted {args.per_class} each). Try more domains, a lower --per-class, "
            f"or --include-adversarial."
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label", "domain", "model"])
        for row in human_rows:
            writer.writerow([row.get("generation", "").strip(), "human", row.get("domain", ""), "human"])
        for row in ai_rows:
            writer.writerow([row.get("generation", "").strip(), "ai", row.get("domain", ""), row.get("model", "")])

    print(f"Wrote {len(human_rows)} human + {len(ai_rows)} ai rows to {out_path}")
    print("Next: merge with the seed dataset and retrain, e.g.:")
    print(f"    python scripts/merge_datasets.py datasets/combined_sample.csv {out_path} "
          f"--out datasets/combined_sample.csv")
    print("    python scripts/train_model.py")


if __name__ == "__main__":
    main()
