"""
Train the AI-likelihood classifier from a CSV dataset with columns:
    text,label
where label is "human" or "ai".

Usage:
    python scripts/train_model.py --dataset datasets/combined_sample.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.detector.classifier import AIDetectorModel  # noqa: E402


def load_dataset(path: Path) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "text" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("Dataset must have 'text' and 'label' columns")
        for row in reader:
            label = row["label"].strip().lower()
            if label not in {"human", "ai"}:
                continue
            text = row["text"].strip()
            if not text:
                continue
            texts.append(text)
            labels.append(label)
    return texts, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HumanLens AI-likelihood classifier")
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).resolve().parent.parent / "datasets" / "combined_sample.csv"),
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    texts, labels = load_dataset(dataset_path)
    if len(texts) < 10:
        print(f"Only {len(texts)} usable rows found — need at least 10 to train.")
        sys.exit(1)

    n_human = labels.count("human")
    n_ai = labels.count("ai")
    print(f"Loaded {len(texts)} rows ({n_human} human, {n_ai} ai) from {dataset_path}")

    model = AIDetectorModel()
    result = model.fit(texts, labels)
    model.save()

    print(f"Trained model: {result.n_samples} samples, {result.n_features} features, "
          f"classes={result.classes}")
    print(f"Saved to {model.model_dir}")


if __name__ == "__main__":
    main()
