"""
Evaluation framework.

Runs stratified k-fold cross-validation over a text,label dataset and
reports real accuracy/precision/recall/F1/ROC-AUC/confusion matrix on
OUT-OF-FOLD predictions (i.e. never scoring a sample the model was
trained on). This matters: evaluating on training data would produce
inflated, dishonest numbers.

Usage:
    python evaluation/evaluate.py --dataset datasets/combined_sample.csv --folds 5

Writes a JSON report to evaluation/reports/latest_report.json, which
the Streamlit "Evaluation" page reads and displays.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.model_selection import StratifiedKFold  # noqa: E402

from app.detector.classifier import AIDetectorModel  # noqa: E402
from evaluation.metrics import compute_metrics  # noqa: E402
from scripts.train_model import load_dataset  # noqa: E402

REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def run_cross_validation(texts: list[str], labels: list[str], n_folds: int = 5, seed: int = 42) -> dict:
    n_folds = min(n_folds, min(labels.count("human"), labels.count("ai")))
    if n_folds < 2:
        raise ValueError(
            "Not enough samples per class for cross-validation. "
            "Need at least 2 examples of each label."
        )

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    y_true_all, y_pred_all, y_score_all = [], [], []
    fold_reports = []

    import numpy as np

    texts_arr = np.array(texts, dtype=object)
    labels_arr = np.array(labels, dtype=object)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(texts_arr, labels_arr)):
        train_texts = list(texts_arr[train_idx])
        train_labels = list(labels_arr[train_idx])
        test_texts = list(texts_arr[test_idx])
        test_labels = list(labels_arr[test_idx])

        model = AIDetectorModel()
        model.fit(train_texts, train_labels)
        scores = model.predict_proba(test_texts)  # P(ai)
        preds = ["ai" if s >= 0.5 else "human" for s in scores]

        y_true_all.extend(test_labels)
        y_pred_all.extend(preds)
        y_score_all.extend(scores)

        fold_metrics = compute_metrics(test_labels, preds, scores)
        fold_reports.append({"fold": fold_idx, "n_test": len(test_idx), **fold_metrics.to_dict()})

    overall = compute_metrics(y_true_all, y_pred_all, y_score_all)

    return {
        "overall": overall.to_dict(),
        "folds": fold_reports,
        "n_folds": n_folds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HumanLens AI-likelihood detector")
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).resolve().parent.parent / "datasets" / "combined_sample.csv"),
    )
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    texts, labels = load_dataset(dataset_path)
    print(f"Loaded {len(texts)} rows ({labels.count('human')} human, {labels.count('ai')} ai)")

    cv_result = run_cross_validation(texts, labels, n_folds=args.folds)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "n_samples": len(texts),
        "n_folds": cv_result["n_folds"],
        "metrics": cv_result["overall"],
        "confusion_matrix": cv_result["overall"]["confusion_matrix"],
        "per_fold": cv_result["folds"],
        "note": (
            "Metrics are computed via stratified k-fold cross-validation on out-of-fold "
            "predictions (the model never sees a test example during that fold's training). "
            "This is a small bootstrap dataset — see README Limitations before treating these "
            "numbers as representative of real-world performance."
        ),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "latest_report.json"
    out_path.write_text(json.dumps(report, indent=2))

    print("\n=== Cross-validated metrics (out-of-fold) ===")
    for k, v in report["metrics"].items():
        print(f"  {k}: {v}")
    print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    main()
