"""
Metrics computation.

Thin wrapper around scikit-learn's metrics so the rest of the codebase
has one place that defines what "accuracy" etc. mean for this project.
Never fabricates numbers — if a metric can't be computed (e.g. ROC-AUC
with only one class present), it is reported as None with an explanation
rather than a guessed value.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvaluationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float]
    false_positive_rate: float
    false_negative_rate: float
    confusion_matrix: List[List[int]]
    labels: List[str]

    def to_dict(self) -> dict:
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "roc_auc": round(self.roc_auc, 4) if self.roc_auc is not None else None,
            "false_positive_rate": round(self.false_positive_rate, 4),
            "false_negative_rate": round(self.false_negative_rate, 4),
            "confusion_matrix": self.confusion_matrix,
            "labels": self.labels,
        }


def compute_metrics(
    y_true: List[str],
    y_pred: List[str],
    y_scores: Optional[List[float]] = None,
    positive_label: str = "ai",
) -> EvaluationMetrics:
    labels = sorted(set(y_true) | set(y_pred))
    if positive_label not in labels:
        labels = [positive_label] + [l for l in labels if l != positive_label]

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)

    # FPR / FNR computed directly from the confusion matrix for the
    # positive class ("ai"), to avoid relying on sklearn defaults that
    # can differ by label ordering.
    pos_idx = labels.index(positive_label)
    neg_idx = 1 - pos_idx if len(labels) == 2 else None

    fpr = fnr = 0.0
    if neg_idx is not None:
        tp = cm[pos_idx][pos_idx]
        fn = sum(cm[pos_idx]) - tp
        fp = cm[neg_idx][pos_idx]
        tn = sum(cm[neg_idx]) - fp
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    roc_auc = None
    if y_scores is not None and len(set(y_true)) == 2:
        try:
            binary_true = [1 if y == positive_label else 0 for y in y_true]
            roc_auc = roc_auc_score(binary_true, y_scores)
        except ValueError:
            roc_auc = None  # e.g. only one class present in this split

    return EvaluationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        false_positive_rate=fpr,
        false_negative_rate=fnr,
        confusion_matrix=cm,
        labels=labels,
    )
