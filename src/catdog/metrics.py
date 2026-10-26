"""Classification metrics computed from a confusion matrix."""

from collections.abc import Sequence

import numpy as np


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> np.ndarray:
    """Counts with rows = true class and columns = predicted class."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def report(cm: np.ndarray, classes: Sequence[str]) -> dict:
    """Accuracy, balanced accuracy and per-class precision, recall, F1 and support."""
    tp = np.diag(cm)
    per_class = {}
    for i, name in enumerate(classes):
        precision = _ratio(tp[i], cm[:, i].sum())
        recall = _ratio(tp[i], cm[i, :].sum())
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": _ratio(2 * precision * recall, precision + recall),
            "support": int(cm[i, :].sum()),
        }
    return {
        "accuracy": _ratio(tp.sum(), cm.sum()),
        "balanced_accuracy": float(np.mean([c["recall"] for c in per_class.values()])),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "n": int(cm.sum()),
    }
