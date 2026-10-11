"""Evaluation helpers for change masks and binary pixel classifiers."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    jaccard_score, confusion_matrix
)


def evaluate_binary_mask(predicted_mask, ground_truth_mask):
    """Evaluate predicted and labelled masks. Non-zero pixels mean 'change'."""
    pred = np.asarray(predicted_mask)
    truth = np.asarray(ground_truth_mask)
    if pred.ndim != 2 or truth.ndim != 2:
        raise ValueError("Prediction and ground-truth masks must be 2D.")
    if pred.shape != truth.shape:
        raise ValueError("Prediction and ground-truth masks must have identical dimensions.")
    y_pred = (pred.reshape(-1) > 0).astype(np.uint8)
    y_true = (truth.reshape(-1) > 0).astype(np.uint8)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "iou": float(jaccard_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "pixels": int(y_true.size),
        "labelled_change_pixels": int(y_true.sum()),
        "predicted_change_pixels": int(y_pred.sum()),
    }
