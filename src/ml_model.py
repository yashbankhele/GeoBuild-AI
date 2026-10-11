"""A small supervised pixel classifier for labelled image-pair data.

This is a baseline model, not a pretrained satellite change-detection network.
Use spatially independent sites/dates for robust evaluation.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


def build_pixel_features(before_rgb, after_rgb):
    before = np.asarray(before_rgb, dtype=np.float32)
    after = np.asarray(after_rgb, dtype=np.float32)
    if before.ndim != 3 or after.ndim != 3 or before.shape[2] != 3 or after.shape[2] != 3:
        raise ValueError("Both images must be RGB arrays.")
    if before.shape != after.shape:
        raise ValueError("Images must be aligned and have identical shapes.")
    before = before / 255.0
    after = after / 255.0
    diff = np.abs(after - before)
    before_gray = before.mean(axis=2, keepdims=True)
    after_gray = after.mean(axis=2, keepdims=True)
    features = np.concatenate([before, after, diff, before_gray, after_gray], axis=2)
    return features.reshape(-1, features.shape[2])


def train_pixel_classifier(before_rgb, after_rgb, ground_truth_mask, random_state=42,
                           max_samples=100_000):
    """Train/test split on pixels for a smoke-test only; not a spatially robust benchmark."""
    mask = np.asarray(ground_truth_mask)
    if mask.ndim != 2 or mask.shape != np.asarray(before_rgb).shape[:2]:
        raise ValueError("Ground-truth mask must match image height and width.")
    X = build_pixel_features(before_rgb, after_rgb)
    y = (mask.reshape(-1) > 0).astype(np.uint8)
    if np.unique(y).size < 2:
        raise ValueError("Ground-truth mask must contain both changed and unchanged pixels.")
    if len(y) > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(y), size=max_samples, replace=False)
        X, y = X[idx], y[idx]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=100, max_depth=14, min_samples_leaf=2,
        class_weight="balanced", random_state=random_state, n_jobs=-1
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    return model, report, int(len(y_train)), int(len(y_test))


def predict_pixel_mask(model, before_rgb, after_rgb):
    features = build_pixel_features(before_rgb, after_rgb)
    h, w = np.asarray(before_rgb).shape[:2]
    return (model.predict(features).reshape(h, w).astype(np.uint8) * 255)


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)
