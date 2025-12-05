import time
from time import perf_counter
import random
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from joblib import load
from extract_features import extract_features

# ONLY USED FOR MOCK FUNCTIONS
LABELS = ["laughing", "coughing", "clapping", "knocking", "alarm"]

# mock model functions for GUI testing
# Both functions behave identically and produce a rando, label
def mock_ml_predict(window):
    t0 = perf_counter()
    time.sleep(0.02 + random.random() * 0.01)  # 20-30 ms simulate
    label = random.choice(LABELS)
    prob = 0.5 + random.random() * 0.5
    t = (perf_counter() - t0) * 1000.0
    return label, float(prob), t

def mock_dl_predict(window):
    t0 = perf_counter()
    time.sleep(0.05 + random.random() * 0.05)  # 50-100 ms simulate
    label = random.choice(LABELS)
    prob = 0.6 + random.random() * 0.4
    t = (perf_counter() - t0) * 1000.0
    return label, float(prob), t

"""
Real ML model loading and prediction helpers.
Expects features matching the training pipeline.
"""

# Globals populated by load_ml_model()
ml_model = None
ml_scaler = None
ml_classes = None
ml_activities = None
ml_feature_dim = None

def load_ml_model(path: str = "svm_small_to_large.joblib"):
    artifact = load(path)
    global ml_model, ml_scaler, ml_classes, ml_activities, ml_feature_dim
    ml_model = artifact["model"]
    ml_scaler = artifact["scaler"]
    ml_classes = artifact["classes"]
    ml_activities = artifact.get("activities")
    ml_feature_dim = artifact.get("feature_dim")
    print("ML Model successfully loaded")


def ml_predict(features: np.ndarray):
    """
    Predict using the loaded SVM on a single feature vector.

    Args:
        features: 1-D numpy array of shape (feature_dim,)

    Returns:
        (pred_class_label, pred_name, confidence, latency_ms)
    """
    if ml_model is None or ml_scaler is None:
        raise RuntimeError("ML model not loaded. Call load_ml_model() first.")

    t0 = perf_counter()

    x_feat = np.asarray(features, dtype=float).reshape(1, -1)
    if ml_feature_dim is not None and x_feat.shape[1] != ml_feature_dim:
        raise ValueError(f"Feature dim mismatch: got {x_feat.shape[1]}, expected {ml_feature_dim}")

    x_scaled = ml_scaler.transform(x_feat)
    probs = ml_model.predict_proba(x_scaled)[0]
    pred_idx = int(np.argmax(probs))
    pred_class_label = ml_classes[pred_idx]
    pred_name = None
    if isinstance(ml_activities, (list, tuple)) and pred_class_label < len(ml_activities):
        pred_name = ml_activities[pred_class_label]
    confidence = float(probs[pred_idx])
    t_ms = (perf_counter() - t0) * 1000.0
    # todo: RETURN NONE if no confidences are above a certain threshold
    return pred_name, confidence, t_ms


def audio_to_features(window: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Convenience to turn raw audio window into feature vector matching training."""
    return extract_features(window, sr=sr, feature_type="all")

if __name__ == '__main__':
    load_ml_model()
    print("ml_model:", type(ml_model))
    print("ml_scaler:", type(ml_scaler))
    print("ml_classes:", ml_classes)
    print("ml_activities:", ml_activities)

