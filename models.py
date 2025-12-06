import time
from time import perf_counter
import random
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from joblib import load
import os
from datetime import datetime
from extract_features import extract_features
import librosa
try:
    from tensorflow.keras.models import load_model as keras_load_model
except Exception:
    keras_load_model = None

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
dl_model = None

def load_ml_model(path: str = "svm_small_to_large.joblib"):
    artifact = load(path)
    global ml_model, ml_scaler, ml_classes, ml_activities, ml_feature_dim
    ml_model = artifact["model"]
    ml_scaler = artifact["scaler"]
    ml_classes = artifact["classes"]
    ml_activities = artifact.get("activities")
    ml_feature_dim = artifact.get("feature_dim")
    print("ML Model successfully loaded")


def load_dl_model(path: str = "best_finetuned_model_small_to_large.h5"):
    """
    Loader for fine-tuned Ubicoustics model from A2
    """
    global dl_model
    if keras_load_model is None:
        raise RuntimeError("TensorFlow/Keras not available. Install tensorflow to load .h5 models.")
    dl_model = keras_load_model(path)
    print("DL Model successfully loaded")


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


def dl_predict(window: np.ndarray, sr: int = 16000):
    """
    Returns (label_str, confidence_float, latency_ms).
    Preprocesses raw audio window into log-mel spectrogram of shape (96, 64, 1)
    to match the DL model input (None, 96, 64, 1).
    """
    if dl_model is None:
        raise RuntimeError("DL model not loaded. Call load_dl_model() first.")

    t0 = perf_counter()
    y = np.asarray(window, dtype=float)
    # Build log-mel spectrogram: 96 time frames x 64 mel bins
    n_mels = 64
    n_fft = 512
    hop_length = 160  # ~10 ms at 16kHz
    win_length = 400  # 25 ms window
    target_time = 96

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        n_mels=n_mels, power=2.0
    )  # shape (n_mels, T)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    feat = mel_db.T  # (T, 64)
    if feat.shape[0] < target_time:
        pad = target_time - feat.shape[0]
        feat = np.pad(feat, ((0, pad), (0, 0)), mode='constant', constant_values=feat.min())
    else:
        feat = feat[:target_time, :]
    x_in = feat[np.newaxis, :, :, np.newaxis].astype(np.float32)  # (1, 96, 64, 1)

    preds = dl_model.predict(x_in, verbose=0)
    # Handle different output shapes: scalar, vector, batch x classes
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    preds = np.asarray(preds)
    if preds.ndim == 2:
        probs = preds[0]
    else:
        probs = preds
    # Softmax-like selection
    if probs.ndim == 0:
        confidence = float(probs)
        pred_idx = 0
    else:
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
    # Map index to name if available from ml_activities; adjust if DL has its own label set
    label = ml_activities[pred_idx] if isinstance(ml_activities, (list, tuple)) and pred_idx < len(ml_activities) else str(pred_idx)
    t_ms = (perf_counter() - t0) * 1000.0
    return label, confidence, t_ms


def audio_to_features(window: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Convenience to turn raw audio window into feature vector matching training."""
    return extract_features(window, sr=sr, feature_type="all")

if __name__ == '__main__':
    load_ml_model()
    print("ml_model:", type(ml_model))
    print("ml_scaler:", type(ml_scaler))
    print("ml_classes:", ml_classes)
    print("ml_activities:", ml_activities)
    
    # Simple test runner for the DL model file
    def test_dl_model_info(path: str = "best_finetuned_model_small_to_large.h5"):
        print("\n--- DL Model File Info ---")
        print(f"Path: {path}")
        if not os.path.exists(path):
            print("Exists: False (file not found)")
            return
        size_bytes = os.path.getsize(path)
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        print("Exists: True")
        print(f"Size: {size_bytes/1_048_576:.2f} MB")
        print(f"Modified: {mtime:%Y-%m-%d %H:%M:%S}")

        try:
            load_dl_model(path)
        except Exception as e:
            print("Load error:", e)
            return

        try:
            print("\n--- DL Model Summary ---")
            dl_model.summary()  # prints to stdout
        except Exception as e:
            print("Summary error:", e)

        try:
            inputs = getattr(dl_model, 'inputs', None)
            outputs = getattr(dl_model, 'outputs', None)
            if inputs:
                print("Input shapes:", [tuple(getattr(t, 'shape', ())) for t in inputs])
            if outputs:
                print("Output shapes:", [tuple(getattr(t, 'shape', ())) for t in outputs])
            if hasattr(dl_model, 'layers'):
                print("Layers:", len(dl_model.layers))
        except Exception as e:
            print("Introspection error:", e)

    # Run the DL model info test
    test_dl_model_info()

