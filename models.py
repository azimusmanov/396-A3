import time
from time import perf_counter
import random
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from joblib import load

# labels used for demo
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

# Load ml model once on startup, make variables global so they can be accessed by ml_predict
def load_ml_model():
    # 1. Load the artifact
    artifact = load("svm_small_to_large.joblib")
    global ml_model, ml_scaler, ml_classes, ml_activities # setting global vars
    ml_model = artifact["model"] # svm
    ml_scaler = artifact["scaler"]
    ml_classes = artifact["classes"]
    ml_activities = artifact["activities"]
    # print(ml_model, ml_scaler, ml_classes, ml_activities)


def ml_predict(window):
    t0 = perf_counter()
    """
    window: numpy array of shape (feature_dim,)
    """
    # reshape to 2D because sklearn expects (1, D)
    x_feat = np.asarray(x_feat).reshape(1, -1)

    # IMPORTANT: scale using the SAME scaler
    x_scaled = scaler.transform(x_feat)

    # get probabilities
    probs = svm.predict_proba(x_scaled)[0]   # shape = (num_classes,)
    pred_idx = probs.argmax()                # index of top prediction
    pred_class = classes[pred_idx]           # integer label
    pred_name = activities[pred_class]       # human-readable name
    confidence = probs[pred_idx]             # probability
    t = (perf_counter() - t0) * 1000.0
    return pred_class, pred_name, confidence, t
    return label, float(prob), t

if __name__ == '__main__':
    load_ml_model()
    print("ml_model:", ml_model, "\n")
    print("ml_scaler:", ml_scaler, "\n")
    print("ml_classes:", ml_classes, "\n")
    print("ml_activities:", ml_activities, "\n")

