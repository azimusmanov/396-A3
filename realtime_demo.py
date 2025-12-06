# realtime_demo.py

# Imports
import sounddevice as sd
import numpy as np
import collections
from collections import Counter
import threading
import time
import matplotlib.pyplot as plt
import soundfile as sf
from queue import Queue
from time import perf_counter
import librosa

# Importing models
from models import (
    load_ml_model,
    ml_predict,
    load_dl_model,
    dl_predict,
)
from extract_features import extract_features, is_silent

SR = 16000  # model sample rate (we'll resample to this if needed)
WINDOW_SEC = 1.0
HOP_SEC = 0.5
# These depend on the capture sample rate and will be initialized in main()
CAPTURE_SR = None
WINDOW_SAMPLES = None
HOP_SAMPLES = None
BUFFER_MAX_SECONDS = 10
BUFFER_MAX_SAMPLES = None

# Thread-safe deque for audio
audio_deque = collections.deque(maxlen=BUFFER_MAX_SAMPLES)
deque_lock = threading.Lock()

# queue for windows for processing and ui update
window_queue = Queue(maxsize=10)

# sounddevice callback: converts incoming frames to numpy and appends to deque
def sd_callback(indata, frames, time_info, status):
    if status:
        print("Stream status:", status)
    # ensure mono
    arr = indata.copy()
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    # protect deque with a lock
    with deque_lock:
        audio_deque.extend(arr.tolist())

# worker: extracts windows at hop rate and pushes to window_queue
def windowing_worker(stop_event):
    next_time = time.time()
    while not stop_event.is_set():
        # ensure enough samples
        with deque_lock:
            cur_len = len(audio_deque)
            if cur_len >= WINDOW_SAMPLES:
                # get last WINDOW_SAMPLES
                tail = list(audio_deque)[-WINDOW_SAMPLES:]
                window = np.asarray(tail, dtype=np.float32)
                # push to queue (non-blocking)
                try:
                    window_queue.put_nowait(window)
                except:
                    pass
                # drop HOP_SAMPLES oldest to simulate hop
                for _ in range(HOP_SAMPLES):
                    if audio_deque:
                        audio_deque.popleft()
        # sleep until next hop
        time.sleep(HOP_SEC * 0.9)


# visualization thread using Matplotlib
# We'll update the plot using Matplotlib's animation on the main thread.
# The animation callback will poll `window_queue` for new windows and
# trigger model inference in background threads so the GUI remains responsive.

# shared state for visualization and results
latest_window = None
latest_window_lock = threading.Lock()
latest_ml = None
latest_dl = None
results_lock = threading.Lock()
# keep a module-level reference to the animation to avoid GC
_anim_ref = None

# simple temporal smoothing for ML labels
ml_history = collections.deque(maxlen=5)
ml_history_lock = threading.Lock()

def start_inference_threads(window):
    # kick off ML and DL inference in background threads; they will write
    # results into latest_ml/latest_dl protected by results_lock
    def run_ml():
        nonlocal window
        global latest_ml
        if is_silent(window):
            with results_lock:
                latest_ml = ("silence", 0.0, 0.0)
            # clear ML history to avoid sticky labels
            with ml_history_lock:
                ml_history.clear()
            return
        # resample to model SR if capture SR differs
        win_for_model = window if CAPTURE_SR == SR else librosa.resample(window, orig_sr=CAPTURE_SR, target_sr=SR)
        features = extract_features(win_for_model, sr=SR)
        label, prob, lat = ml_predict(features)
        with results_lock:
            latest_ml = (label, prob, lat)
        # update history for smoothing
        with ml_history_lock:
            ml_history.append((label, prob))

    def run_dl():
        nonlocal window
        global latest_dl
        if is_silent(window):
            with results_lock:
                latest_dl = ("silence", 0.0, 0.0)
            # optional: do not clear DL state history (none used)
            return
        # resample to model SR if capture SR differs
        win_for_model = window if CAPTURE_SR == SR else librosa.resample(window, orig_sr=CAPTURE_SR, target_sr=SR)
        label, prob, lat = dl_predict(win_for_model, sr=SR)
        with results_lock:
            latest_dl = (label, prob, lat)

    t1 = threading.Thread(target=run_ml, daemon=True)
    t2 = threading.Thread(target=run_dl, daemon=True)
    t1.start(); t2.start()

# main
def main():
    print("running main \n")
    stop_event = threading.Event()
    # Determine a supported capture sample rate for the selected device
    device_index = 18  # Focusrite WASAPI device selected earlier
    try:
        dev_info = sd.query_devices(device_index)
        default_sr = dev_info.get('default_samplerate', None)
    except Exception:
        default_sr = None

    # Try common sample rates if default is unavailable
    candidate_srs = [default_sr] if default_sr else []
    candidate_srs += [16000, 48000, 44100]
    chosen_sr = None
    for cand in candidate_srs:
        try:
            if cand is None:
                continue
            sd.check_input_settings(device=device_index, samplerate=cand, channels=1)
            chosen_sr = int(cand)
            break
        except Exception:
            continue
    if chosen_sr is None:
        # fallback to library default
        chosen_sr = int(sd.query_devices(device_index)['default_samplerate'])

    # initialize globals based on capture sample rate
    global CAPTURE_SR, WINDOW_SAMPLES, HOP_SAMPLES, BUFFER_MAX_SAMPLES
    CAPTURE_SR = chosen_sr
    WINDOW_SAMPLES = int(WINDOW_SEC * CAPTURE_SR)
    HOP_SAMPLES = int(HOP_SEC * CAPTURE_SR)
    BUFFER_MAX_SAMPLES = BUFFER_MAX_SECONDS * CAPTURE_SR

    # start windowing worker
    win_thread = threading.Thread(target=windowing_worker, args=(stop_event,), daemon=True)
    win_thread.start()
    # Create Matplotlib figure on the main thread and use FuncAnimation to
    # update the plot. The animation callback polls the window_queue and
    # starts background inference threads when a new window is available.
    from matplotlib.animation import FuncAnimation
    load_ml_model()
    load_dl_model()
    # Print ML class/activities mapping once
    try:
        from models import ml_classes, ml_activities
        print("ML classes:", ml_classes)
        print("ML activities:", ml_activities)
    except Exception:
        pass
    plt.ion()
    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.linspace(-WINDOW_SEC, 0, WINDOW_SAMPLES)
    line, = ax.plot(x, np.zeros(WINDOW_SAMPLES))
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("Time (s)")
    ax.set_title("Live audio window")
    text_ml = ax.text(0.01, 0.95, "", transform=ax.transAxes, va="top")
    text_dl = ax.text(0.01, 0.85, "", transform=ax.transAxes, va="top")

    def update(frame):
        # try to consume latest window (non-blocking)
        try:
            window = window_queue.get_nowait()
        except:
            window = None

        if window is not None:
            # store latest window for plotting
            with latest_window_lock:
                global latest_window
                latest_window = window
            # start background inference
            start_inference_threads(window)

        # update plot
        with latest_window_lock:
            w = latest_window
        if w is not None:
            line.set_ydata(w)
        # update texts from latest results
        with results_lock:
            ml = latest_ml
            dl = latest_dl
        if ml is not None:
            # apply simple majority smoothing over recent ML labels
            with ml_history_lock:
                if ml_history:
                    counts = Counter([lbl for lbl, _ in ml_history])
                    smoothed_label = counts.most_common(1)[0][0]
                    # show avg confidence for the smoothed label
                    confs = [c for lbl, c in ml_history if lbl == smoothed_label]
                    avg_conf = sum(confs) / len(confs) if confs else ml[1]
                    text_ml.set_text(f"ML: {smoothed_label} ({avg_conf:.2f}) {ml[2]:.1f} ms")
                else:
                    text_ml.set_text(f"ML: {ml[0]} ({ml[1]:.2f}) {ml[2]:.1f} ms")
        if dl is not None:
            text_dl.set_text(f"DL: {dl[0]} ({dl[1]:.2f}) {dl[2]:.1f} ms")

        return line, text_ml, text_dl

    # disable caching of frame data to avoid unbounded memory use warning
    ani = FuncAnimation(fig, update, interval=int(HOP_SEC*1000/2), cache_frame_data=False)
    # keep a reference to the animation so it isn't garbage-collected
    # when this function yields control (Matplotlib may only hold a weakref)
    global _anim_ref
    _anim_ref = ani
    # explicitly start the animation event source to ensure it runs
    try:
        ani.event_source.start()
    except Exception:
        pass

    # open input stream and run GUI main loop
    # IMPORTANT! REMOVE DEVICE COMMAND IF YOU DO NOT HAVE EXTERNAL MICROPHONE. IF YOU DO, ENSURE YOU PICK THE CORRECT SOUNDDEVICE NUMBER
    with sd.InputStream(device=device_index, channels=1, samplerate=CAPTURE_SR, blocksize=1024, dtype='float32', callback=sd_callback):
        print("Recording... press Ctrl+C to stop")
        try:
            # block until the window is closed or interrupted
            plt.show(block=True)
        except KeyboardInterrupt:
            print("Stopping...")

    stop_event.set()
    win_thread.join(timeout=1.0)

if __name__ == "__main__":
    main()