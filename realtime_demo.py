# realtime_demo.py
import sounddevice as sd
import numpy as np
import collections
import threading
import time
import matplotlib.pyplot as plt
import soundfile as sf
import random
from queue import Queue
from time import perf_counter

SR = 16000  # choose sample rate (must match models or resample)
WINDOW_SEC = 1.0
HOP_SEC = 0.5
WINDOW_SAMPLES = int(WINDOW_SEC * SR)
HOP_SAMPLES = int(HOP_SEC * SR)
BUFFER_MAX_SECONDS = 10
BUFFER_MAX_SAMPLES = BUFFER_MAX_SECONDS * SR

# labels used for demo
LABELS = ["laughing", "coughing", "clapping", "knocking", "alarm"]

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

# mock model functions (replace with real model inference)
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

def start_inference_threads(window):
    # kick off ML and DL inference in background threads; they will write
    # results into latest_ml/latest_dl protected by results_lock
    def run_ml():
        nonlocal window
        label, prob, lat = mock_ml_predict(window)
        with results_lock:
            global latest_ml
            latest_ml = (label, prob, lat)

    def run_dl():
        nonlocal window
        label, prob, lat = mock_dl_predict(window)
        with results_lock:
            global latest_dl
            latest_dl = (label, prob, lat)

    t1 = threading.Thread(target=run_ml, daemon=True)
    t2 = threading.Thread(target=run_dl, daemon=True)
    t1.start(); t2.start()

# main
def main():
    print("running main \n")
    stop_event = threading.Event()
    # start windowing worker
    win_thread = threading.Thread(target=windowing_worker, args=(stop_event,), daemon=True)
    win_thread.start()
    # Create Matplotlib figure on the main thread and use FuncAnimation to
    # update the plot. The animation callback polls the window_queue and
    # starts background inference threads when a new window is available.
    from matplotlib.animation import FuncAnimation

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
    with sd.InputStream(channels=1, samplerate=SR, blocksize=1024, callback=sd_callback):
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