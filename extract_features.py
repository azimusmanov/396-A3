import numpy as np
import librosa


def extract_fft(audio_signal, sr=16000):
    """
    Extract simple FFT-based features from an audio signal.

    Returns a 9-D vector consisting of:
    [total_power, mean_power, std_power, frac_low, frac_mid, frac_high,
     spectral_centroid, spectral_bandwidth, spectral_rolloff_85]
    """
    y = np.asarray(audio_signal, dtype=float)
    if y.size == 0:
        return np.zeros(9, dtype=float)

    mag = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), d=1.0 / sr)

    power = mag ** 2
    tot_pow = power.sum() + 1e-12

    def band_fraction(f_lo, f_hi):
        band = (freqs >= f_lo) & (freqs < f_hi)
        return power[band].sum() / tot_pow

    frac_low = band_fraction(0, 300)
    frac_mid = band_fraction(300, 3000)
    frac_high = band_fraction(3000, min(8000, sr / 2))

    mean_p = power.mean()
    std_p = power.std()

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    rolloff85 = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85).mean()

    return np.array(
        [tot_pow, mean_p, std_p, frac_low, frac_mid, frac_high, centroid, bandwidth, rolloff85],
        dtype=float,
    )


def extract_mfcc(audio_signal, sr=16000, n_mfcc=13):
    """
    Extract MFCCs and their per-coefficient mean/std (26-D).
    """
    y = np.asarray(audio_signal, dtype=float)
    if y.size == 0:
        return np.zeros(n_mfcc * 2, dtype=float)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)
    return np.concatenate([mfcc_mean, mfcc_std], axis=0).astype(float)


def extract_rms(audio_signal, sr=16000):
    """
    Extract RMS energy statistics (9-D):
    [mean, std, max, min, median, p25, p75, dynamic_range, coeff_var]
    """
    y = np.asarray(audio_signal, dtype=float)
    if y.size == 0:
        return np.zeros(9, dtype=float)

    rms = librosa.feature.rms(y=y).flatten()
    if rms.size == 0:
        return np.zeros(9, dtype=float)

    r_mean = rms.mean()
    r_std = rms.std()
    r_max = rms.max()
    r_min = rms.min()
    r_med = np.median(rms)
    r_p25 = np.percentile(rms, 25)
    r_p75 = np.percentile(rms, 75)
    dyn = r_max - r_min
    cv = r_std / (r_mean + 1e-12)

    return np.array([r_mean, r_std, r_max, r_min, r_med, r_p25, r_p75, dyn, cv], dtype=float)


def extract_features(window, sr=16000, feature_type="all"):
    """
    Extract features from an audio window.

    Args:
        window: 1-D numpy array of audio samples (mono)
        sr: sample rate of `window`
        feature_type: 'fft', 'mfcc', 'rms', or 'all'

    Returns:
        - If feature_type == 'all': a single 1-D numpy array concatenating
          [fft(9), mfcc(26), rms(9)] -> 44-D vector
        - Else: the corresponding 1-D numpy array
    """
    y = np.asarray(window, dtype=float)

    if feature_type == "fft":
        return extract_fft(y, sr)
    elif feature_type == "mfcc":
        return extract_mfcc(y, sr)
    elif feature_type == "rms":
        return extract_rms(y, sr)
    elif feature_type == "all":
        f_fft = extract_fft(y, sr)
        f_mfcc = extract_mfcc(y, sr)
        f_rms = extract_rms(y, sr)
        return np.concatenate([f_fft, f_mfcc, f_rms], axis=0)
    else:
        raise ValueError("feature_type must be one of 'fft', 'mfcc', 'rms', 'all'")