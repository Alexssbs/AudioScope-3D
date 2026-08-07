from __future__ import annotations

import librosa
import numpy as np

from modules.fft_analysis import dominant_frequency


def extract_features(samples: np.ndarray, sample_rate: int, n_fft: int) -> dict[str, float | int | list[float]]:
    rms_frames = librosa.feature.rms(y=samples, frame_length=n_fft, hop_length=n_fft // 4)[0]
    centroid = librosa.feature.spectral_centroid(y=samples, sr=sample_rate, n_fft=n_fft)[0]
    zcr = librosa.feature.zero_crossing_rate(samples, frame_length=n_fft)[0]
    intervals = librosa.effects.split(samples, top_db=35, frame_length=n_fft)
    active_seconds = sum((end - start) / sample_rate for start, end in intervals)
    silence_seconds = max(0.0, len(samples) / sample_rate - active_seconds)
    return {
        "rms": float(np.sqrt(np.mean(np.square(samples)))) if len(samples) else 0.0,
        "energy": float(np.sum(np.square(samples))),
        "zero_crossing_rate": float(np.mean(zcr)),
        "spectral_centroid_hz": float(np.mean(centroid)),
        "dominant_frequency_hz": dominant_frequency(samples, sample_rate, n_fft),
        "silence_seconds": float(silence_seconds),
        "silence_ratio": float(silence_seconds / (len(samples) / sample_rate)) if len(samples) else 0.0,
        "peak_amplitude": float(np.max(np.abs(samples))) if len(samples) else 0.0,
    }
