from __future__ import annotations

import librosa
import numpy as np


def compute_spectrogram(samples: np.ndarray, sample_rate: int, n_fft: int, max_frequency: float, window: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hop_length = n_fft // 4
    stft = librosa.stft(samples, n_fft=n_fft, hop_length=hop_length, window=window)
    db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(db.shape[1]), sr=sample_rate, hop_length=hop_length)
    cutoff = np.searchsorted(frequencies, min(max_frequency, sample_rate / 2))
    return frequencies[:cutoff], times, db[:cutoff, :]
