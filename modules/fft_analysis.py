from __future__ import annotations

import numpy as np
from scipy.fft import fft, fftfreq


def compute_fft(samples: np.ndarray, sample_rate: int, n_fft: int, max_frequency: float) -> tuple[np.ndarray, np.ndarray]:
    """Refactor of the original Hann-windowed FFT spectrum calculation."""
    segment = samples[:n_fft]
    if len(segment) < n_fft:
        segment = np.pad(segment, (0, n_fft - len(segment)))
    window = np.hanning(len(segment))
    fft_data = fft(segment * window)
    frequencies = fftfreq(len(fft_data), 1 / sample_rate)[: len(fft_data) // 2]
    magnitude = np.abs(fft_data[: len(fft_data) // 2]) * 2 / len(segment)
    magnitude_db = np.clip(20 * np.log10(magnitude + 1e-10), -80, 0)
    cutoff = np.searchsorted(frequencies, min(max_frequency, sample_rate / 2))
    return frequencies[:cutoff], magnitude_db[:cutoff]


def dominant_frequency(samples: np.ndarray, sample_rate: int, n_fft: int) -> float:
    frequencies, magnitude = compute_fft(samples, sample_rate, n_fft, sample_rate / 2)
    return float(frequencies[np.argmax(magnitude)]) if len(magnitude) else 0.0
