from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import librosa
import numpy as np
import soundfile as sf


@dataclass
class AudioData:
    samples: np.ndarray
    sample_rate: int
    channels: int
    name: str
    size_bytes: int
    format: str
    bitrate: str | None = None

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate


def load_audio(source: str | Path | BinaryIO, name: str, size_bytes: int = 0, max_duration: float = 300) -> AudioData:
    """Load audio with a maximum duration limit"""
    try:
        info = sf.info(source)
        channels, audio_format = info.channels, info.format
    except RuntimeError:
        channels, audio_format = 1, Path(name).suffix.lstrip(".").upper() or "Unknown"
    if hasattr(source, "seek"):
        source.seek(0)
    samples, sample_rate = librosa.load(source, sr=None, mono=True)
    
    # Limitar duración
    max_samples = int(max_duration * sample_rate)
    if len(samples) > max_samples:
        samples = samples[:max_samples]
        st.warning(f"Audio truncado a {max_duration} segundos para mantener el rendimiento")
    
    peak = np.max(np.abs(samples)) if len(samples) else 0
    normalized = samples / peak if peak else samples
    return AudioData(normalized.astype(np.float32), sample_rate, channels, name, size_bytes, audio_format)


def audio_bytes(uploaded_file) -> bytes:
    """Return uploaded bytes and rewind the Streamlit file object."""
    payload = uploaded_file.getvalue()
    uploaded_file.seek(0)
    return payload
