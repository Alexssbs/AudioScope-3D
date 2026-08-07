from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def dark_layout(title: str) -> dict:
    return {"template": "plotly_dark", "title": title, "paper_bgcolor": "#10131c", "plot_bgcolor": "#10131c", "margin": {"l": 45, "r": 20, "t": 50, "b": 40}}


def waveform_figure(samples: np.ndarray, sample_rate: int, title: str = "Forma de onda") -> go.Figure:
    time = np.arange(len(samples)) / sample_rate
    figure = go.Figure(go.Scattergl(x=time, y=samples, mode="lines", line={"color": "#65d9ff", "width": 1}))
    figure.update_layout(**dark_layout(title), xaxis_title="Tiempo (s)", yaxis_title="Amplitud", dragmode="pan")
    return figure


def fft_figure(frequencies: np.ndarray, magnitude_db: np.ndarray) -> go.Figure:
    figure = go.Figure(go.Scattergl(x=frequencies, y=magnitude_db, mode="lines", line={"color": "#a78bfa"}))
    figure.update_layout(**dark_layout("Espectro FFT"), xaxis_title="Frecuencia (Hz)", yaxis_title="Magnitud (dB)")
    return figure


def spectrogram_figure(frequencies: np.ndarray, times: np.ndarray, db: np.ndarray, colorscale: str, is_3d: bool = False) -> go.Figure:
    # Reducir datos para el espectrograma 3D
    if is_3d:
        # Reducción más agresiva para 3D
        step_f = max(1, len(frequencies) // 150)
        step_t = max(1, len(times) // 150)
        
        freq_reduced = frequencies[::step_f]
        time_reduced = times[::step_t]
        db_reduced = db[::step_f, ::step_t]
        
        figure = go.Figure(go.Surface(
            x=time_reduced, 
            y=freq_reduced, 
            z=db_reduced, 
            colorscale=colorscale,
            # Reducir aún más la calidad para mejor rendimiento
            opacity=0.9
        ))
        figure.update_layout(
            **dark_layout("Espectrograma 3D"),
            scene={
                "xaxis_title": "Tiempo (s)",
                "yaxis_title": "Frecuencia (Hz)",
                "zaxis_title": "dB",
                "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 0.5}}
            }
        )
    else:
        # Para 2D también reducimos
        step_f = max(1, len(frequencies) // 300)
        step_t = max(1, len(times) // 300)
        
        freq_reduced = frequencies[::step_f]
        time_reduced = times[::step_t]
        db_reduced = db[::step_f, ::step_t]
        
        figure = go.Figure(go.Heatmap(
            x=time_reduced, 
            y=freq_reduced, 
            z=db_reduced, 
            colorscale=colorscale,
            colorbar={"title": "dB"}
        ))
        figure.update_layout(
            **dark_layout("Espectrograma 2D"),
            xaxis_title="Tiempo (s)",
            yaxis_title="Frecuencia (Hz)"
        )
    return figure