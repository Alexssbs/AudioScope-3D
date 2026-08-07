from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from modules.ai_summary import ask_interpreter, build_context
from modules.audio_features import extract_features
from modules.audio_loader import AudioData, load_audio
from modules.comparison import comparison_table
from modules.export import features_csv, features_excel, features_json
from modules.fft_analysis import compute_fft
from modules.spectrogram import compute_spectrogram
from modules.visualization import fft_figure, spectrogram_figure, waveform_figure

# ============================================
# CONFIGURACIÓN DE PÁGINA
# ============================================
st.set_page_config(
    page_title="AudioScope 3D", 
    page_icon="🎧", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# ============================================
# DETECTAR MODO (LIGHT/DARK)
# ============================================
def get_theme_mode() -> str:
    """Detecta si el tema es claro u oscuro"""
    try:
        # Intentar obtener del session_state
        if 'theme_mode' in st.session_state:
            return st.session_state.theme_mode
        
        # Si no está configurado, usar dark por defecto
        return "dark"
    except:
        return "dark"

def set_theme_mode(mode: str):
    """Cambia el tema y actualiza la interfaz"""
    st.session_state.theme_mode = mode
    st.rerun()

# ============================================
# CSS DINÁMICO (SOPORTA LIGHT/DARK)
# ============================================
def get_css(mode: str) -> str:
    """Retorna CSS según el modo"""
    if mode == "dark":
        return """
        <style>
            .stApp { background: #0b0e14; } 
            .block-container { max-width: 1450px; }
            div[data-testid='stMetric'] { 
                background: #151a26; 
                border: 1px solid #252d40; 
                border-radius: 12px; 
                padding: 14px; 
            }
            .hero { 
                padding: 2.4rem; 
                border: 1px solid #25314a; 
                border-radius: 18px; 
                background: linear-gradient(120deg,#111827,#17113b); 
                color: #e6edf3;
            }
            .realtime-container {
                background: #0d1117;
                border: 1px solid #30363d;
                border-radius: 12px;
                padding: 20px;
                margin: 10px 0;
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 2px;
            }
            .stTabs [data-baseweb="tab"] {
                background-color: #161b22;
                border-radius: 4px 4px 0px 0px;
                padding: 8px 16px;
                color: #8b949e;
            }
            .stTabs [aria-selected="true"] {
                background-color: #1f2937;
                color: #e6edf3;
            }
        </style>
        """
    else:  # light mode
        return """
        <style>
            .stApp { background: #f6f8fa; } 
            .block-container { max-width: 1450px; }
            div[data-testid='stMetric'] { 
                background: #ffffff; 
                border: 1px solid #d0d7de; 
                border-radius: 12px; 
                padding: 14px; 
            }
            .hero { 
                padding: 2.4rem; 
                border: 1px solid #d0d7de; 
                border-radius: 18px; 
                background: linear-gradient(120deg,#f6f8fa,#ffffff); 
                color: #24292f;
            }
            .realtime-container {
                background: #ffffff;
                border: 1px solid #d0d7de;
                border-radius: 12px;
                padding: 20px;
                margin: 10px 0;
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 2px;
            }
            .stTabs [data-baseweb="tab"] {
                background-color: #f6f8fa;
                border-radius: 4px 4px 0px 0px;
                padding: 8px 16px;
                color: #57606a;
                border: 1px solid #d0d7de;
                border-bottom: none;
            }
            .stTabs [aria-selected="true"] {
                background-color: #ffffff;
                color: #24292f;
                border-bottom: 2px solid #0969da;
            }
        </style>
        """

# ============================================
# FUNCIONES DE CARGA Y ANÁLISIS
# ============================================
@st.cache_data(show_spinner=False)
def decode_audio(payload: bytes, name: str) -> AudioData:
    return load_audio(BytesIO(payload), name, len(payload))

@st.cache_data(show_spinner=False)
def analyze(samples: np.ndarray, sample_rate: int, n_fft: int, max_frequency: float, window: str):
    """Análisis optimizado con reducción de datos para visualización"""
    features = extract_features(samples, sample_rate, n_fft)
    frequencies, magnitude = compute_fft(samples, sample_rate, n_fft, max_frequency)
    spec_frequencies, times, spec_db = compute_spectrogram(samples, sample_rate, n_fft, max_frequency, window)
    
    # REDUCCIÓN DE DATOS
    max_fft_points = 2000
    if len(frequencies) > max_fft_points:
        step = len(frequencies) // max_fft_points
        frequencies_reduced = frequencies[::step]
        magnitude_reduced = magnitude[::step]
    else:
        frequencies_reduced = frequencies
        magnitude_reduced = magnitude
    
    max_spec_freq = 250
    max_spec_time = 250
    
    if len(spec_frequencies) > max_spec_freq:
        step_f = len(spec_frequencies) // max_spec_freq
        spec_freq_reduced = spec_frequencies[::step_f]
        spec_db_reduced = spec_db[::step_f, :]
    else:
        spec_freq_reduced = spec_frequencies
        spec_db_reduced = spec_db
    
    if len(times) > max_spec_time:
        step_t = len(times) // max_spec_time
        spec_time_reduced = times[::step_t]
        spec_db_reduced = spec_db_reduced[:, ::step_t]
    else:
        spec_time_reduced = times
    
    return features, frequencies_reduced, magnitude_reduced, spec_freq_reduced, spec_time_reduced, spec_db_reduced

# ============================================
# FUNCIONES DE RENDERIZADO
# ============================================
def format_size(size: int) -> str:
    return f"{size / 1_048_576:.2f} MB" if size else "No disponible"

def render_metrics(features: dict) -> None:
    columns = st.columns(3)
    
    metrics = [
        ("RMS", f"{features['rms']:.4f}"),
        ("Energía", f"{features['energy']:.2f}"),
        ("Centroide", f"{features['spectral_centroid_hz']:.0f} Hz"),
        ("Frecuencia dominante", f"{features['dominant_frequency_hz']:.0f} Hz"),
        ("ZCR", f"{features['zero_crossing_rate']:.4f}"),
        ("Silencio", f"{features['silence_seconds']:.2f} s"),
        ("Relación de silencio", f"{features['silence_ratio']:.1%}"),
        ("Pico", f"{features['peak_amplitude']:.3f}"),
    ]
    
    for i, (label, value) in enumerate(metrics):
        columns[i % 3].metric(label, value)

def render_audio_information(audio: AudioData) -> None:
    st.subheader("Información del audio")
    
    cols = st.columns(3)
    
    info_items = [
        ("🎵 Nombre", audio.name),
        ("⏱️ Duración", f"{audio.duration:.2f} s"),
        ("📊 Muestreo", f"{audio.sample_rate:,} Hz"),
        ("🔊 Canales", audio.channels),
        ("📝 Muestras", f"{len(audio.samples):,}"),
        ("💾 Tamaño", format_size(audio.size_bytes)),
        ("📁 Formato", audio.format),
    ]
    
    for i, (label, value) in enumerate(info_items):
        cols[i % 3].metric(label, value)
    
    if audio.duration > 120:
        st.warning(f"⚠️ El audio dura {audio.duration:.1f} segundos. Para mejor rendimiento, usa audios de menos de 2 minutos.")

# ============================================
# FUNCIÓN DE TIEMPO REAL (CORREGIDA)
# ============================================
def render_realtime_simulation(samples: np.ndarray, sample_rate: int, n_fft: int = 2048, max_freq: int = 8000):
    """
    Simulación de visualización en tiempo real para Streamlit.
    Cada vez que se presiona el botón, avanza la posición del audio y actualiza los gráficos.
    """
    st.markdown("### 🎵 Visualización en tiempo real")
    st.markdown("*Cada vez que presiones el botón, avanza la posición del audio y actualiza los gráficos.*")
    
    # Inicializar posición en session_state
    if 'realtime_pos' not in st.session_state:
        st.session_state.realtime_pos = 0
    
    if 'realtime_playing' not in st.session_state:
        st.session_state.realtime_playing = False
    
    # Calcular duración total
    total_duration = len(samples) / sample_rate
    max_pos = max(0, total_duration - 0.1)  # Evitar que el slider se trabe
    
    # Controles
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("▶️ Avanzar", key="avanzar_realtime"):
            # Avanzar 2 segundos
            nueva_pos = min(st.session_state.realtime_pos + 2, max_pos)
            st.session_state.realtime_pos = nueva_pos
            st.session_state.realtime_playing = True
            st.rerun()
    
    with col2:
        if st.button("⏹️ Reiniciar", key="reiniciar_realtime"):
            st.session_state.realtime_pos = 0
            st.session_state.realtime_playing = False
            st.rerun()
    
    with col3:
        # Slider con valores seguros
        if max_pos > 0:
            pos = st.slider(
                "Posición (segundos)", 
                0.0, 
                float(max_pos),  # Ahora max_pos siempre > min
                float(st.session_state.realtime_pos) if st.session_state.realtime_pos <= max_pos else 0.0,
                key="slider_realtime",
                step=0.1
            )
            if pos != st.session_state.realtime_pos:
                st.session_state.realtime_pos = pos
                st.rerun()
        else:
            st.warning("⚠️ El audio es muy corto para la simulación en tiempo real")
            return
    
    # Calcular posición en muestras
    pos_samples = int(st.session_state.realtime_pos * sample_rate)
    
    # Verificar que la posición no exceda el límite
    if pos_samples >= len(samples):
        pos_samples = max(0, len(samples) - n_fft)
        st.session_state.realtime_pos = pos_samples / sample_rate
    
    # Obtener segmento actual
    chunk_size = min(n_fft, len(samples) - pos_samples)
    
    if chunk_size > 0:
        chunk = samples[pos_samples:pos_samples + chunk_size]
        
        # Mostrar información de posición
        progress = (st.session_state.realtime_pos / total_duration) * 100 if total_duration > 0 else 0
        
        st.progress(progress / 100, text=f"Progreso: {progress:.1f}% ({st.session_state.realtime_pos:.1f}s / {total_duration:.1f}s)")
        
        # Calcular espectro del segmento actual
        from modules.fft_analysis import compute_fft
        
        # Asegurar que chunk tenga tamaño suficiente
        if len(chunk) < n_fft:
            chunk_padded = np.pad(chunk, (0, n_fft - len(chunk)))
            freqs, mag = compute_fft(chunk_padded, sample_rate, n_fft, max_freq)
        else:
            freqs, mag = compute_fft(chunk, sample_rate, n_fft, max_freq)
        
        # Mostrar gráficos en dos columnas
        col1, col2 = st.columns(2)
        
        with col1:
            # Forma de onda del segmento
            time = np.arange(len(chunk)) / sample_rate + st.session_state.realtime_pos
            fig_wave = go.Figure(go.Scattergl(
                x=time, 
                y=chunk, 
                mode='lines', 
                line={'color': '#65d9ff', 'width': 1.5},
                name='Segmento actual'
            ))
            
            # Agregar línea de posición en el audio completo (solo si el audio no es muy largo)
            if len(samples) < 100000:  # Solo para audios cortos
                time_full = np.arange(len(samples)) / sample_rate
                fig_wave.add_trace(go.Scattergl(
                    x=time_full,
                    y=samples * 0.3,
                    mode='lines',
                    line={'color': '#4a4a6a', 'width': 0.5},
                    opacity=0.3,
                    name='Audio completo'
                ))
            
            # Marcar posición actual
            fig_wave.add_vline(
                x=st.session_state.realtime_pos, 
                line_color="#ff6b6b", 
                line_width=2,
                annotation_text="📍 Posición"
            )
            
            fig_wave.update_layout(
                template='plotly_dark',
                title=f'Forma de onda - Posición {st.session_state.realtime_pos:.1f}s',
                xaxis_title='Tiempo (s)',
                yaxis_title='Amplitud',
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_wave, use_container_width=True)
        
        with col2:
            # Espectro del segmento actual
            fig_spec = go.Figure(go.Scattergl(
                x=freqs, 
                y=mag, 
                mode='lines',
                line={'color': '#a78bfa', 'width': 1.5},
                fill='tozeroy',
                fillcolor='rgba(167, 139, 250, 0.2)',
                name='Espectro actual'
            ))
            
            fig_spec.update_layout(
                template='plotly_dark',
                title=f'Espectro en tiempo real - {st.session_state.realtime_pos:.1f}s',
                xaxis_title='Frecuencia (Hz)',
                yaxis_title='Magnitud (dB)',
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_spec, use_container_width=True)
        
        # Espectrograma 3D actualizado
        st.markdown("#### 🌊 Espectrograma 3D en tiempo real")
        
        from modules.spectrogram import compute_spectrogram
        from modules.visualization import spectrogram_figure
        
        # Tomar una ventana de 5 segundos alrededor de la posición
        window_start = max(0, pos_samples - int(2 * sample_rate))
        window_end = min(len(samples), pos_samples + int(3 * sample_rate))
        window_samples = samples[window_start:window_end]
        
        if len(window_samples) > n_fft:
            spec_freq, spec_time, spec_db = compute_spectrogram(
                window_samples, sample_rate, n_fft, max_freq, 'hann'
            )
            
            # Ajustar tiempos para que coincidan con la posición
            if len(spec_time) > 0:
                spec_time = spec_time + window_start / sample_rate
                
                fig_3d = spectrogram_figure(spec_freq, spec_time, spec_db, 'Plasma', is_3d=True)
                
                # Marcar posición actual en el espectrograma
                if len(spec_freq) > 0:
                    fig_3d.add_trace(go.Scatter3d(
                        x=[st.session_state.realtime_pos] * 10,
                        y=np.linspace(0, min(max_freq, spec_freq[-1]), 10),
                        z=np.linspace(-80, 0, 10),
                        mode='lines',
                        line=dict(color='red', width=3),
                        name='📍 Posición actual'
                    ))
                
                fig_3d.update_layout(height=400)
                st.plotly_chart(fig_3d, use_container_width=True)
        
    else:
        st.info("🎵 Fin del audio. Presiona 'Reiniciar' para comenzar de nuevo.")

# ============================================
# FUNCIÓN MAIN
# ============================================
def main() -> None:
    # Inicializar tema
    if 'theme_mode' not in st.session_state:
        st.session_state.theme_mode = "dark"
    
    theme = get_theme_mode()
    
    # Aplicar CSS dinámico
    st.markdown(get_css(theme), unsafe_allow_html=True)
    
    with st.sidebar:
        st.title("🎧 AudioScope 3D")
        st.caption("Análisis DSP e interpretación asistida")
        
        # Selector de tema
        col_theme1, col_theme2 = st.columns(2)
        with col_theme1:
            if st.button("🌙 Dark", use_container_width=True, type="primary" if theme == "dark" else "secondary"):
                set_theme_mode("dark")
        with col_theme2:
            if st.button("☀️ Light", use_container_width=True, type="primary" if theme == "light" else "secondary"):
                set_theme_mode("light")
        
        st.divider()
        
        source_type = st.radio("Origen", ["Subir archivo", "Audio de demostración"], horizontal=True)
        
        demo_files = [path for path in Path("assets").glob("*") if path.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}]
        
        uploaded = None
        selected_demo = None
        
        if source_type == "Subir archivo":
            uploaded = st.file_uploader("Archivo principal", type=["wav", "mp3", "flac", "ogg", "m4a"])
        elif demo_files:
            selected_demo = st.selectbox("Audio de demostración", demo_files, format_func=lambda path: path.name)
        else:
            st.caption("Agrega `music.mp3`, `music2.mp3` o `voice.wav` en `assets/` para habilitar demos.")
        
        st.divider()
        st.subheader("⚙️ Parámetros de análisis")
        n_fft = st.select_slider("Tamaño FFT", options=[512, 1024, 2048, 4096], value=2048)
        max_frequency = st.slider("Frecuencia máxima (Hz)", 1000, 22050, 8000, 500)
        window = st.selectbox("Ventana", ["hann", "hamming", "blackman", "bartlett"])
        colorscale = st.selectbox("Mapa de colores", ["Plasma", "Viridis", "Inferno", "Cividis", "Turbo"])
        
        st.divider()
        st.subheader("🔑 Configuración IA")
        # El token se maneja en Secrets
        token = st.secrets.get("HUGGINGFACE_TOKEN")
        token_status = "✅ Configurado" if token else "❌ No configurado"
        st.caption(f"Estado Hugging Face: {token_status}")
        if not token:
            st.caption("💡 Crea `.streamlit/secrets.toml` con:\n`HUGGINGFACE_TOKEN='tu_token'`")
        
        st.caption("💡 Usa los controles de Plotly para zoom, desplazamiento, rotación y guardar imágenes.")

    # Pantalla de inicio
    if uploaded is None and selected_demo is None:
        st.markdown(f"""
        <div class='hero'>
            <h1>🎧 AudioScope 3D</h1>
            <p>Explora tus audios con FFT, espectrogramas interactivos, métricas DSP y una capa de interpretación en lenguaje natural.</p>
            <p style='color: #8899bb; font-size: 0.9rem;'>
                📌 Carga un audio desde la barra lateral para comenzar.<br>
                📌 Para demos, agrega tus propios archivos a la carpeta <code>assets/</code>
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Cargar audio
    try:
        payload = uploaded.getvalue() if uploaded else selected_demo.read_bytes()
        audio_name = uploaded.name if uploaded else selected_demo.name
        audio = decode_audio(payload, audio_name)
    except Exception as error:
        st.error(f"❌ No se pudo leer el archivo: {error}")
        return
    
    # Verificar duración
    if audio.duration > 300:
        st.warning(f"⚠️ El audio dura {audio.duration:.1f} segundos. Recomendación: menos de 2-3 minutos.")
    
    # Análisis
    safe_max = min(max_frequency, audio.sample_rate // 2)
    
    with st.spinner("🔍 Analizando audio..."):
        features, frequencies, magnitude, spec_frequencies, times, spec_db = analyze(
            audio.samples, audio.sample_rate, n_fft, safe_max, window
        )

    # Tabs principales
    st.title("📊 Panel de análisis")
    st.audio(payload)
    
    overview, spectral, realtime, characteristics, compare, export, assistant = st.tabs([
        "📋 Resumen", 
        "📈 Espectro", 
        "🎵 Tiempo Real",
        "📊 Características", 
        "🔀 Comparar", 
        "💾 Exportar", 
        "🤖 IA"
    ])

    with overview:
        render_audio_information(audio)
        st.plotly_chart(
            waveform_figure(audio.samples, audio.sample_rate), 
            use_container_width=True, 
            config={"scrollZoom": True, "displayModeBar": True}
        )
    
    with spectral:
        st.plotly_chart(fft_figure(frequencies, magnitude), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                spectrogram_figure(spec_frequencies, times, spec_db, colorscale, is_3d=False), 
                use_container_width=True
            )
        with col2:
            st.plotly_chart(
                spectrogram_figure(spec_frequencies, times, spec_db, colorscale, is_3d=True), 
                use_container_width=True
            )
    
    with realtime:
        render_realtime_simulation(
            audio.samples, 
            audio.sample_rate, 
            n_fft, 
            safe_max
        )
    
    with characteristics:
        render_metrics(features)
        st.caption("""
        📌 **Interpretación de métricas**:
        - **RMS**: Intensidad promedio del audio
        - **Energía**: Potencia total de la señal
        - **Centroide**: "Brillantez" del sonido (frecuencia promedio)
        - **Frecuencia dominante**: Frecuencia más presente
        - **ZCR**: Tasa de cruce por cero (ruidosidad)
        - **Silencio**: Porcentaje de silencio en el audio
        """)
    
    with compare:
        st.subheader("🔀 Comparación de dos audios")
        secondary = st.file_uploader("Segundo archivo", type=["wav", "mp3", "flac", "ogg", "m4a"], key="secondary")
        
        if secondary:
            try:
                secondary_audio = decode_audio(secondary.getvalue(), secondary.name)
                second_features, *_ = analyze(
                    secondary_audio.samples, 
                    secondary_audio.sample_rate, 
                    n_fft, 
                    min(safe_max, secondary_audio.sample_rate // 2), 
                    window
                )
                
                table = comparison_table(features, second_features, audio.name, secondary_audio.name)
                st.dataframe(table, use_container_width=True, hide_index=True)
                
                radar_keys = ["rms", "zero_crossing_rate", "spectral_centroid_hz", "dominant_frequency_hz"]
                labels = ["RMS", "ZCR", "Centroide", "Frec. dominante"]
                
                maxima = [max(float(features[key]), float(second_features[key]), 1e-9) for key in radar_keys]
                first_values = [float(features[key]) / maximum for key, maximum in zip(radar_keys, maxima)]
                second_values = [float(second_features[key]) / maximum for key, maximum in zip(radar_keys, maxima)]
                
                radar = go.Figure([
                    go.Scatterpolar(
                        r=first_values + [first_values[0]], 
                        theta=labels + [labels[0]], 
                        fill="toself", 
                        name=audio.name
                    ),
                    go.Scatterpolar(
                        r=second_values + [second_values[0]], 
                        theta=labels + [labels[0]], 
                        fill="toself", 
                        name=secondary_audio.name
                    )
                ])
                radar.update_layout(
                    template="plotly_dark", 
                    title="📊 Radar comparativo (normalizado)",
                    polar={"radialaxis": {"visible": True, "range": [0, 1]}}
                )
                st.plotly_chart(radar, use_container_width=True)
                
            except Exception as error:
                st.error(f"❌ No se pudo analizar el segundo archivo: {error}")
        else:
            st.info("📌 Carga un segundo audio para ver diferencias y el radar comparativo.")
    
    with export:
        st.subheader("💾 Exportar resultados")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                "📄 JSON", 
                features_json(features), 
                f"{Path(audio.name).stem}_analysis.json", 
                "application/json"
            )
        with col2:
            st.download_button(
                "📊 CSV", 
                features_csv(features), 
                f"{Path(audio.name).stem}_analysis.csv", 
                "text/csv"
            )
        with col3:
            st.download_button(
                "📑 Excel", 
                features_excel(features), 
                f"{Path(audio.name).stem}_analysis.xlsx", 
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        st.caption("💡 Las figuras Plotly incluyen el botón de cámara 📷 para exportarlas como PNG.")
    
    with assistant:
        st.subheader("🤖 Asistente de interpretación")
        st.caption("🧠 Llama recibe solamente las métricas ya calculadas; nunca se envía el audio a la IA.")
        
        question = st.text_area(
            "❓ Pregunta sobre los resultados", 
            placeholder="Ej: ¿Qué significa este espectrograma? ¿Cómo describirías la calidad del audio?"
        )
        
        if st.button("💬 Interpretar con IA", type="primary", disabled=not question):
            # ====== AQUÍ SE USA EL TOKEN DE HUGGING FACE ======
            token = st.secrets.get("HUGGINGFACE_TOKEN")
            if not token:
                st.warning("⚠️ Configura HUGGINGFACE_TOKEN en los Secrets de Streamlit antes de usar la IA.")
                st.info("📌 Para configurar:\n1. Crea archivo `.streamlit/secrets.toml`\n2. Agrega: `HUGGINGFACE_TOKEN='tu_token_aqui'`")
            else:
                with st.spinner("🤔 Generando interpretación..."):
                    try:
                        answer = ask_interpreter(
                            token, 
                            build_context(audio.name, features), 
                            question
                        )
                        st.markdown(f"**Respuesta:**\n\n{answer}")
                    except Exception as error:
                        st.error(f"❌ La API de Hugging Face no respondió: {error}")


if __name__ == "__main__":
    main()