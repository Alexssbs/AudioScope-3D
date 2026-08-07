# AudioScope 3D

Aplicación web de análisis de audio construida con Streamlit. Reutiliza una cadena DSP clásica —normalización, ventana Hann, FFT y magnitud en dB— y la adapta a una interfaz interactiva con Plotly. La IA no procesa archivos de audio: solo interpreta el resumen de métricas generado localmente.

## Funcionalidades

- Carga de WAV, MP3, FLAC, OGG y M4A, con reproductor web e información técnica.
- Forma de onda navegable, FFT, espectrograma 2D y superficie 3D interactiva.
- RMS, energía, ZCR, centroide espectral, frecuencia dominante, pico y estimación de silencios.
- Comparación de dos audios mediante tabla de diferencias y radar normalizado.
- Exportación de resultados a JSON, CSV y Excel; las figuras Plotly pueden guardarse como imagen desde su barra de herramientas.
- Asistente opcional con Meta-Llama-3.1-8B-Instruct, limitado al resumen estructurado de resultados.

## Tecnologías

Python, Streamlit, Plotly, Librosa, NumPy, SciPy, Pandas, SoundFile y Hugging Face Inference API.

El archivo `runtime.txt` fija Python 3.12 para el despliegue en Streamlit Community Cloud, compatible con las dependencias científicas del proyecto.

## Instalación y ejecución local

```bash
git clone <tu-repositorio>
cd AudioScope3D
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

La aplicación estará disponible normalmente en `http://localhost:8501`.

## Uso

1. Carga el audio principal desde la barra lateral.
2. Ajusta tamaño FFT, frecuencia máxima, ventana y mapa de color.
3. Explora las pestañas de resumen, espectro y características.
4. En **Comparar**, carga un segundo audio para obtener diferencias y radar.
5. Descarga resultados en **Exportar**. Para imágenes, usa el icono de cámara de cada gráfica Plotly.

Los audios de demostración no se incluyen. Copia tus archivos en `assets/`; consulta [assets/README.md](assets/README.md) para los nombres sugeridos y comentarios de reemplazo.

## Estructura

```text
AudioScope3D/
├── app.py
├── requirements.txt
├── .gitignore
├── assets/                 # Audios locales opcionales (ignorados por Git)
├── data/
└── modules/
    ├── audio_loader.py     # Decodificación y metadatos
    ├── fft_analysis.py     # FFT original refactorizada
    ├── spectrogram.py      # STFT/espectrogramas
    ├── audio_features.py   # Métricas DSP
    ├── visualization.py    # Figuras Plotly
    ├── comparison.py
    ├── export.py
    └── ai_summary.py       # Solo interpretación de resultados
```

## Hugging Face Secrets

La función de IA es opcional. El token nunca se escribe en el código ni debe subirse a Git.

### Desarrollo local

Crea `.streamlit/secrets.toml` (está ignorado por Git) con:

```toml
HUGGINGFACE_TOKEN="colocar_aqui_tu_token"
```

### Repositorio GitHub

No subas `secrets.toml`, `.env` ni el token. El `.gitignore` del proyecto ya excluye esos secretos. Comparte solo el código y configura el secreto por entorno.

### Streamlit Community Cloud

En la configuración de la aplicación, abre **Secrets** y agrega:

```toml
HUGGINGFACE_TOKEN="colocar_aqui_tu_token"
```

Después guarda y reinicia la aplicación. No es necesario modificar el código para desplegarla.

## Despliegue en Streamlit Community Cloud

1. Sube este proyecto a GitHub, sin audios privados ni secretos.
2. En [Streamlit Community Cloud](https://share.streamlit.io/), crea una aplicación desde el repositorio.
3. Selecciona `app.py` como archivo principal.
4. Añade `HUGGINGFACE_TOKEN` en **Secrets** solo si deseas habilitar la pestaña IA.

## Capturas

_Agrega aquí una captura del panel principal._

_Agrega aquí una captura del espectrograma 3D._

## Notas de diseño

El prototipo original reproducía audio mediante `sounddevice` y actualizaba una animación Matplotlib por bloques. En una app web, el navegador reproduce el archivo y Plotly aporta zoom, paneo, rotación y exportación de imagen; así se conserva la lógica FFT/DSP, eliminando dependencias de audio de escritorio que no funcionan en Streamlit Cloud. Para análisis de archivos extensos, el espectrograma 3D se reduce antes de dibujarse para mantener la interfaz fluida.
