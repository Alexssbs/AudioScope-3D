# 🎧 AudioScope 3D

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tu-app.streamlit.app)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Aplicación web de análisis de audio con visualizaciones 3D interactivas, métricas DSP y asistente de IA con Llama 3.1.

## 🌐 Demo en Producción

**🔗 URL:** [https://tu-app.streamlit.app](https://audioscope-3d-nxdpbza9heatv2bt5ndqjz.streamlit.app/)

> *Reemplaza la URL anterior con el enlace de tu aplicación desplegada.*

---

## 📋 Tabla de Contenidos

- [Características](#-características)
- [Tecnologías](#-tecnologías)
- [Instalación Local](#-instalación-local)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Uso](#-uso)
- [Configuración de IA](#-configuración-de-ia)
- [Despliegue](#-despliegue)
- [Capturas de Pantalla](#-capturas-de-pantalla)
- [Contribuciones](#-contribuciones)
- [Licencia](#-licencia)

---

## ✨ Características

### 📊 Análisis DSP
- **Forma de onda** interactiva con zoom y paneo
- **Espectro FFT** con visualización en tiempo real
- **Espectrograma 2D** y **3D** con mapa de colores personalizable
- **Métricas avanzadas**: RMS, energía, centroide espectral, frecuencia dominante, ZCR, silencio y pico

### 🔀 Comparación
- Comparación lado a lado de dos archivos de audio
- Tabla de diferencias con métricas clave
- Radar comparativo normalizado

### 🎵 Visualización en Tiempo Real (Simulada)
- Navegación por segmentos del audio
- Forma de onda y espectro actualizados dinámicamente
- Espectrograma 3D con marcador de posición

### 🤖 Asistente de IA
- Interpretación en lenguaje natural de los resultados
- **Llama 3.1 8B** vía Hugging Face Inference API
- **Seguridad**: La IA solo recibe métricas, nunca el audio crudo

### 💾 Exportación
- JSON (estructura completa)
- CSV (tabla plana)
- Excel (.xlsx) con formato profesional

### 🎨 Personalización
- **Temas claro/oscuro** con un solo clic
- Parámetros ajustables: tamaño FFT, ventana, frecuencia máxima
- Mapas de colores para espectrogramas

---

## 🛠️ Tecnologías

| Tecnología | Uso |
|------------|-----|
| **Streamlit** | Framework web e interfaz de usuario |
| **Plotly** | Visualizaciones interactivas (2D y 3D) |
| **Librosa** | Procesamiento de audio y extracción de características |
| **NumPy/SciPy** | Cálculos numéricos y FFT |
| **Pandas** | Manipulación de datos |
| **SoundFile** | Lectura de archivos de audio |
| **Hugging Face Hub** | Inferencia de modelos de IA |
| **OpenPyXL** | Exportación a Excel |

---

## 📦 Instalación Local

### Requisitos previos
- Python 3.12 o superior
- Git

### Pasos

1. **Clonar el repositorio**
```bash
git clone https://github.com/tu-usuario/AudioScope3D.git
cd AudioScope3D
```

2. **Crear y activar entorno virtual**
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Configurar secrets (opcional - para IA)**
Crear el archivo `.streamlit/secrets.toml`:
```toml
HUGGINGFACE_TOKEN = "hf_tu_token_aqui"
```

> **Obtén tu token:** [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

5. **Ejecutar la aplicación**
```bash
streamlit run app.py
```

6. **Abrir en el navegador**
La aplicación estará disponible en `http://localhost:8501`

---

## 📁 Estructura del Proyecto

```
AudioScope3D/
├── app.py                      # Punto de entrada principal
├── requirements.txt            # Dependencias
├── runtime.txt                 # Versión de Python para Cloud
├── README.md                   # Documentación
├── .gitignore                  # Archivos ignorados por Git
│
├── .streamlit/
│   └── secrets.toml           # Secrets (NO subir a Git)
│
├── modules/
│   ├── __init__.py
│   ├── ai_summary.py          # Integración con Hugging Face
│   ├── audio_features.py      # Extracción de métricas DSP
│   ├── audio_loader.py        # Carga y decodificación
│   ├── comparison.py          # Comparación de audios
│   ├── export.py              # Exportación a JSON/CSV/Excel
│   ├── fft_analysis.py        # Cálculo de FFT
│   ├── spectrogram.py         # Generación de espectrogramas
│   └── visualization.py       # Figuras Plotly
│
├── assets/                     # Audios de demostración (ignorados)
│   ├── README.md              # Instrucciones para audios
│   └── .gitkeep
│
└── data/                       # Datos generados (ignorados)
    └── .gitkeep
```

---

## 🚀 Uso

### Carga de audio
1. En la barra lateral, selecciona "Subir archivo" o "Audio de demostración"
2. Si usas demostración, selecciona un archivo de `assets/`
3. O sube tu propio archivo (soporta MP3, WAV, FLAC, OGG, M4A)

### Análisis
- Ajusta los parámetros en la barra lateral (FFT, frecuencia, ventana)
- Los gráficos se actualizan automáticamente

### Pestañas disponibles
- **📋 Resumen**: Información del audio y forma de onda
- **📈 Espectro**: FFT y espectrogramas 2D/3D
- **🎵 Tiempo Real**: Navegación interactiva por segmentos
- **📊 Características**: Métricas DSP detalladas
- **🔀 Comparar**: Compara dos archivos de audio
- **💾 Exportar**: Descarga resultados en múltiples formatos
- **🤖 IA**: Asistente con Llama 3.1

### Atajos de Plotly
- **Zoom**: Rueda del mouse
- **Pan**: Arrastrar
- **Rotación** (3D): Arrastrar con clic
- **Exportar imagen**: Botón de cámara en la barra de herramientas

---

## 🔑 Configuración de IA

### Desarrollo Local
Crear `.streamlit/secrets.toml`:
```toml
HUGGINGFACE_TOKEN = "hf_tu_token_aqui"
```

### Streamlit Cloud
1. Ve a la configuración de tu app
2. En **Secrets**, agrega:
```toml
HUGGINGFACE_TOKEN = "hf_tu_token_aqui"
```

### Modelo usado
- **Meta Llama 3.1 8B** (`meta-llama/Meta-Llama-3.1-8B-Instruct`)
- La IA solo recibe métricas, no el audio crudo

---

## 🌐 Despliegue

### Opción 1: Streamlit Community Cloud (Recomendado)

1. Sube el proyecto a GitHub
2. Ve a [Streamlit Cloud](https://share.streamlit.io/)
3. Conecta tu repositorio
4. Selecciona `app.py` como archivo principal
5. Configura el secret `HUGGINGFACE_TOKEN`
6. Haz clic en **Deploy**

### Opción 2: Render.com

```bash
# Crear archivo render.yaml
services:
  - type: web
    name: AudioScope3D
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app.py --server.port $PORT
    envVars:
      - key: HUGGINGFACE_TOKEN
        sync: false
```

### Opción 3: Docker

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
---

## 🧪 Pruebas

### Probar la conexión con Hugging Face
```bash
python test_llama.py
```

### Probar componentes individuales
```bash
# Probar carga de audio
python -c "from modules.audio_loader import load_audio; print('OK')"

# Probar extracción de características
python -c "from modules.audio_features import extract_features; print('OK')"
```

---

## 📝 Notas de Diseño

- **Seguridad**: El audio nunca se envía a la IA, solo las métricas
- **Rendimiento**: Los datos se reducen automáticamente para visualización
- **Compatibilidad**: Soporta archivos de hasta 200MB
- **Responsive**: Diseño adaptado a diferentes tamaños de pantalla

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## 📄 Licencia

Distribuido bajo la Licencia MIT. Ver `LICENSE` para más información.

---

**⭐ Si te gusta este proyecto, no olvides darle una estrella en GitHub!**

## 🔧 Cómo personalizar el README

### 1. URL de producción
Reemplaza `https://tu-app.streamlit.app` con tu URL real después del despliegue.

### 2. Capturas de pantalla
Reemplaza los placeholders de imágenes con capturas reales:
```markdown
![Panel Principal](ruta/a/tu/imagen.png)
```
