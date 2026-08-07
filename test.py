"""
Prueba específica para Llama 3.1 con chat_completion
Basado en el código que SÍ funciona
"""

import os
import streamlit as st

def test_llama_chat():
    """Prueba Llama 3.1 usando chat_completion - VERSIÓN CORREGIDA"""
    
    # Obtener token
    token = st.secrets.get("HUGGINGFACE_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    
    if not token:
        print("❌ ERROR: No se encontró el token")
        print("   Configura en .streamlit/secrets.toml")
        return False
    
    print(f"✅ Token encontrado: {token[:8]}...{token[-4:]}")
    
    # Probar el modelo - EXACTAMENTE como en el código que funciona
    from huggingface_hub import InferenceClient
    
    model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    client = InferenceClient(
        model=model,
        token=token
    )
    
    print(f"\n🔍 Probando modelo: {model}")
    print("   Usando chat_completion (sin do_sample)...\n")
    
    try:
        # Mensajes de prueba
        messages = [
            {"role": "system", "content": "Eres un asistente útil. Responde en español de manera breve."},
            {"role": "user", "content": "¿Cuál es la capital de México?"}
        ]
        
        # Llamada EXACTAMENTE igual al código que funciona
        response = client.chat_completion(
            messages=messages,
            max_tokens=100,
            temperature=0.3,
            top_p=0.95
        )
        
        # Obtener respuesta
        respuesta = response.choices[0].message.content
        print(f"✅ ÉXITO: {respuesta}")
        print(f"\n📝 Respuesta completa: {respuesta}")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        print("\n📌 Posibles causas:")
        print("- Token incorrecto o no generado en Hugging Face")
        print("- Límite de créditos gratuitos alcanzado (tienes $0.10 al mes)")
        print("- El modelo puede estar en 'cold start' (espera 30 segundos y reintenta)")
        return False


def test_with_audio_context():
    """Prueba con un contexto de audio real - VERSIÓN CORREGIDA"""
    
    token = st.secrets.get("HUGGINGFACE_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    
    if not token:
        print("❌ ERROR: No se encontró el token")
        return False
    
    from huggingface_hub import InferenceClient
    
    model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    client = InferenceClient(
        model=model,
        token=token
    )
    
    # Contexto de audio simulado
    context = {
        "audio": "test.mp3",
        "analysis": {
            "rms": 0.45,
            "energy": 1250.5,
            "spectral_centroid_hz": 3450.0,
            "dominant_frequency_hz": 1200.0,
            "zero_crossing_rate": 0.023,
            "silence_seconds": 1.5,
            "silence_ratio": 0.15,
            "peak_amplitude": 0.98
        }
    }
    
    import json
    context_str = json.dumps(context, ensure_ascii=False, indent=2)
    
    system_prompt = (
        "Eres un asistente de análisis de audio. Responde en español usando SOLO el resumen "
        "estructurado provisto. No inventes mediciones ni afirmes diagnóstico médico."
    )
    
    user_prompt = f"RESUMEN:\n{context_str}\n\nPREGUNTA: ¿Qué características destacas de este audio?"
    
    print("\n" + "="*60)
    print("PRUEBA CON CONTEXTO DE AUDIO")
    print("="*60)
    print(f"\n📊 Contexto:\n{context_str[:200]}...\n")
    
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = client.chat_completion(
            messages=messages,
            max_tokens=350,
            temperature=0.3,
            top_p=0.95
        )
        
        respuesta = response.choices[0].message.content
        print(f"🤖 RESPUESTA:\n{respuesta}")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False


def test_using_ai_summary():
    """Prueba usando directamente la función de ai_summary"""
    
    print("\n" + "="*60)
    print("PRUEBA USANDO AI_SUMMARY.PY")
    print("="*60)
    
    try:
        from modules.ai_summary import ask_interpreter, build_context
        
        token = st.secrets.get("HUGGINGFACE_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        
        if not token:
            print("❌ Token no encontrado")
            return False
        
        # Crear contexto de prueba
        features = {
            "rms": 0.45,
            "energy": 1250.5,
            "spectral_centroid_hz": 3450.0,
            "dominant_frequency_hz": 1200.0,
            "zero_crossing_rate": 0.023,
            "silence_seconds": 1.5,
            "silence_ratio": 0.15,
            "peak_amplitude": 0.98
        }
        
        context = build_context("test.mp3", features)
        question = "¿Qué características destacas de este audio?"
        
        print("📤 Enviando pregunta...")
        respuesta = ask_interpreter(token, context, question)
        
        print(f"\n🤖 RESPUESTA:\n{respuesta}")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False


if __name__ == "__main__":
    print("="*60)
    print("🧪 PRUEBA DE LLAMA 3.1 CON CHAT_COMPLETION (CORREGIDO)")
    print("="*60)
    
    # Prueba 1: Pregunta simple
    print("\n📌 Prueba 1: Pregunta simple")
    print("-"*40)
    test_llama_chat()
    
    # Prueba 2: Con contexto de audio
    print("\n\n📌 Prueba 2: Con contexto de audio")
    print("-"*40)
    test_with_audio_context()
    
    # Prueba 3: Usando ai_summary directamente
    print("\n\n📌 Prueba 3: Usando ai_summary directamente")
    print("-"*40)
    test_using_ai_summary()
    
    print("\n" + "="*60)
    print("✅ PRUEBAS COMPLETADAS")
    print("="*60)