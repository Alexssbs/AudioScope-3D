from __future__ import annotations

import json

MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def build_context(audio_name: str, features: dict, comparison: dict | None = None) -> str:
    payload = {"audio": audio_name, "analysis": features}
    if comparison:
        payload["comparison"] = comparison
    return json.dumps(payload, ensure_ascii=False, indent=2)


def ask_interpreter(token: str, context: str, question: str) -> str:
    """
    Call Llama 3.1 usando chat_completion (tarea conversational)
    Basado en el código que SÍ funciona
    """
    from huggingface_hub import InferenceClient

    # Configurar el sistema y usuario
    system_prompt = (
        "Eres un asistente de análisis de audio. Responde en español usando SOLO el resumen "
        "estructurado provisto. No inventes mediciones ni afirmes diagnóstico médico."
    )
    
    user_prompt = f"RESUMEN:\n{context}\n\nPREGUNTA: {question}"
    
    # Crear el cliente (igual que en el código que funciona)
    client = InferenceClient(
        model=MODEL,
        token=token
    )
    
    # Mensajes en el formato correcto
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        # Llamada exactamente igual al código que funciona
        respuesta = client.chat_completion(
            messages=messages,
            max_tokens=350,
            temperature=0.3,  # Más bajo para respuestas más precisas
            top_p=0.95
        )
        
        # Extraer la respuesta
        return respuesta.choices[0].message.content
        
    except Exception as e:
        error_msg = str(e)
        
        # Manejar errores comunes
        if "cold start" in error_msg.lower():
            return "⏳ El modelo está iniciando. Espera 30 segundos y vuelve a intentar."
        elif "credits" in error_msg.lower():
            return "💳 Se han agotado los créditos gratuitos. Espera al próximo mes o usa un modelo gratuito."
        elif "token" in error_msg.lower():
            return "🔑 Token inválido. Verifica tu token en Hugging Face."
        else:
            return f"❌ Error: {error_msg}"