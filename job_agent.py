import os
import sys
import logging
import time
import json
import requests
from google import genai
from google.genai import types

# Configuración de logging estructurado
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] JobAgent: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("JobAgent")

# Validación Fail Fast de entorno
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")

if not GEMINI_API_KEY or not N8N_WEBHOOK_URL:
    logger.error("Faltan variables de entorno requeridas (GEMINI_API_KEY o N8N_WEBHOOK_URL).")
    sys.exit(1)

# Perfil técnico de búsqueda
DEVELOPER_PROFILE = """
Candidato: Desarrollador Python Backend con experiencia en FastAPI, Docker, SQL databases, Node.js y automatización de flujos.
Buscamos ofertas alineadas con: Python, Backend Developer, FastAPI, APIs, Docker, bases de datos o ingeniería de software.
"""

def fetch_remote_jobs() -> list[dict]:
    """Consulta ofertas de empleo públicas desde Remotive API."""
    logger.info("Consultando ofertas de empleo recientes...")
    url = "https://remotive.com/api/remote-jobs?category=software-dev&limit=10"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        jobs = data.get("jobs", [])
        logger.info(f"Se obtuvieron {len(jobs)} ofertas brutas exitosamente.")
        return jobs
    except Exception as e:
        logger.error(f"Error consultando la API de empleos: {e}")
        return []

def evaluate_and_extract_job(job_raw: dict, ai_client: genai.Client) -> dict | None:
    """Utiliza Gemini para analizar la oferta con reintentos automáticos ante alta demanda."""
    title = job_raw.get('title', 'Sin título')
    company = job_raw.get('company_name', 'Desconocida')
    
    prompt = f"""
    Eres un reclutador técnico especializado en desarrollo de software.
    Analiza la siguiente oferta de empleo y compárala con el perfil del candidato.

    {DEVELOPER_PROFILE}

    Oferta de empleo a evaluar:
    - Título: {title}
    - Empresa: {company}
    - Descripción: {job_raw.get('description', '')}
    - Enlace/Contacto: {job_raw.get('url', '')}

    INSTRUCCIONES DE SALIDA:
    Responde ÚNICAMENTE con un objeto JSON válido (sin bloques de código markdown ni texto adicional) con la siguiente estructura:
    {{
        "is_fit": true o false (solo true si encaja fuertemente con Python/Backend/FastAPI/Docker),
        "empresa": "{company}",
        "cargo": "{title}",
        "correo": "Correo de contacto o recruiter si existe en el texto, de lo contrario 'talento@empresa.com'",
        "razon": "Breve explicación técnica de por qué encaja o no"
    }}
    """

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = ai_client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            return json.loads(response.text)
        except Exception as e:
            logger.warning(f"Intento {attempt + 1}/{max_retries} fallido para '{title}': {e}")
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))  # Espera progresiva (2s, 4s...)
            else:
                logger.error(f"Error definitivo procesando la oferta '{title}' con Gemini.")
                return None

def send_to_n8n(job_data: dict) -> None:
    """Envía la oferta filtrada al Webhook de n8n."""
    try:
        response = requests.post(N8N_WEBHOOK_URL, json=job_data, timeout=10)
        if response.status_code in [200, 201]:
            logger.info(f"Postulación enviada exitosamente a n8n para: {job_data.get('cargo')}")
        else:
            logger.warning(f"n8n respondió con código {response.status_code}: {response.text}")
    except Exception as e:
        logger.error(f"Error enviando datos a n8n: {e}")

def main():
    logger.info("🤖 Iniciando Agente Autónomo de Búsqueda de Empleo...")
    
    # Inicializar cliente de Google GenAI
    ai_client = genai.Client(api_key=GEMINI_API_KEY)
    
    jobs = fetch_remote_jobs()
    if not jobs:
        logger.warning("No se encontraron ofertas para procesar.")
        return

    sent_count = 0
    for job in jobs:
        analysis = evaluate_and_extract_job(job, ai_client)
        if analysis and analysis.get("is_fit") is True:
            logger.info(f"✨ ¡Oferta compatible encontrada!: {analysis.get('cargo')} en {analysis.get('empresa')}")
            # Añadir la URL original de la oferta al JSON resultante
            analysis["url"] = job.get("url", "")
            send_to_n8n(analysis)
            sent_count += 1
        # Pausa corta entre iteraciones para cuidar la cuota de la API
        time.sleep(1)

    logger.info(f"🎯 Proceso finalizado. Postulaciones enviadas en esta ejecución: {sent_count}")

if __name__ == "__main__":
    main()