import os
import sys
import logging
import time
import json
import requests
from google import genai
from google.genai import types
from groq import Groq  # Importamos el SDK oficial

# Configuración de logging estructurado
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] JobAgent: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("JobAgent")

# Archivo local para almacenar la memoria de correos ya contactados
HISTORY_FILE = "historial_enviados.json"

# Palabras clave para pre-filtrado local (Ahorra llamadas)
TARGET_KEYWORDS = ["python", "backend", "fastapi", "django", "flask", "developer", "engineer", "software", "api", "docker"]

def load_history() -> list:
    """Carga el historial de correos o empresas ya contactadas para evitar duplicados."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"No se pudo leer el archivo de historial: {e}")
    return []

def save_to_history(company: str, email: str) -> None:
    """Guarda una empresa y correo en el historial para proteger contra reenvíos."""
    history = load_history()
    history.append({"company": company, "email": email, "date": time.strftime("%Y-%m-%d %H:%M:%S")})
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.error(f"Error guardando en el historial: {e}")

# Validación Fail Fast de entorno
# Validación Fail Fast de entorno actualizada
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")

if not GEMINI_API_KEY or not GROQ_API_KEY or not N8N_WEBHOOK_URL:
    logger.error("Faltan variables de entorno requeridas (GEMINI_API_KEY, GROQ_API_KEY o N8N_WEBHOOK_URL).")
    sys.exit(1)

# Perfil técnico de búsqueda
DEVELOPER_PROFILE = """
Candidato: Desarrollador Python Backend con experiencia en FastAPI, Docker, SQL databases, Node.js y automatización de flujos.
Buscamos ofertas alineadas con: Python, Backend Developer, FastAPI, APIs, Docker, bases de datos o ingeniería de software.
"""

def is_relevant_title(title: str) -> bool:
    """Valida localmente si el título contiene palabras clave de software antes de consultar la IA."""
    title_lower = title.lower()
    return any(keyword in title_lower for keyword in TARGET_KEYWORDS)

def fetch_remote_jobs() -> list[dict]:
    """Consulta ofertas de empleo públicas desde Remotive API (Límite optimizado a 5)."""
    logger.info("Consultando ofertas de empleo recientes...")
    url = "https://remotive.com/api/remote-jobs?category=software-dev&limit=5"
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


def evaluate_with_groq(prompt: str) -> dict | None:
    """Proveedor Secundario (Fallback): Ejecuta el prompt en Groq usando el modelo oficial detectado."""
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY no configurada. No se puede ejecutar el fallback.")
        return None
    
    logger.info("🔄 Activando Fallback Multi-Proveedor: Consultando Groq con modelo oficial...")
    
    try:
        # Inicializa usando la variable de entorno o la llave cargada
        groq_client = Groq(api_key=GROQ_API_KEY)
        
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",  # El modelo exacto que encontraste en la consola de Groq
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"},  # Clave para que devuelva JSON puro
            temperature=0.2,
            max_completion_tokens=2048,
            top_p=1,
            stream=False  # Lo ponemos en False porque tu agente necesita el JSON completo de un solo golpe, no por partes (streaming)
        )
        
        content = completion.choices[0].message.content
        return json.loads(content)
        
    except Exception as e:
        logger.error(f"Error en el Fallback con Groq (SDK): {e}")
        return None

def evaluate_and_extract_job(job_raw: dict, ai_client: genai.Client) -> dict | None:
    """Utiliza Gemini. Si falla por cuota (429) o alta demanda (503), conmuta automáticamente a Groq."""
    title = job_raw.get('title', 'Sin título')
    company = job_raw.get('company_name', 'Desconocida')
    
    prompt = f"""
    Eres un asistente de carrera y reclutador técnico experto.
    Analiza la siguiente oferta de empleo y compárala con el perfil del candidato.

    {DEVELOPER_PROFILE}

    Oferta de empleo a evaluar:
    - Título: {title}
    - Empresa: {company}
    - Descripción: {job_raw.get('description', '')}
    - Enlace/Contacto: {job_raw.get('url', '')}

    INSTRUCCIONES DE SALIDA:
    Responde ÚNICAMENTE con un objeto JSON válido (sin bloques de código markdown ni texto adicional) con la siguiente estructura exacta:
    {{
        "is_fit": true o false (solo true si encaja fuertemente con Python/Backend/FastAPI/Docker),
        "empresa": "{company}",
        "cargo": "{title}",
        "correo": "Correo de contacto o de recursos humanos encontrado en la descripción, o 'talento@empresa.com' si no se menciona explícitamente",
        "razon": "Breve explicación técnica de por qué encaja",
        "borrador_correo": "Un mensaje formal y profesional redactado en primera persona para enviar a la empresa, destacando la experiencia del candidato en Python, FastAPI y Docker para este cargo específico."
    }}
    """

    # 1. Intentar con Proveedor Principal (Gemini)
    try:
        response = ai_client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return json.loads(response.text)
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "503" in err_msg:
            logger.warning(f"⚠️ Gemini fuera de servicio/cuota ({err_msg[:40]}...). Conmutando a Groq...")
            # 2. Conmutación por error a Groq
            return evaluate_with_groq(prompt)
        else:
            logger.error(f"Error inesperado en Gemini procesando '{title}': {e}")
            return None

def send_to_n8n(job_data: dict) -> None:
    """Envía la oferta filtrada, el correo extraído y el borrador al Webhook de n8n."""
    try:
        response = requests.post(N8N_WEBHOOK_URL, json=job_data, timeout=10)
        if response.status_code in [200, 201]:
            logger.info(f"Asistente: Datos y borrador enviados exitosamente a n8n para: {job_data.get('cargo')}")
        else:
            logger.warning(f"n8n respondió con código {response.status_code}: {response.text}")
    except Exception as e:
        logger.error(f"Error enviando datos a n8n: {e}")

def main():
    logger.info("🤖 Iniciando Agente Asistente con Resiliencia Multi-Proveedor...")
    
    ai_client = genai.Client(api_key=GEMINI_API_KEY)
    history = load_history()
    contacted_emails = [h["email"] for h in history]
    contacted_companies = [h["company"].lower() for h in history]

    jobs = fetch_remote_jobs()
    if not jobs:
        logger.warning("No se encontraron ofertas para procesar.")
        return

    sent_count = 0
    for job in jobs:
        title = job.get('title', 'Sin título')
        company_name = job.get('company_name', 'Desconocida')

        # Filtro local por palabras clave
        if not is_relevant_title(title):
            logger.info(f"Omitiendo '{title}': No coincide con palabras clave técnicas backend.")
            continue

        # Validación de duplicados en el historial local
        if company_name.lower() in contacted_companies:
            logger.info(f"Omitiendo '{company_name}': Ya existe en el historial de contactos previos.")
            continue

        # Evaluación con IA (Gemini con fallback a Groq)
        analysis = evaluate_and_extract_job(job, ai_client)
        if analysis and analysis.get("is_fit") is True:
            target_email = analysis.get("correo", "")
            
            if target_email in contacted_emails:
                logger.info(f"Omitiendo correo {target_email}: Ya fue contactado anteriormente.")
                continue

            logger.info(f"✨ ¡Oportunidad compatible y nueva encontrada!: {analysis.get('cargo')} en {company_name}")
            analysis["url"] = job.get("url", "")
            
            send_to_n8n(analysis)
            save_to_history(company_name, target_email)
            contacted_emails.append(target_email)
            contacted_companies.append(company_name.lower())
            
            sent_count += 1
            
        time.sleep(2)

    logger.info(f"🎯 Proceso finalizado. Nuevas oportunidades procesadas y enviadas: {sent_count}")

if __name__ == "__main__":
    main()