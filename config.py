"""
config.py - Central configuration for University AI Chatbot
All settings loaded from .env file
"""

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ─── DATABASE ───────────────────────────────────────────────
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME", "university_chatbot")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# Full asyncpg DSN
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ─── OLLAMA (LOCAL AI) ───────────────────────────────────────
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "mistral:7b-instruct-q4_K_M")

# Available models — fast / balanced / accurate
AVAILABLE_MODELS = {
    "fast":     "phi3:mini",
    "balanced": "mistral:7b-instruct-q4_K_M",
    "accurate": "llama3.1:latest",
}

# ─── CHROMADB (VECTOR DATABASE) ─────────────────────────────
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
CHROMA_COLLECTION = "university_knowledge"

# ─── EMBEDDINGS ──────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ─── DOCUMENT PROCESSING ─────────────────────────────────────
UPLOAD_PATH = os.getenv("UPLOAD_PATH", "./uploads")
CHUNK_SIZE = 512          # characters per chunk
CHUNK_OVERLAP = 50        # overlap between chunks
MAX_CHUNKS_PER_QUERY = 5  # how many chunks to retrieve per question

# ─── WEB CRAWLER ─────────────────────────────────────────────
MAX_CRAWL_PAGES = 20       # max pages to crawl per domain
CRAWL_DELAY = 1.0          # seconds between page requests (be respectful)
CRAWL_TIMEOUT = 10         # seconds before giving up on a page

# ─── RAG / ANSWER QUALITY ────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.4   # below this = low confidence, flag for review
MIN_CONTEXT_LENGTH = 100     # minimum chars of context before answering

# ─── API ─────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", 8000))
API_TITLE = "University AI Chatbot API"
API_VERSION = "1.3.0"
API_DESCRIPTION = "AI-powered university assistant with RAG, ChromaDB and PostgreSQL"

# ─── CORS (allowed origins for widget embedding) ─────────────
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# ─── PROMPT TEMPLATE ─────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful and friendly university assistant.
Your job is to answer questions from students, staff, and visitors accurately.

Rules:
1. Answer using ONLY the context provided below.
2. If the answer is not in the context, say: "I don't have that information. Please contact the university directly."
3. Be concise but complete.
4. Never make up facts, dates, fees, or contact details.
5. If asked something outside university topics, politely redirect.

Context:
{context}

Question: {question}

Answer:"""


def get_model(speed: str = "balanced") -> str:
    """Return the model name for a given speed preference."""
    return AVAILABLE_MODELS.get(speed, DEFAULT_MODEL)


def validate_config() -> dict:
    """Check all required config values are present. Returns status dict."""
    issues = []

    if not DB_PASSWORD:
        issues.append("DB_PASSWORD is not set in .env")
    if not OLLAMA_URL:
        issues.append("OLLAMA_URL is not set in .env")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "database_url": f"postgresql://{DB_USER}:***@{DB_HOST}:{DB_PORT}/{DB_NAME}",
        "ollama_url": OLLAMA_URL,
        "default_model": DEFAULT_MODEL,
        "chroma_path": CHROMA_PATH,
        "upload_path": UPLOAD_PATH,
    }
