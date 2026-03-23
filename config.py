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

# ─── GROQ (CLOUD AI) ─────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "groq"

# Groq models
GROQ_MODELS = {
    "fast":     "llama-3.1-8b-instant",
    "balanced": "llama-3.3-70b-versatile",
    "accurate": "openai/gpt-oss-120b",
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
MAX_CHUNKS_PER_QUERY = 3  # how many chunks to retrieve per question (reduce model latency)

# ─── WEB CRAWLER ─────────────────────────────────────────────
MAX_CRAWL_PAGES = 20       # max pages to crawl per domain
CRAWL_DELAY = 1.0          # seconds between page requests (be respectful)
CRAWL_TIMEOUT = 10         # seconds before giving up on a page

# ─── RAG / ANSWER QUALITY ────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.3   # below this = low confidence, flag for review (lowered from 0.4)
MIN_CONTEXT_LENGTH = 50      # minimum chars of context before answering (lowered from 100)

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
Your job is to answer questions from students, staff, and visitors about university topics.

Rules:
1. Use the context provided below as your primary source of information.
2. If specific details are in the context, use them accurately.
3. For questions about fees, costs, or specific procedures not in the context, provide general guidance and direct users to contact the university for current information.
4. For questions about campus visits, facilities, or general inquiries, provide helpful information based on typical university practices.
5. Be conversational and friendly - don't just give short answers.
6. If asked about non-university topics, politely redirect to university-related questions.
7. Never make up specific facts, dates, or contact details that aren't in the context.

Context:
{context}

Question: {question}

Answer:"""


def get_model(speed: str = "balanced") -> str:
    """Return the model name for a given speed preference."""
    if LLM_PROVIDER.lower() == "groq":
        return GROQ_MODELS.get(speed, "mixtral-8x7b-32768")
    else:
        return AVAILABLE_MODELS.get(speed, DEFAULT_MODEL)


def validate_config() -> dict:
    """Check all required config values are present. Returns status dict."""
    issues = []

    if not DB_PASSWORD:
        issues.append("DB_PASSWORD is not set in .env")

    if LLM_PROVIDER.lower() == "groq":
        if not GROQ_API_KEY:
            issues.append("GROQ_API_KEY is not set in .env")
    else:
        if not OLLAMA_URL:
            issues.append("OLLAMA_URL is not set in .env")

    provider_info = {
        "llm_provider": LLM_PROVIDER,
        "database_url": f"postgresql://{DB_USER}:***@{DB_HOST}:{DB_PORT}/{DB_NAME}",
        "chroma_path": CHROMA_PATH,
        "upload_path": UPLOAD_PATH,
    }

    if LLM_PROVIDER.lower() == "groq":
        provider_info["groq_models"] = list(GROQ_MODELS.values())
    else:
        provider_info["ollama_url"] = OLLAMA_URL
        provider_info["default_model"] = DEFAULT_MODEL

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        **provider_info,
    }
