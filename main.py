"""
main.py - FastAPI application entry point
University AI Chatbot v1.3
"""

import os
import logging
import asyncio
from uuid import uuid4
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from config import (
    API_TITLE, API_VERSION, API_DESCRIPTION,
    ALLOWED_ORIGINS, UPLOAD_PATH, API_HOST, API_PORT,
    DATABASE_URL, validate_config, KEEP_UPLOADED_FILES,
)
from models import (
    QuestionRequest, QuestionResponse,
    FeedbackRequest, FeedbackResponse,
    UploadResponse, TaskStatusResponse, DocumentStatus,
    CrawlRequest, CrawlResponse,
    StatsResponse, HealthResponse, ServiceStatus,
    KnowledgeListResponse, KnowledgeItem,
    ConversationHistoryResponse, ConversationItem,
)
import postgres_db as db
import learning_engine as ai
from document_loader import extract_text, get_file_metadata, is_supported
from website_crawler import crawl_website

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── STARTUP / SHUTDOWN ──────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info(f"  {API_TITLE} v{API_VERSION} starting...")
    logger.info("=" * 60)

    config_status = validate_config()
    if not config_status["valid"]:
        for issue in config_status["issues"]:
            logger.warning(f"Config issue: {issue}")

    Path(UPLOAD_PATH).mkdir(parents=True, exist_ok=True)

    try:
        await db.create_tables()
        logger.info("✓ PostgreSQL ready")
    except Exception as e:
        logger.error(f"✗ PostgreSQL failed: {e}")

    ollama_status = await ai.check_ollama_health()
    if ollama_status["status"] == "ok":
        logger.info(f"✓ Ollama ready — models: {ollama_status['models']}")
        # Warm up the default model to avoid slow first request
        try:
            logger.info("Warming up AI model...")
            await ai.query_ollama("Hello", "phi3:mini", timeout=30)
            logger.info("✓ Model warmed up")
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")
    else:
        logger.warning("✗ Ollama not reachable — start with: ollama serve")

    try:
        stats = ai.get_knowledge_base_stats()
        logger.info(f"✓ ChromaDB ready — {stats['total_chunks']} chunks indexed")
    except Exception as e:
        logger.error(f"✗ ChromaDB failed: {e}")

    logger.info(f"  Chat Widget: http://{API_HOST}:{API_PORT}/chat-widget")
    logger.info(f"  Admin Panel: http://{API_HOST}:{API_PORT}/admin")
    logger.info(f"  API Docs:    http://{API_HOST}:{API_PORT}/docs")
    logger.info("=" * 60)

    yield

    await db.close_pool()
    logger.info("University AI Chatbot shut down gracefully")


# ─── APP ─────────────────────────────────────────────────────

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


# ─── BACKGROUND DB HELPERS ───────────────────────────────────
# Background threads MUST use fresh DB connections (not the main pool)

async def _bg_save_document(filename, chunks, file_size_kb, task_id):
    import asyncpg
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        await conn.execute("""
            INSERT INTO knowledge_base (source_name, source_type, chunks_count, file_size_kb)
            VALUES ($1, 'document', $2, $3)
        """, filename, chunks, file_size_kb)
        await conn.execute("""
            UPDATE processing_tasks
            SET status = 'complete', chunks_created = $2, completed_at = NOW()
            WHERE id = $1::uuid
        """, task_id, chunks)
    finally:
        await conn.close()


async def _bg_save_website(source_name, chunks, url, task_id):
    import asyncpg
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        await conn.execute("""
            INSERT INTO knowledge_base (source_name, source_type, chunks_count, url)
            VALUES ($1, 'website', $2, $3)
        """, source_name, chunks, url)
        await conn.execute("""
            UPDATE processing_tasks
            SET status = 'complete', chunks_created = $2, completed_at = NOW()
            WHERE id = $1::uuid
        """, task_id, chunks)
    finally:
        await conn.close()


async def _bg_fail_task(task_id, error_message):
    import asyncpg
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        await conn.execute("""
            UPDATE processing_tasks
            SET status = 'failed', error_message = $2, completed_at = NOW()
            WHERE id = $1::uuid
        """, task_id, error_message)
    finally:
        await conn.close()


# ─── BACKGROUND TASKS ────────────────────────────────────────

def _process_document(filepath: str, filename: str, task_id: str):
    """Runs in background thread with its own event loop."""
    import traceback
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        logger.info(f"⏳ Processing: {filename}")

        text = extract_text(filepath)
        if not text:
            raise ValueError("No text could be extracted")
        logger.info(f"   Extracted {len(text)} chars")

        chunks = ai.add_to_knowledge_base(text, source_name=filename)
        logger.info(f"   Created {chunks} chunks in ChromaDB")

        meta = get_file_metadata(filepath)
        loop.run_until_complete(
            _bg_save_document(filename, chunks, meta["size_kb"], task_id)
        )
        logger.info(f"✓ Done: {filename} → {chunks} chunks")

    except Exception as e:
        logger.error(f"✗ Failed: {filename} — {e}")
        logger.error(traceback.format_exc())
        try:
            loop.run_until_complete(_bg_fail_task(task_id, str(e)))
        except Exception:
            pass
    finally:
        loop.close()
        if not KEEP_UPLOADED_FILES:
            try:
                os.remove(filepath)
            except Exception:
                pass
        else:
            logger.info(f"📁 Kept file: {filepath}")


def _process_crawl(start_url: str, max_pages: int, task_id: str):
    """Runs in background thread with its own event loop."""
    import traceback
    from urllib.parse import urlparse
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        logger.info(f"⏳ Crawling: {start_url}")

        result = loop.run_until_complete(
            crawl_website(start_url, max_pages=max_pages)
        )
        if result.get("error"):
            raise ValueError(result["error"])
        if not result["combined_text"]:
            raise ValueError("No content extracted from website")

        logger.info(f"   Crawled {result['pages_crawled']} pages")

        domain = urlparse(start_url).netloc
        source_name = f"web:{domain}"
        chunks = ai.add_to_knowledge_base(result["combined_text"], source_name=source_name)
        logger.info(f"   Created {chunks} chunks in ChromaDB")

        loop.run_until_complete(
            _bg_save_website(source_name, chunks, start_url, task_id)
        )
        logger.info(f"✓ Crawl done: {result['pages_crawled']} pages → {chunks} chunks")

    except Exception as e:
        logger.error(f"✗ Crawl failed: {start_url} — {e}")
        logger.error(traceback.format_exc())
        try:
            loop.run_until_complete(_bg_fail_task(task_id, str(e)))
        except Exception:
            pass
    finally:
        loop.close()


# ─── HEALTH CHECK ────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    services = {}
    try:
        stats = await db.get_stats()
        services["postgresql"] = ServiceStatus(status="ok", message=f"{stats['total_conversations']} conversations")
    except Exception as e:
        services["postgresql"] = ServiceStatus(status="down", message=str(e))

    ollama = await ai.check_ollama_health()
    if ollama["status"] == "ok":
        services["ollama"] = ServiceStatus(status="ok", message=f"Models: {', '.join(ollama['models'])}")
    else:
        services["ollama"] = ServiceStatus(status="down", message=ollama.get("error", "Not reachable"))

    try:
        chroma_stats = ai.get_knowledge_base_stats()
        services["chromadb"] = ServiceStatus(status="ok", message=f"{chroma_stats['total_chunks']} chunks indexed")
    except Exception as e:
        services["chromadb"] = ServiceStatus(status="down", message=str(e))

    overall = "ok" if all(s.status == "ok" for s in services.values()) else "degraded"
    return HealthResponse(status=overall, version=API_VERSION, services=services, timestamp=datetime.utcnow())


# ─── QUESTION ANSWERING ──────────────────────────────────────

@app.post("/ask", response_model=QuestionResponse, tags=["Chat"])
async def ask_question(request: QuestionRequest):
    session_id = request.session_id or f"session_{uuid4().hex[:12]}"
    result = await ai.ask_question(question=request.question, model_speed=request.model_speed.value)
    conversation_id = await db.save_conversation(
        session_id=session_id, question=request.question,
        answer=result["answer"], confidence=result["confidence"],
        model_used=result["model_used"], sources_found=result["sources_found"],
        processing_ms=result["processing_ms"],
    )
    return QuestionResponse(
        answer=result["answer"], confidence=result["confidence"],
        conversation_id=conversation_id, session_id=session_id,
        model_used=result["model_used"], sources_found=result["sources_found"],
        low_confidence=result["low_confidence"], processing_time_ms=result["processing_ms"],
    )


# ─── FEEDBACK ────────────────────────────────────────────────

@app.post("/feedback", response_model=FeedbackResponse, tags=["Chat"])
async def submit_feedback(request: FeedbackRequest):
    success = await db.save_feedback(
        conversation_id=request.conversation_id,
        rating=request.rating, comment=request.comment,
    )
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return FeedbackResponse(success=True, message=f"Thank you! Rating: {request.rating}/5")


# ─── DOCUMENT UPLOAD ─────────────────────────────────────────

@app.post("/upload", response_model=UploadResponse, tags=["Knowledge Base"])
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not is_supported(file.filename):
        raise HTTPException(status_code=400, detail="Unsupported file type. Supported: PDF, DOCX, TXT, MD")

    safe_filename = f"{uuid4().hex}_{file.filename}"
    filepath = os.path.join(UPLOAD_PATH, safe_filename)
    with open(filepath, "wb") as f:
        content = await file.read()
        f.write(content)

    task_id = await db.create_task(filename=file.filename)
    background_tasks.add_task(_process_document, filepath, file.filename, task_id)

    return UploadResponse(
        task_id=task_id, status=DocumentStatus.processing,
        filename=file.filename,
        message=f"'{file.filename}' is being processed. Poll /status/{task_id}",
    )


@app.get("/status/{task_id}", response_model=TaskStatusResponse, tags=["Knowledge Base"])
async def get_task_status(task_id: str):
    task = await db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskStatusResponse(
        task_id=task["id"], status=DocumentStatus(task["status"]),
        filename=task["filename"], chunks_created=task["chunks_created"],
        error=task["error_message"], completed_at=task["completed_at"],
    )


# ─── WEBSITE CRAWLER ─────────────────────────────────────────

@app.post("/crawl", response_model=CrawlResponse, tags=["Knowledge Base"])
async def crawl_website_endpoint(request: CrawlRequest, background_tasks: BackgroundTasks):
    task_id = await db.create_task(filename=request.url)
    background_tasks.add_task(_process_crawl, request.url, request.max_pages, task_id)
    return CrawlResponse(
        task_id=task_id, status=DocumentStatus.processing,
        start_url=request.url,
        message=f"Crawling started for {request.url}. Poll /status/{task_id}",
    )


# ─── KNOWLEDGE BASE ──────────────────────────────────────────

@app.get("/knowledge", response_model=KnowledgeListResponse, tags=["Knowledge Base"])
async def list_knowledge_sources():
    sources = await db.get_knowledge_sources()
    items = [
        KnowledgeItem(
            id=s["id"], source_name=s["source_name"], source_type=s["source_type"],
            chunks_count=s["chunks_count"], created_at=s["created_at"],
            file_size_kb=s.get("file_size_kb"), url=s.get("url"),
        )
        for s in sources
    ]
    return KnowledgeListResponse(items=items, total=len(items))


@app.delete("/knowledge/{source_id}", tags=["Knowledge Base"])
async def delete_knowledge_source(source_id: str):
    sources = await db.get_knowledge_sources()
    source = next((s for s in sources if s["id"] == source_id), None)
    if not source:
        raise HTTPException(status_code=404, detail="Knowledge source not found")
    ai.remove_from_knowledge_base(source["source_name"])
    await db.delete_knowledge_source(source_id)
    return {"success": True, "message": f"Removed '{source['source_name']}'"}


# ─── ANALYTICS ───────────────────────────────────────────────

@app.get("/stats", response_model=StatsResponse, tags=["Analytics"])
async def get_statistics():
    stats = await db.get_stats()
    return StatsResponse(**stats)


@app.get("/conversations/flagged", tags=["Analytics"])
async def get_flagged_conversations(limit: int = Query(20, ge=1, le=100)):
    flagged = await db.get_low_confidence_answers(limit)
    return {"flagged": flagged, "total": len(flagged)}


@app.get("/conversations", response_model=ConversationHistoryResponse, tags=["Chat"])
async def get_recent_conversations(limit: int = Query(20, ge=1, le=100)):
    history = await db.get_recent_conversations(limit)
    items = [
        ConversationItem(
            id=h["id"], session_id=h["session_id"], question=h["question"],
            answer=h["answer"], confidence=h["confidence"], model_used=h["model_used"],
            rating=h.get("rating"), created_at=h["created_at"],
        )
        for h in history
    ]
    return ConversationHistoryResponse(session_id="recent", conversations=items, total=len(items))


@app.get("/conversations/{session_id}", response_model=ConversationHistoryResponse, tags=["Chat"])
async def get_conversation_history(session_id: str):
    history = await db.get_session_history(session_id)
    items = [
        ConversationItem(
            id=h["id"], session_id=h["session_id"], question=h["question"],
            answer=h["answer"], confidence=h["confidence"], model_used=h["model_used"],
            rating=h.get("rating"), created_at=h["created_at"],
        )
        for h in history
    ]
    return ConversationHistoryResponse(session_id=session_id, conversations=items, total=len(items))


# ─── FRONTEND ────────────────────────────────────────────────

@app.get("/chat-widget", response_class=HTMLResponse, tags=["Frontend"])
async def chat_widget():
    widget_path = Path("static/widget.html")
    if widget_path.exists():
        return widget_path.read_text(encoding="utf-8")
    return HTMLResponse("<h2>Widget not found.</h2>")


@app.get("/admin", response_class=HTMLResponse, tags=["Frontend"])
async def admin_dashboard():
    admin_path = Path("static/admin.html")
    if admin_path.exists():
        return admin_path.read_text(encoding="utf-8")
    return HTMLResponse("<h2>Admin panel not found.</h2>")


@app.get("/", tags=["System"])
async def root():
    return {
        "name": API_TITLE, "version": API_VERSION, "status": "running",
        "links": {
            "chat_widget": f"http://{API_HOST}:{API_PORT}/chat-widget",
            "admin":       f"http://{API_HOST}:{API_PORT}/admin",
            "api_docs":    f"http://{API_HOST}:{API_PORT}/docs",
            "health":      f"http://{API_HOST}:{API_PORT}/health",
        }
    }


# ─── RUN ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True, log_level="info")