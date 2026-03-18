"""
main.py - FastAPI application entry point
University AI Chatbot v1.3
All routes, middleware, startup/shutdown, and background tasks
"""

import os
import logging
import asyncio
from uuid import uuid4
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from config import (
    API_TITLE, API_VERSION, API_DESCRIPTION,
    ALLOWED_ORIGINS, UPLOAD_PATH, API_HOST, API_PORT,
    validate_config,
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

# ─── LOGGING ─────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── STARTUP / SHUTDOWN ──────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup, clean up on shutdown."""
    logger.info("=" * 60)
    logger.info(f"  {API_TITLE} v{API_VERSION} starting...")
    logger.info("=" * 60)

    # Validate config
    config_status = validate_config()
    if not config_status["valid"]:
        for issue in config_status["issues"]:
            logger.warning(f"Config issue: {issue}")

    # Create upload directory
    Path(UPLOAD_PATH).mkdir(parents=True, exist_ok=True)

    # Initialize database tables
    try:
        await db.create_tables()
        logger.info("✓ PostgreSQL ready")
    except Exception as e:
        logger.error(f"✗ PostgreSQL failed: {e}")

    # Check Ollama
    ollama_status = await ai.check_ollama_health()
    if ollama_status["status"] == "ok":
        logger.info(f"✓ Ollama ready — models: {ollama_status['models']}")
    else:
        logger.warning(f"✗ Ollama not reachable — start with: ollama serve")

    # Check ChromaDB
    try:
        stats = ai.get_knowledge_base_stats()
        logger.info(f"✓ ChromaDB ready — {stats['total_chunks']} chunks indexed")
    except Exception as e:
        logger.error(f"✗ ChromaDB failed: {e}")

    logger.info(f"  Chat Widget: http://{API_HOST}:{API_PORT}/chat-widget")
    logger.info(f"  Admin Panel: http://{API_HOST}:{API_PORT}/admin")
    logger.info(f"  API Docs:    http://{API_HOST}:{API_PORT}/docs")
    logger.info("=" * 60)

    yield  # Application runs here

    # Shutdown
    await db.close_pool()
    logger.info("University AI Chatbot shut down gracefully")


# ─── APP CREATION ─────────────────────────────────────────────

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow widget to be embedded on any university website
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (widget, admin dashboard)
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


# ─── HEALTH CHECK ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check status of all services."""
    services = {}

    # PostgreSQL
    try:
        stats = await db.get_stats()
        services["postgresql"] = ServiceStatus(status="ok", message=f"{stats['total_conversations']} conversations")
    except Exception as e:
        services["postgresql"] = ServiceStatus(status="down", message=str(e))

    # Ollama
    ollama = await ai.check_ollama_health()
    if ollama["status"] == "ok":
        services["ollama"] = ServiceStatus(status="ok", message=f"Models: {', '.join(ollama['models'])}")
    else:
        services["ollama"] = ServiceStatus(status="down", message=ollama.get("error", "Not reachable"))

    # ChromaDB
    try:
        chroma_stats = ai.get_knowledge_base_stats()
        services["chromadb"] = ServiceStatus(status="ok", message=f"{chroma_stats['total_chunks']} chunks indexed")
    except Exception as e:
        services["chromadb"] = ServiceStatus(status="down", message=str(e))

    overall = "ok" if all(s.status == "ok" for s in services.values()) else "degraded"

    return HealthResponse(
        status=overall,
        version=API_VERSION,
        services=services,
        timestamp=datetime.utcnow(),
    )


# ─── QUESTION ANSWERING ──────────────────────────────────────

@app.post("/ask", response_model=QuestionResponse, tags=["Chat"])
async def ask_question(request: QuestionRequest):
    """
    Ask the AI a question. Returns an answer with confidence score.
    If no session_id is provided, one is auto-generated.
    """
    session_id = request.session_id or f"session_{uuid4().hex[:12]}"

    # Get answer from AI
    result = await ai.ask_question(
        question=request.question,
        model_speed=request.model_speed.value,
    )

    # Save to database
    conversation_id = await db.save_conversation(
        session_id=    session_id,
        question=      request.question,
        answer=        result["answer"],
        confidence=    result["confidence"],
        model_used=    result["model_used"],
        sources_found= result["sources_found"],
        processing_ms= result["processing_ms"],
    )

    return QuestionResponse(
        answer=          result["answer"],
        confidence=      result["confidence"],
        conversation_id= conversation_id,
        session_id=      session_id,
        model_used=      result["model_used"],
        sources_found=   result["sources_found"],
        low_confidence=  result["low_confidence"],
        processing_time_ms= result["processing_ms"],
    )


# ─── FEEDBACK ────────────────────────────────────────────────

@app.post("/feedback", response_model=FeedbackResponse, tags=["Chat"])
async def submit_feedback(request: FeedbackRequest):
    """Submit a rating (1-5 stars) for an answer."""
    success = await db.save_feedback(
        conversation_id= request.conversation_id,
        rating=          request.rating,
        comment=         request.comment,
    )

    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return FeedbackResponse(
        success=True,
        message=f"Thank you for your feedback! Rating: {request.rating}/5"
    )


# ─── DOCUMENT UPLOAD ─────────────────────────────────────────

async def _process_document(filepath: str, filename: str, task_id: str):
    """Background task: extract text and add to knowledge base."""
    try:
        logger.info(f"Processing document: {filename}")

        # Extract text
        text = extract_text(filepath)
        if not text:
            raise ValueError("No text could be extracted from the document")

        # Add to ChromaDB
        chunks = ai.add_to_knowledge_base(text, source_name=filename)

        # Get file size
        meta = get_file_metadata(filepath)

        # Save metadata to PostgreSQL
        await db.save_knowledge_source(
            source_name= filename,
            source_type= "document",
            chunks_count= chunks,
            file_size_kb= meta["size_kb"],
        )

        # Update task
        await db.update_task(task_id, "complete", chunks_created=chunks)
        logger.info(f"✓ Document processed: {filename} → {chunks} chunks")

    except Exception as e:
        logger.error(f"✗ Document processing failed for {filename}: {e}")
        await db.update_task(task_id, "failed", error_message=str(e))
    finally:
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except Exception:
            pass


@app.post("/upload", response_model=UploadResponse, tags=["Knowledge Base"])
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a document (PDF, DOCX, TXT) to the knowledge base.
    Processing happens in the background — use /status/{task_id} to track.
    """
    if not is_supported(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: PDF, DOCX, TXT, MD"
        )

    # Save file to uploads directory
    safe_filename = f"{uuid4().hex}_{file.filename}"
    filepath = os.path.join(UPLOAD_PATH, safe_filename)

    with open(filepath, "wb") as f:
        content = await file.read()
        f.write(content)

    # Create task record
    task_id = await db.create_task(filename=file.filename)

    # Process in background (non-blocking)
    background_tasks.add_task(_process_document, filepath, file.filename, task_id)

    return UploadResponse(
        task_id=  task_id,
        status=   DocumentStatus.processing,
        filename= file.filename,
        message=  f"Document '{file.filename}' is being processed. Poll /status/{task_id}",
    )


@app.get("/status/{task_id}", response_model=TaskStatusResponse, tags=["Knowledge Base"])
async def get_task_status(task_id: str):
    """Check the status of a document upload or crawl task."""
    task = await db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return TaskStatusResponse(
        task_id=       task["id"],
        status=        DocumentStatus(task["status"]),
        filename=      task["filename"],
        chunks_created= task["chunks_created"],
        error=         task["error_message"],
        completed_at=  task["completed_at"],
    )


# ─── WEBSITE CRAWLER ─────────────────────────────────────────

async def _process_crawl(start_url: str, max_pages: int, task_id: str):
    """Background task: crawl website and add to knowledge base."""
    try:
        result = await crawl_website(start_url, max_pages=max_pages)

        if result.get("error"):
            raise ValueError(result["error"])

        if not result["combined_text"]:
            raise ValueError("No content could be extracted from the website")

        # Source name = domain
        from urllib.parse import urlparse
        domain = urlparse(start_url).netloc
        source_name = f"web:{domain}"

        # Add to ChromaDB
        chunks = ai.add_to_knowledge_base(result["combined_text"], source_name=source_name)

        # Save metadata
        await db.save_knowledge_source(
            source_name= source_name,
            source_type= "website",
            chunks_count= chunks,
            url=          start_url,
        )

        await db.update_task(task_id, "complete", chunks_created=chunks)
        logger.info(
            f"✓ Crawl complete: {result['pages_crawled']} pages → {chunks} chunks"
        )

    except Exception as e:
        logger.error(f"✗ Crawl failed for {start_url}: {e}")
        await db.update_task(task_id, "failed", error_message=str(e))


@app.post("/crawl", response_model=CrawlResponse, tags=["Knowledge Base"])
async def crawl_website_endpoint(
    request: CrawlRequest,
    background_tasks: BackgroundTasks,
):
    """Crawl a website and add its content to the knowledge base."""
    task_id = await db.create_task(filename=request.url)
    background_tasks.add_task(
        _process_crawl, request.url, request.max_pages, task_id
    )

    return CrawlResponse(
        task_id=    task_id,
        status=     DocumentStatus.processing,
        start_url=  request.url,
        message=    f"Crawling started for {request.url}. Poll /status/{task_id}",
    )


# ─── KNOWLEDGE BASE MANAGEMENT ───────────────────────────────

@app.get("/knowledge", response_model=KnowledgeListResponse, tags=["Knowledge Base"])
async def list_knowledge_sources():
    """List all documents and websites in the knowledge base."""
    sources = await db.get_knowledge_sources()
    items = [
        KnowledgeItem(
            id=          s["id"],
            source_name= s["source_name"],
            source_type= s["source_type"],
            chunks_count= s["chunks_count"],
            created_at=  s["created_at"],
            file_size_kb= s.get("file_size_kb"),
            url=         s.get("url"),
        )
        for s in sources
    ]
    return KnowledgeListResponse(items=items, total=len(items))


@app.delete("/knowledge/{source_id}", tags=["Knowledge Base"])
async def delete_knowledge_source(source_id: str):
    """Remove a document or website from the knowledge base."""
    # Get source info first
    sources = await db.get_knowledge_sources()
    source = next((s for s in sources if s["id"] == source_id), None)

    if not source:
        raise HTTPException(status_code=404, detail="Knowledge source not found")

    # Remove from ChromaDB
    ai.remove_from_knowledge_base(source["source_name"])

    # Remove from PostgreSQL
    await db.delete_knowledge_source(source_id)

    return {"success": True, "message": f"Removed '{source['source_name']}' from knowledge base"}


# ─── ANALYTICS ───────────────────────────────────────────────

@app.get("/stats", response_model=StatsResponse, tags=["Analytics"])
async def get_statistics():
    """Get chatbot analytics and statistics."""
    stats = await db.get_stats()
    return StatsResponse(**stats)


@app.get("/conversations/flagged", tags=["Analytics"])
async def get_flagged_conversations(limit: int = Query(20, ge=1, le=100)):
    """Get low-confidence answers that need human review."""
    flagged = await db.get_low_confidence_answers(limit)
    return {"flagged": flagged, "total": len(flagged)}


@app.get("/conversations/{session_id}", response_model=ConversationHistoryResponse, tags=["Chat"])
async def get_conversation_history(session_id: str):
    """Get chat history for a session."""
    history = await db.get_session_history(session_id)
    items = [
        ConversationItem(
            id=         h["id"],
            session_id= h["session_id"],
            question=   h["question"],
            answer=     h["answer"],
            confidence= h["confidence"],
            model_used= h["model_used"],
            rating=     h.get("rating"),
            created_at= h["created_at"],
        )
        for h in history
    ]
    return ConversationHistoryResponse(
        session_id=     session_id,
        conversations=  items,
        total=          len(items),
    )


# ─── FRONTEND ROUTES ─────────────────────────────────────────

@app.get("/chat-widget", response_class=HTMLResponse, tags=["Frontend"])
async def chat_widget():
    """Serve the embeddable chat widget."""
    widget_path = Path("static/widget.html")
    if widget_path.exists():
        return widget_path.read_text()
    return HTMLResponse("<h2>Widget not found. Make sure static/widget.html exists.</h2>")


@app.get("/admin", response_class=HTMLResponse, tags=["Frontend"])
async def admin_dashboard():
    """Serve the admin dashboard."""
    admin_path = Path("static/admin.html")
    if admin_path.exists():
        return admin_path.read_text()
    return HTMLResponse("<h2>Admin panel not found. Make sure static/admin.html exists.</h2>")


@app.get("/", tags=["System"])
async def root():
    """API root — quick links."""
    return {
        "name":        API_TITLE,
        "version":     API_VERSION,
        "status":      "running",
        "links": {
            "chat_widget":  f"http://{API_HOST}:{API_PORT}/chat-widget",
            "admin":        f"http://{API_HOST}:{API_PORT}/admin",
            "api_docs":     f"http://{API_HOST}:{API_PORT}/docs",
            "health":       f"http://{API_HOST}:{API_PORT}/health",
        }
    }


# ─── RUN ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,          # Auto-reload on code changes (dev mode)
        log_level="info",
    )
