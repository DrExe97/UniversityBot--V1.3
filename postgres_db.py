"""
postgres_db.py - Async PostgreSQL database layer
Handles all conversations, feedback, knowledge metadata and analytics
"""

import asyncpg
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from uuid import uuid4

from config import DATABASE_URL, CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)


# ─── CONNECTION POOL ─────────────────────────────────────────

_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    """Get or create the connection pool."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=2,
            max_size=10,
            command_timeout=30,
        )
        logger.info("PostgreSQL connection pool created")
    return _pool


async def close_pool():
    """Close the connection pool on shutdown."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("PostgreSQL connection pool closed")


# ─── SCHEMA SETUP ────────────────────────────────────────────

async def create_tables():
    """Create all tables if they don't exist."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                session_id    TEXT NOT NULL,
                question      TEXT NOT NULL,
                answer        TEXT NOT NULL,
                confidence    FLOAT NOT NULL DEFAULT 0.0,
                model_used    VARCHAR(50) NOT NULL DEFAULT 'mistral',
                sources_found INTEGER NOT NULL DEFAULT 0,
                low_confidence BOOLEAN NOT NULL DEFAULT FALSE,
                processing_ms INTEGER NOT NULL DEFAULT 0,
                created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                conversation_id   UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                rating            INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                comment           TEXT,
                created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS knowledge_base (
                id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                source_name   TEXT NOT NULL,
                source_type   VARCHAR(20) NOT NULL CHECK (source_type IN ('document', 'website')),
                chunks_count  INTEGER NOT NULL DEFAULT 0,
                file_size_kb  FLOAT,
                url           TEXT,
                created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS processing_tasks (
                id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                filename      TEXT,
                status        VARCHAR(20) NOT NULL DEFAULT 'processing',
                chunks_created INTEGER,
                error_message TEXT,
                created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                completed_at  TIMESTAMPTZ
            );

            -- Indexes for fast queries
            CREATE INDEX IF NOT EXISTS idx_conversations_session
                ON conversations(session_id);
            CREATE INDEX IF NOT EXISTS idx_conversations_created
                ON conversations(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_conversations_confidence
                ON conversations(confidence);
            CREATE INDEX IF NOT EXISTS idx_feedback_conversation
                ON feedback(conversation_id);
        """)
        logger.info("Database tables created/verified")


# ─── CONVERSATIONS ───────────────────────────────────────────

async def save_conversation(
    session_id: str,
    question: str,
    answer: str,
    confidence: float,
    model_used: str,
    sources_found: int,
    processing_ms: int,
) -> str:
    """Save a Q&A pair. Returns the conversation UUID."""
    pool = await get_pool()
    low_confidence = confidence < CONFIDENCE_THRESHOLD

    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            INSERT INTO conversations
                (session_id, question, answer, confidence, model_used,
                 sources_found, low_confidence, processing_ms)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id::text
        """, session_id, question, answer, confidence, model_used,
             sources_found, low_confidence, processing_ms)

    conv_id = row["id"]
    logger.debug(f"Saved conversation {conv_id} (confidence={confidence:.2f})")
    return conv_id


async def get_conversation(conversation_id: str) -> Optional[Dict]:
    """Fetch a single conversation by ID."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT c.*, f.rating, f.comment
            FROM conversations c
            LEFT JOIN feedback f ON f.conversation_id = c.id
            WHERE c.id = $1::uuid
        """, conversation_id)
    return dict(row) if row else None


async def get_session_history(session_id: str, limit: int = 20) -> List[Dict]:
    """Get conversation history for a session."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT c.id::text, c.session_id, c.question, c.answer,
                   c.confidence, c.model_used, c.created_at,
                   f.rating
            FROM conversations c
            LEFT JOIN feedback f ON f.conversation_id = c.id
            WHERE c.session_id = $1
            ORDER BY c.created_at DESC
            LIMIT $2
        """, session_id, limit)
    return [dict(r) for r in rows]


async def get_recent_conversations(limit: int = 20) -> List[Dict]:
    """Get recent conversations across all sessions."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT c.id::text, c.session_id, c.question, c.answer,
                   c.confidence, c.model_used, c.created_at,
                   f.rating
            FROM conversations c
            LEFT JOIN feedback f ON f.conversation_id = c.id
            ORDER BY c.created_at DESC
            LIMIT $1
        """, limit)
    return [dict(r) for r in rows]


# ─── FEEDBACK ────────────────────────────────────────────────

async def save_feedback(
    conversation_id: str,
    rating: int,
    comment: Optional[str] = None,
) -> bool:
    """Save user feedback for a conversation."""
    pool = await get_pool()
    try:
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO feedback (conversation_id, rating, comment)
                VALUES ($1::uuid, $2, $3)
                ON CONFLICT DO NOTHING
            """, conversation_id, rating, comment)
        logger.debug(f"Saved feedback for {conversation_id}: {rating}/5")
        return True
    except Exception as e:
        logger.error(f"Failed to save feedback: {e}")
        return False


# ─── KNOWLEDGE BASE METADATA ─────────────────────────────────

async def save_knowledge_source(
    source_name: str,
    source_type: str,
    chunks_count: int,
    file_size_kb: Optional[float] = None,
    url: Optional[str] = None,
) -> str:
    """Record a document or website added to the knowledge base."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            INSERT INTO knowledge_base
                (source_name, source_type, chunks_count, file_size_kb, url)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id::text
        """, source_name, source_type, chunks_count, file_size_kb, url)
    return row["id"]


async def get_knowledge_sources() -> List[Dict]:
    """List all documents/websites in the knowledge base."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id::text, source_name, source_type, chunks_count,
                   file_size_kb, url, created_at
            FROM knowledge_base
            ORDER BY created_at DESC
        """)
    return [dict(r) for r in rows]


async def delete_knowledge_source(source_id: str) -> bool:
    """Remove a knowledge source record."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute("""
            DELETE FROM knowledge_base WHERE id = $1::uuid
        """, source_id)
    return result == "DELETE 1"


# ─── PROCESSING TASKS ────────────────────────────────────────

async def create_task(filename: Optional[str] = None) -> str:
    """Create a background task record. Returns task ID."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            INSERT INTO processing_tasks (filename, status)
            VALUES ($1, 'processing')
            RETURNING id::text
        """, filename)
    return row["id"]


async def update_task(
    task_id: str,
    status: str,
    chunks_created: Optional[int] = None,
    error_message: Optional[str] = None,
):
    """Update a processing task status."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            UPDATE processing_tasks
            SET status = $2,
                chunks_created = COALESCE($3, chunks_created),
                error_message = $4,
                completed_at = CASE WHEN $2 IN ('complete', 'failed')
                               THEN NOW() ELSE NULL END
            WHERE id = $1::uuid
        """, task_id, status, chunks_created, error_message)


async def get_task(task_id: str) -> Optional[Dict]:
    """Get task status."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT id::text, filename, status, chunks_created,
                   error_message, created_at, completed_at
            FROM processing_tasks
            WHERE id = $1::uuid
        """, task_id)
    return dict(row) if row else None


# ─── ANALYTICS ───────────────────────────────────────────────

async def get_stats() -> Dict:
    """Aggregate statistics for the admin dashboard."""
    pool = await get_pool()
    async with pool.acquire() as conn:

        # Totals
        totals = await conn.fetchrow("""
            SELECT
                COUNT(*)                                    AS total_conversations,
                ROUND(AVG(confidence)::numeric, 3)         AS avg_confidence,
                COUNT(*) FILTER (WHERE low_confidence)     AS low_confidence_count,
                COUNT(*) FILTER (
                    WHERE created_at >= NOW() - INTERVAL '1 day'
                )                                          AS conversations_today,
                COUNT(*) FILTER (
                    WHERE created_at >= NOW() - INTERVAL '7 days'
                )                                          AS conversations_this_week
            FROM conversations
        """)

        # Average rating
        rating_row = await conn.fetchrow("""
            SELECT ROUND(AVG(rating)::numeric, 2) AS avg_rating
            FROM feedback
        """)

        # Total documents
        doc_row = await conn.fetchrow("""
            SELECT COUNT(*) AS total_documents FROM knowledge_base
        """)

        # Popular questions (top 10)
        popular = await conn.fetch("""
            SELECT
                question,
                COUNT(*)                           AS count,
                ROUND(AVG(confidence)::numeric, 3) AS avg_confidence,
                ROUND(AVG(f.rating)::numeric, 2)   AS avg_rating
            FROM conversations c
            LEFT JOIN feedback f ON f.conversation_id = c.id
            GROUP BY question
            ORDER BY count DESC
            LIMIT 10
        """)

    return {
        "total_conversations":    totals["total_conversations"] or 0,
        "avg_confidence":         float(totals["avg_confidence"] or 0),
        "low_confidence_count":   totals["low_confidence_count"] or 0,
        "conversations_today":    totals["conversations_today"] or 0,
        "conversations_this_week": totals["conversations_this_week"] or 0,
        "avg_rating":             float(rating_row["avg_rating"] or 0),
        "total_documents":        doc_row["total_documents"] or 0,
        "popular_questions": [
            {
                "question":       r["question"],
                "count":          r["count"],
                "avg_confidence": float(r["avg_confidence"] or 0),
                "avg_rating":     float(r["avg_rating"]) if r["avg_rating"] else None,
            }
            for r in popular
        ],
    }


async def get_low_confidence_answers(limit: int = 20) -> List[Dict]:
    """Fetch answers flagged as low confidence for human review."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT c.id::text, c.question, c.answer, c.confidence,
                   c.model_used, c.created_at, f.rating
            FROM conversations c
            LEFT JOIN feedback f ON f.conversation_id = c.id
            WHERE c.low_confidence = TRUE
            ORDER BY c.created_at DESC
            LIMIT $1
        """, limit)
    return [dict(r) for r in rows]
