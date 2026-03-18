"""
models.py - Pydantic schemas for request/response validation
FastAPI uses these to auto-validate all incoming and outgoing data
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
from uuid import UUID, uuid4
from enum import Enum


# ─── ENUMS ───────────────────────────────────────────────────

class ModelSpeed(str, Enum):
    fast     = "fast"
    balanced = "balanced"
    accurate = "accurate"


class DocumentStatus(str, Enum):
    processing = "processing"
    complete   = "complete"
    failed     = "failed"


# ─── QUESTION / ANSWER ───────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, description="The question to ask")
    session_id: Optional[str] = Field(None, description="Session ID to group conversation (auto-generated if not provided)")
    model_speed: ModelSpeed = Field(ModelSpeed.balanced, description="Model speed: fast / balanced / accurate")

    @validator("question")
    def question_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty or whitespace")
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the admission requirements for undergraduate programs?",
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "model_speed": "balanced"
            }
        }


class QuestionResponse(BaseModel):
    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    conversation_id: str
    session_id: str
    model_used: str
    sources_found: int = Field(..., description="Number of relevant chunks found")
    low_confidence: bool = Field(False, description="True if confidence below threshold")
    processing_time_ms: int

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Undergraduate admission requires a minimum GPA of 3.0...",
                "confidence": 0.87,
                "conversation_id": "abc123",
                "session_id": "sess_xyz",
                "model_used": "mistral",
                "sources_found": 3,
                "low_confidence": False,
                "processing_time_ms": 1240
            }
        }


# ─── FEEDBACK ────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    conversation_id: str = Field(..., description="ID of the conversation to rate")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 (bad) to 5 (excellent)")
    comment: Optional[str] = Field(None, max_length=500, description="Optional comment")

    class Config:
        json_schema_extra = {
            "example": {
                "conversation_id": "abc123",
                "rating": 4,
                "comment": "Good answer but could include more details"
            }
        }


class FeedbackResponse(BaseModel):
    success: bool
    message: str


# ─── DOCUMENT UPLOAD ─────────────────────────────────────────

class UploadResponse(BaseModel):
    task_id: str
    status: DocumentStatus
    filename: str
    message: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: DocumentStatus
    filename: Optional[str] = None
    chunks_created: Optional[int] = None
    error: Optional[str] = None
    completed_at: Optional[datetime] = None


# ─── WEB CRAWLER ─────────────────────────────────────────────

class CrawlRequest(BaseModel):
    url: str = Field(..., description="Starting URL to crawl")
    max_pages: int = Field(20, ge=1, le=50, description="Maximum pages to crawl")

    @validator("url")
    def url_must_start_with_http(cls, v):
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://www.university.edu/admissions",
                "max_pages": 15
            }
        }


class CrawlResponse(BaseModel):
    task_id: str
    status: DocumentStatus
    start_url: str
    message: str


# ─── ANALYTICS / STATS ───────────────────────────────────────

class PopularQuestion(BaseModel):
    question: str
    count: int
    avg_confidence: float
    avg_rating: Optional[float]


class StatsResponse(BaseModel):
    total_conversations: int
    total_documents: int
    avg_confidence: float
    avg_rating: float
    low_confidence_count: int
    popular_questions: List[PopularQuestion]
    conversations_today: int
    conversations_this_week: int


# ─── KNOWLEDGE BASE ──────────────────────────────────────────

class KnowledgeItem(BaseModel):
    id: str
    source_name: str
    source_type: str        # "document" or "website"
    chunks_count: int
    created_at: datetime
    file_size_kb: Optional[float] = None
    url: Optional[str] = None


class KnowledgeListResponse(BaseModel):
    items: List[KnowledgeItem]
    total: int


# ─── CONVERSATION HISTORY ─────────────────────────────────────

class ConversationItem(BaseModel):
    id: str
    session_id: str
    question: str
    answer: str
    confidence: float
    model_used: str
    rating: Optional[int] = None
    created_at: datetime


class ConversationHistoryResponse(BaseModel):
    session_id: str
    conversations: List[ConversationItem]
    total: int


# ─── HEALTH CHECK ─────────────────────────────────────────────

class ServiceStatus(BaseModel):
    status: str             # "ok", "degraded", "down"
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str
    services: dict[str, ServiceStatus]
    timestamp: datetime
