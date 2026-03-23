"""
learning_engine.py - The AI brain of the chatbot
Handles: embeddings, vector search (ChromaDB), answer generation (Ollama/Groq)
Uses RAG: Retrieval-Augmented Generation
"""

import logging
import re
import time
import httpx
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Tuple, Optional
from uuid import uuid4
from groq import Groq

from config import (
    CHROMA_PATH, CHROMA_COLLECTION, EMBEDDING_MODEL,
    OLLAMA_URL, SYSTEM_PROMPT, CHUNK_SIZE, CHUNK_OVERLAP,
    MAX_CHUNKS_PER_QUERY, CONFIDENCE_THRESHOLD, MIN_CONTEXT_LENGTH,
    get_model, GROQ_API_KEY, LLM_PROVIDER,
)

logger = logging.getLogger(__name__)

GREETING_RE = re.compile(r"^\s*(hi|hello|hey|greetings|good\s+morning|good\s+afternoon|good\s+evening)[\s!.,]*$", re.IGNORECASE)

def is_greeting(text: str) -> bool:
    """Detect simple greetings so the bot can respond naturally even without context."""
    return bool(GREETING_RE.match(text.strip()))


# ─── CHROMADB SETUP ──────────────────────────────────────────

def get_chroma_client() -> chromadb.PersistentClient:
    """Get persistent ChromaDB client (survives restarts)."""
    return chromadb.PersistentClient(path=CHROMA_PATH)


def get_collection(client: Optional[chromadb.PersistentClient] = None):
    """Get or create the university knowledge collection."""
    if client is None:
        client = get_chroma_client()

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for text
    )
    return collection


# ─── TEXT CHUNKING ───────────────────────────────────────────

def chunk_text(text: str, source_name: str) -> List[dict]:
    """
    Split text into overlapping chunks for better retrieval.
    Overlap preserves context at chunk boundaries.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + CHUNK_SIZE

        # Try to end at a natural boundary (sentence or paragraph)
        if end < len(text):
            # Look for sentence end near the chunk boundary
            for boundary in ["\n\n", "\n", ". ", "? ", "! "]:
                pos = text.rfind(boundary, start + CHUNK_SIZE // 2, end)
                if pos != -1:
                    end = pos + len(boundary)
                    break

        chunk_content = text[start:end].strip()

        if len(chunk_content) > 50:  # skip tiny fragments
            chunks.append({
                "id":      f"{source_name}_{chunk_index}_{uuid4().hex[:8]}",
                "text":    chunk_content,
                "source":  source_name,
                "index":   chunk_index,
            })
            chunk_index += 1

        # Move forward with overlap
        start = end - CHUNK_OVERLAP

    logger.debug(f"Chunked '{source_name}' into {len(chunks)} chunks")
    return chunks


# ─── ADD TO KNOWLEDGE BASE ───────────────────────────────────

def add_to_knowledge_base(
    text: str,
    source_name: str,
    batch_size: int = 50,
) -> int:
    """
    Add text to ChromaDB vector store.
    Returns number of chunks created.
    """
    chunks = chunk_text(text, source_name)
    if not chunks:
        logger.warning(f"No chunks generated for '{source_name}'")
        return 0

    collection = get_collection()

    # Add in batches to avoid memory issues with large documents
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        collection.add(
            ids=       [c["id"]   for c in batch],
            documents= [c["text"] for c in batch],
            metadatas= [{"source": c["source"], "index": c["index"]} for c in batch],
        )

    logger.info(f"Added {len(chunks)} chunks from '{source_name}' to ChromaDB")
    return len(chunks)


def remove_from_knowledge_base(source_name: str) -> bool:
    """Remove all chunks from a specific source."""
    try:
        collection = get_collection()
        collection.delete(where={"source": source_name})
        logger.info(f"Removed all chunks for source '{source_name}'")
        return True
    except Exception as e:
        logger.error(f"Failed to remove '{source_name}': {e}")
        return False


def get_knowledge_base_stats() -> dict:
    """Get ChromaDB collection statistics."""
    collection = get_collection()
    count = collection.count()
    return {
        "collection": CHROMA_COLLECTION,
        "total_chunks": count,
        "embedding_model": EMBEDDING_MODEL,
    }


# ─── VECTOR SEARCH ───────────────────────────────────────────

def search_knowledge_base(
    query: str,
    n_results: int = MAX_CHUNKS_PER_QUERY,
) -> Tuple[List[str], float]:
    """
    Search for relevant chunks using semantic similarity.
    Returns (list_of_relevant_texts, confidence_score).
    """
    collection = get_collection()

    if collection.count() == 0:
        logger.warning("Knowledge base is empty — no documents uploaded yet")
        return [], 0.0

    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, collection.count()),
            include=["documents", "distances", "metadatas"],
        )

        documents = results["documents"][0]
        distances = results["distances"][0]

        if not documents:
            return [], 0.0

        # Convert cosine distance (0=identical, 2=opposite) to confidence (0-1)
        # distance 0.0 → confidence 1.0
        # distance 1.0 → confidence 0.5
        # distance 2.0 → confidence 0.0
        avg_distance = sum(distances) / len(distances)
        confidence = max(0.0, min(1.0, 1.0 - (avg_distance / 2.0)))

        logger.debug(
            f"Found {len(documents)} chunks for query (confidence={confidence:.2f})"
        )
        return documents, confidence

    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return [], 0.0


# ─── OLLAMA COMMUNICATION ────────────────────────────────────

async def query_ollama(
    prompt: str,
    model: str,
    timeout: int = 180,
) -> str:
    """
    Send prompt to local Ollama and get response.
    Async HTTP call — doesn't block other requests.
    """
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": False,  # get full response at once
        "options": {
            "temperature": 0.3,    # lower = more factual, less creative
            "top_p": 0.9,
            "num_predict": 256,    # max tokens in response (smaller → faster)
        }
    }

    try:
        logger.info(f"Querying Ollama (model={model} timeout={timeout}s prompt_len={len(prompt)})")
        t0 = time.time()
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            elapsed = time.time() - t0
            logger.info(f"Ollama response received in {elapsed:.1f}s")
            return data.get("response", "").strip()

    except httpx.ConnectError:
        logger.error("Cannot connect to Ollama. Is it running? Run: ollama serve")
        raise ConnectionError(
            "Ollama is not running. Please start it with: ollama serve"
        )
    except httpx.TimeoutException:
        logger.error(f"Ollama timed out after {timeout}s")
        raise TimeoutError(f"AI model took too long to respond (>{timeout}s)")
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        raise RuntimeError(f"AI model error: {str(e)}")


async def query_groq(
    prompt: str,
    model: str,
    timeout: int = 60,
) -> str:
    """
    Send prompt to Groq API and get response.
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set")

    client = Groq(api_key=GROQ_API_KEY)

    try:
        logger.info(f"Querying Groq (model={model} timeout={timeout}s prompt_len={len(prompt)})")
        t0 = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.3,
        )
        elapsed = time.time() - t0
        logger.info(f"Groq response received in {elapsed:.1f}s")
        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"Groq error: {e}")
        raise RuntimeError(f"Groq API error: {str(e)}")


async def check_ollama_health() -> dict:
    """Check if Ollama is running and what models are available."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            response.raise_for_status()
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            return {"status": "ok", "models": models}
    except Exception as e:
        return {"status": "down", "error": str(e), "models": []}


# ─── MAIN: ASK A QUESTION ────────────────────────────────────

async def ask_question(
    question: str,
    model_speed: str = "balanced",
) -> dict:
    """
    Full RAG pipeline:
    1. Search knowledge base for relevant chunks
    2. Build context prompt
    3. Query AI provider (Ollama or Groq based on LLM_PROVIDER)
    4. Return answer with metadata

    Returns dict with: answer, confidence, model_used,
                       sources_found, low_confidence, processing_ms
    """
    start_time = time.time()
    model = get_model(model_speed)

    # Quick handling for greetings (avoid unnecessary model calls)
    if is_greeting(question):
        logger.info(f"Detected greeting: '{question}' - responding instantly")
        answer = (
            "Hello! I'm your university assistant. "
            "Ask me anything about admissions, programs, fees, campus life, or other university topics."
        )
        processing_ms = int((time.time() - start_time) * 1000)
        return {
            "answer": answer,
            "confidence": 1.0,
            "model_used": "system",
            "sources_found": 0,
            "low_confidence": False,
            "processing_ms": processing_ms,
        }

    # Step 1: Retrieve relevant context
    step1_start = time.time()
    relevant_chunks, confidence = search_knowledge_base(question)
    step1_ms = int((time.time() - step1_start) * 1000)
    logger.info(f"Vector search time: {step1_ms}ms (found {len(relevant_chunks)} chunks)")
    sources_found = len(relevant_chunks)

    # Step 2: Build context string
    if relevant_chunks:
        context = "\n\n---\n\n".join(relevant_chunks)
    else:
        context = "No relevant information found in the knowledge base."

    # Step 3: Check if context is substantial enough
    # For general university questions, allow AI to provide helpful guidance even without specific context
    general_questions = any(keyword in question.lower() for keyword in [
        'fee', 'cost', 'price', 'pay', 'tuition', 'admission', 'application',
        'campus', 'visit', 'contact', 'phone', 'email', 'address', 'location',
        'how much', 'how to', 'where', 'when'
    ])

    if len(context) < MIN_CONTEXT_LENGTH and not relevant_chunks and not general_questions:
        answer = (
            "I don't have enough information to answer that question. "
            "Please try uploading relevant university documents first, "
            "or contact the university directly for assistance."
        )
        confidence = 0.0
        model_used = "none"
    else:
        # Step 4: Build the prompt
        build_start = time.time()
        prompt = SYSTEM_PROMPT.format(
            context=context,
            question=question,
        )
        build_ms = int((time.time() - build_start) * 1000)
        logger.info(f"Prompt build time: {build_ms}ms (prompt length: {len(prompt)} chars)")

        # Step 5: Query AI provider
        try:
            if LLM_PROVIDER.lower() == "groq":
                answer = await query_groq(prompt, model)
            else:
                answer = await query_ollama(prompt, model)
            model_used = model
        except (ConnectionError, TimeoutError, RuntimeError, ValueError) as e:
            answer = f"I'm having trouble connecting to the AI engine. Error: {str(e)}"
            model_used = "error"
            confidence = 0.0

    processing_ms = int((time.time() - start_time) * 1000)
    low_confidence = confidence < CONFIDENCE_THRESHOLD

    if low_confidence and relevant_chunks:
        logger.warning(
            f"Low confidence answer ({confidence:.2f}) for: '{question[:60]}...'"
        )

    return {
        "answer":          answer,
        "confidence":      round(confidence, 4),
        "model_used":      model_used,
        "sources_found":   sources_found,
        "low_confidence":  low_confidence,
        "processing_ms":   processing_ms,
    }
