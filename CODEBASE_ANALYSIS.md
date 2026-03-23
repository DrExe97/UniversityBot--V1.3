# Codebase Architecture Analysis

## 1. Current Conversation Schema (PostgreSQL)

### Conversations Table
```sql
CREATE TABLE conversations (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id    TEXT NOT NULL,                          -- Groups conversations
    question      TEXT NOT NULL,
    answer        TEXT NOT NULL,
    confidence    FLOAT NOT NULL DEFAULT 0.0,            -- 0.0 to 1.0
    model_used    VARCHAR(50) NOT NULL DEFAULT 'mistral', -- system/ollama/groq
    sources_found INTEGER NOT NULL DEFAULT 0,            -- num of chunks retrieved
    low_confidence BOOLEAN NOT NULL DEFAULT FALSE,       -- confidence < THRESHOLD
    processing_ms INTEGER NOT NULL DEFAULT 0,            -- response time in ms
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_conversations_session ON conversations(session_id);
CREATE INDEX idx_conversations_created ON conversations(created_at DESC);
CREATE INDEX idx_conversations_confidence ON conversations(confidence);
```

**Key Points:**
- `session_id` is a **TEXT field** (not UUID), used to group related conversations
- NO `upload_path` column exists (see section 3)
- Each conversation is independent but grouped by `session_id`

---

## 2. Session Management (Current Implementation)

### How Sessions Work Today

**Session Creation:**
```python
# main.py: /ask endpoint
session_id = request.session_id or f"session_{uuid4().hex[:12]}"
# Example: "session_a1b2c3d4e5f6"
```

**Session Tracking:**
- Sessions are **implicit groupings** - no separate "sessions" table
- All conversations with the same `session_id` belong to one session
- Client bears responsibility for preserving `session_id` across requests

**Retrieving Session History:**
```python
async def get_session_history(session_id: str, limit: int = 20) -> List[Dict]:
    # Fetches all conversations with matching session_id
    SELECT c.*, f.rating FROM conversations c
    LEFT JOIN feedback f ON f.conversation_id = c.id
    WHERE c.session_id = $1
    ORDER BY c.created_at DESC
```

### Session Flow
```
Client                          API                         Database
  │                              │                              │
  ├─ POST /ask (no session_id)  │                              │
  │                              ├─ Generate: session_abc123   │
  │                              ├─────────────────────────────>│
  │                              │    Save conversation         │
  │                              │<─ Return session_id in response
  │<─ Receive session_id         │
  │
  ├─ POST /ask (with session_id) │
  │                              ├─────────────────────────────>│
  │                              │    Save with same session_id │
  │                              │<─ Return updated conversation
  │<─ Response                   │
  │
  └─ GET /conversations/{session_id}
                                 ├─────────────────────────────>│
                                 │    Retrieve all conversations│
                                 │    with this session_id      │
                                 │<─ Return history
```

---

## 3. Document Upload Handling (Current Implementation)

### Upload Flow

```
Client                    API                          Filesystem              Database
  │                        │                              │                       │
  ├─ POST /upload (file)   │                              │                       │
  │                        ├─ Generate safe_filename      │                       │
  │                        │  (uuid.hex + original name) │                       │
  │                        │                              │                       │
  │                        ├─ Save to uploads/ dir        │                       │
  │                        ├──────────────────────────────>│ Write temp file       │
  │                        │                              │                       │
  │                        ├─ Create processing_tasks     │                       │
  │                        ├─────────────────────────────────────────────────────>│
  │                        │  (status='processing')       │                       │
  │                        │                              │                       │
  │<─ Return task_id       │                              │                       │
  │   (in UploadResponse)  │                              │                       │
  │                        │                              │                       │
  │ (Background Task)      ├─ Extract text               │                       │
  │                        ├─ Add to ChromaDB            │                       │
  │                        │                              │                       │
  │                        ├─ Delete temp file           │                       │
  │                        ├──────────────────────────────>│ DELETE file          │
  │                        │                              │                       │
  │                        ├─ Record in knowledge_base    │                       │
  │                        ├─────────────────────────────────────────────────────>│
  │                        │  (source_name, chunks_count) │                       │
  │                        │                              │                       │
  │                        ├─ Update processing_tasks     │                       │
  │                        ├─────────────────────────────────────────────────────>│
  │                        │  (status='complete')         │                       │
```

### Upload Code (main.py)
```python
@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    # Step 1: Create safe filename with UUID prefix
    safe_filename = f"{uuid4().hex}_{file.filename}"
    filepath = os.path.join(UPLOAD_PATH, safe_filename)
    
    # Step 2: Save to disk temporarily
    with open(filepath, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Step 3: Create processing task record
    task_id = await db.create_task(filename=file.filename)
    
    # Step 4: Start background processing
    background_tasks.add_task(_process_document, filepath, file.filename, task_id)
    
    return UploadResponse(task_id=task_id, status="processing", ...)

# Background processing (runs in separate thread)
def _process_document(filepath: str, filename: str, task_id: str):
    # Extract text from doc
    text = extract_text(filepath)
    
    # Add chunks to ChromaDB vector DB
    chunks = ai.add_to_knowledge_base(text, source_name=filename)
    
    # Save metadata to PostgreSQL
    loop.run_until_complete(
        _bg_save_document(filename, chunks, meta["size_kb"], task_id)
    )
    
    # DELETE temp file after processing
    os.remove(filepath)  # <-- Key: File is removed!
```

### Background Database Save (main.py)
```python
async def _bg_save_document(filename, chunks, file_size_kb, task_id):
    # Record the source in knowledge_base table
    await conn.execute("""
        INSERT INTO knowledge_base (source_name, source_type, chunks_count, file_size_kb)
        VALUES ($1, 'document', $2, $3)
    """, filename, chunks, file_size_kb)
    
    # Mark task as complete
    await conn.execute("""
        UPDATE processing_tasks
        SET status = 'complete', chunks_created = $2, completed_at = NOW()
        WHERE id = $1::uuid
    """, task_id, chunks)
```

---

## 4. Database Tables for File Management

### Processing Tasks Table
```sql
CREATE TABLE processing_tasks (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename      TEXT,                              -- Original filename
    status        VARCHAR(20) NOT NULL DEFAULT 'processing',  -- processing|complete|failed
    chunks_created INTEGER,
    error_message TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at  TIMESTAMPTZ
);
```

### Knowledge Base Table
```sql
CREATE TABLE knowledge_base (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_name   TEXT NOT NULL,                     -- filename or "web:domain.com"
    source_type   VARCHAR(20) NOT NULL,              -- 'document' or 'website'
    chunks_count  INTEGER NOT NULL DEFAULT 0,         -- number of vector chunks
    file_size_kb  FLOAT,                              -- document size (if applicable)
    url           TEXT,                               -- website URL (if applicable)
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

---

## 5. Is `upload_path` Actually Used?

### Current Usage
✅ **YES** - But only temporarily:
1. Files are saved to `UPLOAD_PATH` (default: `./uploads`)
2. A unique filename is generated: `{uuid}_{original_filename}`
3. Background task processes the file
4. **After processing, the temporary file is DELETED** (`os.remove(filepath)`)

### NOT Stored in Database
❌ **NO** - The `upload_path` is:
- NOT a column in any table
- NOT linked to conversations
- NOT persisted after the upload task completes
- NOT used for retrieving documents later

### Data Actually Persisted
✅ Files are NOT re-stored in the database. Instead:
- **Text chunks** → Stored in ChromaDB (vector DB) for semantic search
- **Metadata** → Stored in `knowledge_base` table (filename, size, chunk count)
- **Task status** → Stored in `processing_tasks` table (for monitoring)
- **Original file** → DELETED after extraction

### Example
```
1. User uploads: "syllabus.pdf"
   ↓
2. Saved temporarily at: ./uploads/a1b2c3d4e5f6_syllabus.pdf
   ↓
3. Text extracted, chunked, sent to ChromaDB
   ↓
4. knowledge_base.source_name = "syllabus.pdf" (TEXT)
   ↓
5. DELETED from disk: rm ./uploads/a1b2c3d4e5f6_syllabus.pdf
```

---

## 6. Summary: Key Findings

| Aspect | Current Implementation | Storage | Notes |
|--------|----------------------|---------|-------|
| **Session Tracking** | Implicit via `session_id` TEXT field | PostgreSQL | Auto-generated if not provided; client must preserve |
| **Conversation Linking** | Via `session_id` in conversations table | PostgreSQL | No separate sessions table |
| **Upload Path** | Temporary filesystem storage only | Disk (deletes after processing) | NOT persisted in DB |
| **Document Content** | Chunked vectors in ChromaDB | ChromaDB | Semantic search optimization |
| **Document Metadata** | Filename, size, chunk count | PostgreSQL (knowledge_base) | Used for admin panel |
| **Processing Tracking** | Task ID → status updates | PostgreSQL (processing_tasks) | Allows polling via `/status/{task_id}` |
| **Feedback** | Linked to conversation_id | PostgreSQL (feedback table) | Optional 1-5 rating |

---

## 7. Potential Issues & Considerations

### ⚠️ Session Management
- Sessions are client-managed (stateless on server)
- No server-side session expiration/cleanup
- No protection against session ID enumeration

### ⚠️ Upload/Document Handling
- No audit trail of what files were uploaded and when
- No link between conversations and documents used
- Deleted files cannot be recovered
- No versioning of knowledge base

### ⚠️ Scalability
- Each conversation creates one DB row (OK for now)
- No pagination indices on sessions beyond `created_at`
- ChromaDB is persistent but not replicated
