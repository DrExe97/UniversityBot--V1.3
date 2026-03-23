# Document Upload & Conversation History Guide

## 📁 Document Upload Storage

### Where Are Uploaded Documents Stored?

Your uploaded documents are **not permanently stored in the `./uploads/` folder**. Instead, they follow this process:

1. **Temporary Storage**: Files are saved to `./uploads/{uuid}_{filename}` temporarily
2. **Text Extraction**: The document text is extracted (PDFs, DOCX, images, etc.)
3. **Vector Indexing**: The text is split into chunks and indexed in **ChromaDB** (vector database)
4. **Metadata Saved**: Document metadata is stored in PostgreSQL `knowledge_base` table
5. **Original File Deleted**: The original uploaded file is deleted to save storage

### This is By Design

This approach offers several benefits:
- ✅ **Efficient Storage**: Original files take up disk space; chunks in ChromaDB are optimized
- ✅ **Fast Search**: Vector embeddings enable semantic search across all documents
- ✅ **Easy Management**: Metadata in PostgreSQL tracks all documents
- ✅ **Clean Architecture**: No file system clutter

### View Your Uploaded Documents

You can see all uploaded documents and their processing status in the **Admin Panel**:

1. Access: `http://localhost:8000/admin.html`
2. Go to the **Knowledge Base** tab
3. View all uploaded documents with:
   - Document name/source
   - Upload date
   - Number of chunks created
   - File size
   - Chunk count

### Document Endpoints

**Get all document metadata:**
```bash
GET http://localhost:8000/knowledge-base
```

**Get document by source name:**
```bash
GET http://localhost:8000/knowledge-base/{source_name}
```

### Database Location

Your document metadata is in PostgreSQL:
```sql
SELECT * FROM knowledge_base;
-- Shows: source_name, source_type, chunks_count, file_size_kb, created_at
```

---

## 💬 Conversation History

### How It Works

- **Sessions**: Each chat session has a unique session ID (stored in browser localStorage)
- **Session ID Format**: `session_{12-random-hex-characters}`
- **Persistence**: Conversations are stored in PostgreSQL `conversations` table
- **Grouping**: All messages with the same `session_id` belong to one conversation thread

### View Conversation History

#### In the Widget
1. Click the **📋 History button** in the chat panel header
2. See all conversations in your current session
3. Click any conversation to select it

#### Via API
```bash
# Get all conversations for a session
GET http://localhost:8000/conversations/{session_id}

# Example:
curl "http://localhost:8000/conversations/session_a1b2c3d4e5f6g7h8"
```

#### In Admin Panel
1. Access: `http://localhost:8000/admin.html`
2. Go to **Conversations** tab
3. Search by session ID or view recent conversations
4. Click conversation to see full details

### Database Location

Your conversations are in PostgreSQL:
```sql
SELECT * FROM conversations WHERE session_id = 'session_xxx';
-- Shows: id, session_id, question, answer, confidence, model_used, created_at, etc.
```

### Session Management

**Current Session ID:**
- Displayed in browser console: `console.log(sessionId)`
- Stored in browser: `localStorage.getItem('uni_session')`

**Export Session History:**
```bash
curl "http://localhost:8000/conversations/{session_id}" > my_session.json
```

---

## 🗑️ Cleanup & Management

### Delete Old Documents
Use the Admin Panel or API:
```bash
DELETE http://localhost:8000//documents/{source_name}
```

### Archive Conversation History
Export via API, then delete old sessions from PostgreSQL if needed.

### Database Backup
Since everything is in PostgreSQL:
```bash
pg_dump university_chatbot > backup.sql
```

---

## 🔍 Quick Reference

| Item | Storage | Access |
|------|---------|--------|
| **Uploaded documents** | ChromaDB (vectors) + PostgreSQL (metadata) | Admin Panel / API |
| **Document chunks** | ChromaDB vector database | Automatic (RAG search) |
| **Conversation history** | PostgreSQL `conversations` table | Widget / Admin Panel / API |
| **Sessions** | Browser localStorage + PostgreSQL | Widget history / API |
| **Original files** | Deleted after processing | Not available |

---

## ❓ FAQ

**Q: Can I get my original uploaded file back?**
A: No, the original file is deleted after text extraction to save storage. However, all the text content is preserved in ChromaDB and searchable.

**Q: Why is my document in the knowledge base but not in uploads folder?**
A: The file was processed and deleted. Check the knowledge base table in the database to see it exists.

**Q: How do I prevent documents from being deleted?**
A: Modify the document upload handler in `main.py` to skip the deletion step (around line 150).

**Q: Can I export my conversation history?**
A: Yes! Use the API endpoint or export directly from PostgreSQL.

**Q: How long is history kept?**
A: Indefinitely, until you manually delete it from the database.
