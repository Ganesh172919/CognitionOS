# Memory Service

Manages long-term, short-term, and episodic memory for CognitionOS agents.

## Purpose

The Memory Service provides persistent storage and retrieval of memories across multiple layers, enabling agents to maintain context, learn from experience, and provide personalized responses.

## Features

- **Multi-Layer Memory**: Working, short-term, long-term, and episodic
- **Vector Storage**: Semantic search with embeddings
- **Memory Isolation**: User-scoped memories with row-level security
- **Context Compression**: Automatic summarization of old memories
- **Time Decay**: Weight recent memories higher
- **Conflict Resolution**: Handle contradictory memories

## Memory Types

### 1. Long-Term Memory
- Persistent semantic storage
- Vector embeddings for similarity search
- Metadata filtering (type, confidence, tags)
- User preferences and learned patterns

### 2. Short-Term Memory
- Session-scoped (24h TTL)
- Current goal and task context
- Recent execution history
- Stored in Redis for fast access

### 3. Episodic Memory
- Complete execution history
- Task outcomes and learnings
- Agent decision logs
- Performance metrics

## API Endpoints

### Memory Operations
- `POST /memories` - Store new memory
- `GET /memories/search` - Semantic search
- `GET /memories/{id}` - Get specific memory
- `PUT /memories/{id}` - Update memory
- `DELETE /memories/{id}` - Delete memory

### Retrieval
- `POST /retrieve` - Retrieve relevant memories for context
- `POST /embed` - Generate embeddings for text

## Environment Variables

```
MEMORY_SERVICE_PORT=8004
DATABASE_URL=postgresql://user:pass@localhost:5432/cognition
REDIS_URL=redis://localhost:6379/0
VECTOR_DIMENSION=1536
EMBEDDING_MODEL=text-embedding-ada-002
MAX_MEMORIES_PER_USER=10000
DEFAULT_RETRIEVAL_K=5
```

## Tech Stack

- FastAPI for REST API
- PostgreSQL with pgvector extension
- Redis for short-term memory
- OpenAI embeddings API
