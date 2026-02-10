# Memory Model

## Overview

Memory is critical to CognitionOS's ability to maintain context, learn from experience, and provide personalized assistance. Unlike simple prompt-based systems, CognitionOS implements a sophisticated multi-layered memory architecture that scales with user interactions.

## Memory Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MEMORY HIERARCHY                         │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  WORKING MEMORY (Agent Context Window)              │  │
│  │  - Current task state                                │  │
│  │  - Immediate conversation                            │  │
│  │  - Volatile, cleared after task                      │  │
│  │  Storage: In-memory (RAM)                            │  │
│  │  Capacity: 8K-32K tokens                             │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ▲                                  │
│                          │ (load on-demand)                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  SHORT-TERM MEMORY (Session Memory)                  │  │
│  │  - Current goal and sub-tasks                        │  │
│  │  - Recent execution history                          │  │
│  │  - Session-specific preferences                      │  │
│  │  Storage: Redis (TTL: 24 hours)                      │  │
│  │  Capacity: ~100K tokens compressed                   │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ▲                                  │
│                          │ (retrieved by relevance)         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  LONG-TERM MEMORY (Semantic Memory)                  │  │
│  │  - User knowledge base                               │  │
│  │  - Learned facts and patterns                        │  │
│  │  - Domain expertise                                  │  │
│  │  Storage: PostgreSQL + Vector DB                     │  │
│  │  Capacity: Unlimited (compressed)                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ▲                                  │
│                          │ (queried by metadata)            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  EPISODIC MEMORY (Execution History)                 │  │
│  │  - Past task executions                              │  │
│  │  - Success/failure patterns                          │  │
│  │  - User corrections and feedback                     │  │
│  │  Storage: PostgreSQL (time-series)                   │  │
│  │  Capacity: Unlimited (partitioned)                   │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Memory Types

### 1. Working Memory

**Purpose**: Immediate context for active agent

**Structure**:
```python
class WorkingMemory:
    agent_id: str
    task_id: str
    context_window: List[Message]  # Recent conversation
    variables: Dict[str, Any]  # Temporary state
    loaded_memories: List[Memory]  # Retrieved from LTM
    created_at: datetime
    max_tokens: int  # Model-specific limit
```

**Lifecycle**:
- Created: When agent starts task
- Updated: As conversation progresses
- Cleared: When task completes or fails

**Management Strategy**:
- Sliding window: Keep last N messages
- Summarization: Compress old messages
- Eviction: Remove least recently used

### 2. Short-Term Memory

**Purpose**: Session-level persistence

**Structure**:
```python
class ShortTermMemory:
    session_id: str
    user_id: str
    goal: str
    task_graph: DAG
    execution_log: List[Event]
    context_summary: str
    created_at: datetime
    expires_at: datetime  # TTL: 24 hours
```

**Storage**: Redis with TTL

**Use Cases**:
- Resume interrupted sessions
- Provide context across tasks
- Track progress toward goal

**Eviction Policy**:
- Auto-expire after 24 hours
- Compress to long-term if important
- User can explicitly save

### 3. Long-Term Memory

**Purpose**: Persistent knowledge base

**Structure**:
```python
class LongTermMemory:
    id: str
    user_id: str
    content: str
    embedding: List[float]  # Vector representation
    memory_type: MemoryType  # FACT, PATTERN, PREFERENCE, SKILL
    metadata: Dict[str, Any]
    source: str  # Where this memory came from
    confidence: float  # 0.0 to 1.0
    created_at: datetime
    last_accessed: datetime
    access_count: int
    version: int  # For versioning/updates
```

**Memory Types**:

1. **FACT**: Objective information
   - Example: "User's company is Acme Corp"
   - Confidence: High (verified)

2. **PATTERN**: Learned behaviors
   - Example: "User prefers Python over JavaScript"
   - Confidence: Medium (inferred)

3. **PREFERENCE**: Explicit user choices
   - Example: "User likes detailed explanations"
   - Confidence: High (stated)

4. **SKILL**: Domain knowledge
   - Example: "User is expert in ML, beginner in DevOps"
   - Confidence: Medium (observed)

**Storage**:
- Text: PostgreSQL
- Vectors: pgvector extension or dedicated vector DB
- Embeddings: Generated via text-embedding-ada-002 or similar

### 4. Episodic Memory

**Purpose**: Execution history and timeline

**Structure**:
```python
class EpisodicMemory:
    id: str
    user_id: str
    session_id: str
    task_id: str
    timestamp: datetime
    event_type: EventType  # TASK_START, TOOL_EXEC, DECISION, TASK_END
    agent_id: str
    agent_role: str
    input_summary: str
    output_summary: str
    reasoning: str
    success: bool
    cost: float
    duration_ms: int
    metadata: Dict[str, Any]
```

**Storage**: PostgreSQL with time-series partitioning

**Use Cases**:
- Analyze what worked/failed
- Learn from past mistakes
- Provide audit trail
- Improve future planning

**Retention Policy**:
- Keep detailed logs for 30 days
- Compress to summaries after 30 days
- Archive after 1 year

## Memory Operations

### 1. Storage (Write)

```python
async def store_memory(
    user_id: str,
    content: str,
    memory_type: MemoryType,
    metadata: Dict[str, Any]
) -> Memory:
    # 1. Generate embedding
    embedding = await embedding_service.embed(content)

    # 2. Extract entities and keywords
    entities = nlp.extract_entities(content)
    keywords = nlp.extract_keywords(content)

    # 3. Check for duplicates
    similar = await vector_db.find_similar(
        embedding=embedding,
        user_id=user_id,
        threshold=0.95
    )

    if similar:
        # Update existing memory instead of creating duplicate
        return await update_memory(similar[0].id, content)

    # 4. Store in database
    memory = Memory(
        user_id=user_id,
        content=content,
        embedding=embedding,
        memory_type=memory_type,
        metadata={**metadata, "entities": entities, "keywords": keywords},
        created_at=datetime.now()
    )

    await db.insert(memory)
    return memory
```

### 2. Retrieval (Read)

```python
async def retrieve_memories(
    user_id: str,
    query: str,
    k: int = 5,
    memory_types: List[MemoryType] = None,
    time_range: Tuple[datetime, datetime] = None
) -> List[Memory]:
    # 1. Generate query embedding
    query_embedding = await embedding_service.embed(query)

    # 2. Vector similarity search
    candidates = await vector_db.search(
        embedding=query_embedding,
        user_id=user_id,
        k=k * 3,  # Get more candidates for reranking
        filters={
            "memory_type": memory_types,
            "created_at": time_range
        }
    )

    # 3. Apply time decay weighting
    now = datetime.now()
    for candidate in candidates:
        age_days = (now - candidate.created_at).days
        decay_factor = math.exp(-age_days / 30)  # Exponential decay
        candidate.score *= decay_factor

    # 4. Apply access frequency weighting
    for candidate in candidates:
        frequency_boost = math.log(1 + candidate.access_count)
        candidate.score *= (1 + frequency_boost * 0.1)

    # 5. Rerank and return top k
    ranked = sorted(candidates, key=lambda m: m.score, reverse=True)[:k]

    # 6. Update access statistics
    for memory in ranked:
        await db.update(memory.id, {
            "last_accessed": now,
            "access_count": memory.access_count + 1
        })

    return ranked
```

### 3. Update

```python
async def update_memory(
    memory_id: str,
    new_content: str,
    confidence_adjustment: float = 0.0
) -> Memory:
    # 1. Load existing memory
    memory = await db.get(memory_id)

    # 2. Create version history
    await db.insert_version(MemoryVersion(
        memory_id=memory_id,
        version=memory.version,
        content=memory.content,
        timestamp=datetime.now()
    ))

    # 3. Update with new content
    new_embedding = await embedding_service.embed(new_content)
    memory.content = new_content
    memory.embedding = new_embedding
    memory.version += 1
    memory.confidence = max(0.0, min(1.0, memory.confidence + confidence_adjustment))

    await db.update(memory)
    return memory
```

### 4. Deletion

```python
async def delete_memory(memory_id: str, soft_delete: bool = True):
    if soft_delete:
        # Mark as deleted but keep in database
        await db.update(memory_id, {"deleted": True})
    else:
        # Permanently remove (requires user confirmation)
        await db.delete(memory_id)
```

## Context Window Management

### The Problem
- Models have limited context windows (e.g., 8K, 32K, 128K tokens)
- Long conversations exceed this limit
- Simply truncating loses important context

### The Solution: Dynamic Context Compression

```python
async def optimize_context_window(
    task: Task,
    max_tokens: int
) -> ContextWindow:
    # 1. Start with essential context
    context = [
        system_prompt,  # Always included
        user_goal,      # Current objective
        task_definition # What we're doing now
    ]

    current_tokens = count_tokens(context)
    remaining_tokens = max_tokens - current_tokens - 1000  # Reserve for response

    # 2. Retrieve relevant memories
    memories = await retrieve_memories(
        user_id=task.user_id,
        query=task.description,
        k=10
    )

    # 3. Add memories by priority until budget exhausted
    for memory in memories:
        memory_tokens = count_tokens(memory.content)
        if current_tokens + memory_tokens > remaining_tokens:
            # Summarize remaining memories
            remaining = memories[memories.index(memory):]
            summary = await summarize_memories(remaining)
            context.append(summary)
            break
        else:
            context.append(memory.content)
            current_tokens += memory_tokens

    # 4. Add recent conversation history
    conversation = await get_recent_messages(task.session_id, limit=10)
    for msg in reversed(conversation):  # Newest first
        msg_tokens = count_tokens(msg)
        if current_tokens + msg_tokens > remaining_tokens:
            break
        context.insert(-1, msg)  # Insert before task definition
        current_tokens += msg_tokens

    return ContextWindow(
        messages=context,
        total_tokens=current_tokens,
        compression_ratio=calculate_compression(task, context)
    )
```

### Summarization Pipeline

When context is too large, we compress:

```python
async def summarize_memories(memories: List[Memory]) -> str:
    # Group by topic
    topics = cluster_by_topic(memories)

    summaries = []
    for topic, topic_memories in topics.items():
        # Extract key facts
        facts = extract_key_facts(topic_memories)

        # Generate concise summary
        summary = await ai_runtime.summarize(
            content=facts,
            max_length=200,
            style="bullet_points"
        )

        summaries.append(f"{topic}:\n{summary}")

    return "\n\n".join(summaries)
```

## Memory Isolation

### User Isolation (CRITICAL)

All memory operations MUST include user_id:

```sql
-- PostgreSQL Row-Level Security
CREATE POLICY user_isolation ON memories
    FOR ALL
    TO app_user
    USING (user_id = current_setting('app.user_id')::uuid);

-- All queries automatically filtered
SELECT * FROM memories WHERE embedding <-> query_embedding < 0.8;
-- Automatically becomes:
-- SELECT * FROM memories WHERE user_id = current_user_id
--   AND embedding <-> query_embedding < 0.8;
```

### Session Isolation

Memories can be scoped to session:

```python
class Memory:
    scope: MemoryScope  # USER (persistent) or SESSION (temporary)
```

## Memory Conflict Resolution

When memories contradict:

```python
async def resolve_conflict(memory1: Memory, memory2: Memory) -> Memory:
    # 1. Compare confidence scores
    if memory1.confidence > memory2.confidence + 0.2:
        return memory1
    elif memory2.confidence > memory1.confidence + 0.2:
        return memory2

    # 2. Compare recency (newer information may be more accurate)
    if memory2.created_at > memory1.created_at:
        # Ask user to confirm
        confirmed = await ask_user_confirmation(memory1, memory2)
        return confirmed

    # 3. Keep both with lower confidence
    memory1.confidence *= 0.8
    memory2.confidence *= 0.8
    await db.update(memory1)
    await db.update(memory2)

    return None  # Unresolved
```

## Memory Versioning

Track how memories evolve:

```python
class MemoryVersion:
    memory_id: str
    version: int
    content: str
    timestamp: datetime
    changed_by: str  # agent_id or "user"
    change_reason: str
```

This enables:
- Rollback to previous versions
- Audit trail of memory changes
- Understanding how knowledge evolves

## Privacy and Security

### Sensitive Data

```python
class Memory:
    is_sensitive: bool  # PII, credentials, etc.
    encryption_key_id: str  # If encrypted at rest
```

### User Control

Users can:
- View all stored memories
- Delete specific memories
- Export their data
- Clear all memories

### Compliance

- GDPR: Right to be forgotten (delete all user data)
- CCPA: Data export and deletion
- Encryption at rest and in transit

## Performance Optimization

### Caching

```python
# Redis cache for frequently accessed memories
@cache(ttl=3600)
async def get_user_preferences(user_id: str) -> Dict[str, Any]:
    memories = await retrieve_memories(
        user_id=user_id,
        memory_types=[MemoryType.PREFERENCE]
    )
    return {m.metadata["key"]: m.content for m in memories}
```

### Batch Operations

```python
# Embed multiple memories in one API call
embeddings = await embedding_service.embed_batch([
    memory.content for memory in memories
])
```

### Index Optimization

```sql
-- Vector similarity index
CREATE INDEX memories_embedding_idx ON memories
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Metadata filtering
CREATE INDEX memories_user_type_idx ON memories (user_id, memory_type);
CREATE INDEX memories_created_idx ON memories (created_at DESC);
```

## Monitoring

### Key Metrics

- **Storage**:
  - Total memories per user
  - Storage size per user
  - Growth rate

- **Retrieval**:
  - Query latency (p50, p95, p99)
  - Cache hit rate
  - Retrieval relevance score

- **Quality**:
  - Memory conflict rate
  - User deletion rate
  - Confidence distribution

### Alerts

- User exceeds memory quota
- Retrieval latency > 500ms
- High memory conflict rate
- Embedding service down

## Future Enhancements

1. **Federated Memory**: Share memories across users (with permission)
2. **Memory Graphs**: Represent memories as knowledge graphs
3. **Active Forgetting**: Automatically prune low-value memories
4. **Memory Synthesis**: Combine multiple memories into higher-level insights
5. **Cross-Modal Memory**: Images, audio, video alongside text
6. **Collaborative Memory**: Team-shared knowledge bases
