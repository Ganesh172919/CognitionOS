# Database Module

This directory contains all database-related code for CognitionOS.

## Structure

```
database/
├── models.py              # SQLAlchemy ORM models
├── connection.py          # Database connection and session management
├── run_migrations.py      # Migration runner script
├── migrations/            # SQL migration files
│   └── 001_initial_schema.sql
└── README.md             # This file
```

## Database Schema

### Core Tables

- **users**: User accounts and authentication
- **sessions**: Active user sessions
- **tasks**: Task definitions and execution
- **task_execution_logs**: Detailed task logs
- **memories**: Multi-layer memory storage with vector embeddings
- **agents**: AI agent definitions
- **agent_task_assignments**: Agent-to-task mapping
- **tools**: Available tools for execution
- **tool_executions**: Tool execution history
- **conversations**: User conversations
- **messages**: Conversation messages
- **api_usage**: API endpoint usage tracking
- **llm_usage**: LLM token usage and costs

## Setup

### Prerequisites

- PostgreSQL 14+ with pgvector extension
- Python 3.11+

### Installation

1. Install PostgreSQL and pgvector:
```bash
# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib
sudo apt-get install postgresql-14-pgvector

# macOS
brew install postgresql pgvector
```

2. Create database and user:
```bash
sudo -u postgres psql
CREATE DATABASE cognitionos;
CREATE USER cognition WITH PASSWORD 'cognition';
GRANT ALL PRIVILEGES ON DATABASE cognitionos TO cognition;
\q
```

3. Set environment variable:
```bash
export DATABASE_URL="postgresql://cognition:cognition@localhost:5432/cognitionos"
```

## Running Migrations

### Initialize Database (First Time)

```bash
python database/run_migrations.py init
```

### Apply Pending Migrations

```bash
python database/run_migrations.py up
```

### Rollback Last Migration

```bash
python database/run_migrations.py down
```

## Usage in Services

### Async Database Access

```python
from database import get_db, User
from sqlalchemy import select

@app.get("/users/{user_id}")
async def get_user(user_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    return user
```

### Context Manager

```python
from database import get_db_context, Task, TaskStatus

async def process_task(task_id: str):
    async with get_db_context() as db:
        task = await db.get(Task, task_id)
        task.status = TaskStatus.IN_PROGRESS
        await db.commit()
```

### Vector Search Example

```python
from database import get_db, Memory
from sqlalchemy import select, func

async def search_memories(query_embedding: list[float], user_id: str, k: int = 5):
    async with get_db_context() as db:
        result = await db.execute(
            select(Memory)
            .where(Memory.user_id == user_id)
            .order_by(Memory.embedding.cosine_distance(query_embedding))
            .limit(k)
        )
        return result.scalars().all()
```

## Creating New Migrations

1. Create a new SQL file in `migrations/`:
```bash
touch database/migrations/002_add_feature.sql
```

2. Write your migration SQL:
```sql
-- Add new column
ALTER TABLE users ADD COLUMN preferences JSONB DEFAULT '{}';

-- Create index
CREATE INDEX idx_users_preferences ON users USING gin(preferences);
```

3. (Optional) Create rollback file:
```bash
touch database/migrations/002_add_feature_down.sql
```

```sql
-- Rollback
DROP INDEX IF EXISTS idx_users_preferences;
ALTER TABLE users DROP COLUMN preferences;
```

4. Run migration:
```bash
python database/run_migrations.py up
```

## Environment Variables

- `DATABASE_URL`: PostgreSQL connection string (default: `postgresql://cognition:cognition@postgres:5432/cognitionos`)
- `SQL_ECHO`: Enable SQL query logging (`true` or `false`, default: `false`)

## Docker Setup

The database is configured in `docker-compose.yml`:

```yaml
postgres:
  image: pgvector/pgvector:pg16
  environment:
    POSTGRES_USER: cognition
    POSTGRES_PASSWORD: cognition
    POSTGRES_DB: cognitionos
  volumes:
    - postgres_data:/var/lib/postgresql/data
  ports:
    - "5432:5432"
```

## Performance Considerations

### Indexes

All critical foreign keys and frequently queried columns have indexes:
- User lookups: `email`, `api_key`
- Task queries: `user_id`, `status`, `created_at`
- Memory search: Vector index (IVFFlat) on `embedding` column
- Session management: `token`, `expires_at`

### Connection Pooling

- Async pool size: 10 connections
- Max overflow: 20 connections
- Pre-ping: Enabled (detects stale connections)

### Vector Search Optimization

The `memories` table uses IVFFlat index for approximate nearest neighbor search:
```sql
CREATE INDEX idx_memories_embedding ON memories
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

For production, tune `lists` parameter based on dataset size:
- Small (< 1M rows): lists = 100
- Medium (1M-10M): lists = 1000
- Large (> 10M): lists = 10000

## Monitoring

### Check Connection Health

```python
from database import check_db_health

is_healthy = await check_db_health()
```

### Query Performance

Enable SQL logging for debugging:
```bash
export SQL_ECHO=true
```

### Connection Pool Stats

```python
from database import async_engine

pool = async_engine.pool
print(f"Pool size: {pool.size()}")
print(f"Checked out: {pool.checkedout()}")
```

## Backup and Recovery

### Backup Database

```bash
pg_dump -U cognition cognitionos > backup.sql
```

### Restore Database

```bash
psql -U cognition cognitionos < backup.sql
```

## Security

- Use environment variables for credentials (never commit passwords)
- Enable SSL in production: `?sslmode=require`
- Rotate API keys regularly
- Use prepared statements (SQLAlchemy does this automatically)
- Sanitize user input (Pydantic models handle this)

## Troubleshooting

### Connection Refused

```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Check connection
psql -U cognition -d cognitionos -h localhost
```

### pgvector Extension Not Found

```bash
# Install pgvector
sudo apt-get install postgresql-14-pgvector

# Enable in database
psql -U cognition cognitionos -c "CREATE EXTENSION vector;"
```

### Migration Failed

```bash
# Check applied migrations
psql -U cognition cognitionos -c "SELECT * FROM schema_migrations;"

# Manually rollback if needed
python database/run_migrations.py down
```
