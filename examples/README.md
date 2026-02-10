# CognitionOS Examples and Tests

This directory contains example workflows and integration tests for CognitionOS.

## Quick Start

### Prerequisites

1. **Services Running**: Ensure all CognitionOS services are running:
   ```bash
   docker-compose up -d
   ```

2. **API Keys** (Optional): For real LLM functionality, set in `.env`:
   ```bash
   OPENAI_API_KEY=sk-your-key-here
   # OR
   ANTHROPIC_API_KEY=sk-ant-your-key-here
   ```

3. **Python Dependencies**:
   ```bash
   pip install -r tests/requirements.txt
   ```

## Running Examples

### Example 1: Complete Workflow

Demonstrates the full CognitionOS workflow:

```bash
python examples/workflow_example.py
```

**What it does:**
- ✓ Checks system health
- ✓ Generates Python code using AI Runtime
- ✓ Stores generated code in Memory Service
- ✓ Retrieves memories using semantic search
- ✓ Demonstrates embedding generation

**Expected Output:**
```
======================================================================
CognitionOS Example Workflow
======================================================================

User ID: a1b2c3d4-...

[Step 1] Checking system health...
  AI Runtime: healthy
  - OpenAI available: True
  - Anthropic available: False
  - Simulation mode: False

[Step 2] Generating code with AI Runtime...
  ✓ Code generated successfully
  - Model: gpt-4-turbo-preview
  - Tokens: 234 (45 + 189)
  - Cost: $0.006120
  - Latency: 1523ms

[...more steps...]

Workflow Summary
  ✓ Generated 2 code functions
  ✓ Stored 2 memories with semantic search
  ✓ Retrieved memories using natural language queries

CognitionOS is ready for production use!
```

## Running Integration Tests

### Full Test Suite

Run all integration tests:

```bash
pytest tests/integration/test_integration.py -v
```

### Specific Test Classes

**Authentication Tests:**
```bash
pytest tests/integration/test_integration.py::TestAuthFlow -v
```

**AI Runtime Tests:**
```bash
pytest tests/integration/test_integration.py::TestAIRuntime -v
```

**Memory Service Tests:**
```bash
pytest tests/integration/test_integration.py::TestMemoryService -v
```

**End-to-End Workflow:**
```bash
pytest tests/integration/test_integration.py::TestFullWorkflow -v
```

**Performance Tests:**
```bash
pytest tests/integration/test_integration.py::TestPerformance -v
```

### Test Coverage

The test suite covers:

1. **Authentication Flow**
   - User registration
   - Login and JWT token generation

2. **AI Runtime Service**
   - Health checks
   - Model listing
   - Text completions
   - Embedding generation
   - Cost tracking
   - Latency measurement

3. **Memory Service**
   - Memory storage
   - Semantic retrieval
   - Memory updates
   - Access tracking

4. **Full Workflow**
   - Complete end-to-end task execution
   - AI generation → Memory storage → Retrieval

5. **Performance**
   - Concurrent request handling
   - Response time measurement

## Example Test Output

```bash
$ pytest tests/integration/test_integration.py::TestFullWorkflow -v -s

tests/integration/test_integration.py::TestFullWorkflow::test_complete_task_workflow

=== Checking service health ===
✓ AI Runtime: healthy
✓ Memory Service: healthy

=== Generating AI completion ===
✓ Generated code (234 tokens)
  Model: gpt-4-turbo-preview
  Cost: $0.006120

=== Storing in memory ===
✓ Stored memory ID: 8f9e7d6c-...

=== Retrieving from memory ===
✓ Retrieved 1 relevant memories
  Top match (score: 0.923):
  Factorial function implementation: def factorial(n: int) -> int:...

=== Workflow completed successfully! ===

PASSED
```

## Manual Testing

### Test AI Runtime Directly

```bash
# Check health
curl http://localhost:8005/health | jq

# List available models
curl http://localhost:8005/models | jq

# Generate completion
curl -X POST http://localhost:8005/complete \
  -H "Content-Type: application/json" \
  -d '{
    "role": "executor",
    "prompt": "Write hello world in Python",
    "context": [],
    "max_tokens": 200,
    "temperature": 0.7,
    "user_id": "00000000-0000-0000-0000-000000000001"
  }' | jq

# Generate embeddings
curl -X POST http://localhost:8005/embed \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Hello world", "How are you?"],
    "model": "text-embedding-ada-002"
  }' | jq
```

### Test Memory Service Directly

```bash
# Check health
curl http://localhost:8004/health | jq

# Store memory
curl -X POST http://localhost:8004/memories \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "00000000-0000-0000-0000-000000000001",
    "content": "Python is a programming language",
    "memory_type": "semantic",
    "scope": "user",
    "metadata": {},
    "source": "test",
    "confidence": 1.0,
    "is_sensitive": false
  }' | jq

# Retrieve memories
curl -X POST http://localhost:8004/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "00000000-0000-0000-0000-000000000001",
    "query": "What programming languages?",
    "k": 5,
    "min_confidence": 0.0
  }' | jq
```

## Troubleshooting

### Services Not Running

```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs ai-runtime
docker-compose logs memory-service

# Restart services
docker-compose restart
```

### Database Connection Issues

```bash
# Check PostgreSQL
docker-compose logs postgres

# Initialize database
docker-compose exec api-gateway python /app/scripts/init_database.py
```

### API Key Issues

If running without API keys (simulation mode):
- AI Runtime will return simulated responses
- Memory Service will use simulated embeddings
- Tests will still pass but won't use real AI

To enable real AI:
```bash
# Set in .env file
OPENAI_API_KEY=sk-your-key-here

# Restart services
docker-compose restart ai-runtime memory-service
```

### Test Failures

**Connection Refused:**
```bash
# Ensure services are running on correct ports
docker-compose ps

# Check if ports are accessible
curl http://localhost:8005/health
curl http://localhost:8004/health
```

**Timeout Errors:**
```bash
# Increase test timeout
pytest tests/integration/test_integration.py --timeout=60
```

**Database Errors:**
```bash
# Reset database
docker-compose down -v
docker-compose up -d
python scripts/init_database.py
```

## Performance Benchmarks

Expected performance (with real APIs):

| Operation | Latency | Cost |
|-----------|---------|------|
| Text Completion (gpt-4-turbo) | 1-3s | $0.01-0.05 |
| Text Completion (gpt-3.5-turbo) | 0.5-1s | $0.001-0.005 |
| Embedding Generation | 0.1-0.3s | $0.0001 |
| Memory Storage | 50-200ms | Free |
| Memory Retrieval | 100-500ms | Free |

## CI/CD Integration

To run tests in CI/CD:

```yaml
# GitHub Actions example
- name: Run Integration Tests
  run: |
    docker-compose up -d
    sleep 10  # Wait for services to start
    python scripts/init_database.py
    pytest tests/integration/test_integration.py -v
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## Contributing

When adding new tests:

1. Follow existing test structure
2. Use descriptive test names
3. Add docstrings explaining what's tested
4. Clean up test data after runs
5. Use appropriate timeouts
6. Test both success and failure cases

## Next Steps

1. **Load Testing**: Use `locust` for load testing
2. **Security Testing**: Penetration testing
3. **API Contract Tests**: OpenAPI schema validation
4. **Unit Tests**: Test individual service components
5. **E2E Tests**: Full user journey testing
