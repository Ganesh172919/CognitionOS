# CognitionOS Test Suite

This directory contains the test suite for CognitionOS V3.

## Test Organization

```
tests/
├── unit/               # Unit tests for individual components
│   ├── test_auth.py           # Authentication tests
│   ├── test_workflows.py      # Workflow API tests
│   └── test_schemas.py        # Schema validation tests
├── integration/        # Integration tests
│   └── test_integration.py    # End-to-end integration tests
├── fixtures/           # Test fixtures and factories
│   ├── __init__.py
│   ├── users.py              # User fixtures
│   ├── workflows.py          # Workflow fixtures
│   └── database.py           # Database fixtures
├── conftest.py         # Shared pytest configuration
└── requirements.txt    # Test dependencies
```

## Running Tests

### All Tests
```bash
pytest
```

### Unit Tests Only
```bash
pytest tests/unit/
```

### Integration Tests Only
```bash
pytest tests/integration/
```

### With Coverage
```bash
pytest --cov=services/api --cov=infrastructure --cov-report=html
```

### Specific Test File
```bash
pytest tests/unit/test_auth.py -v
```

## Test Configuration

Tests use the following configuration:
- Async support via pytest-asyncio
- In-memory SQLite database for unit tests
- Docker containers for integration tests
- Mock LLM providers
- Test fixtures for common data

## Writing Tests

### Unit Test Example
```python
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_register_user(client: AsyncClient):
    response = await client.post(
        "/api/v3/auth/register",
        json={
            "email": "test@example.com",
            "password": "testpass123"
        }
    )
    assert response.status_code == 200
    assert "user_id" in response.json()
```

### Integration Test Example
```python
import pytest

@pytest.mark.asyncio
@pytest.mark.integration
async def test_workflow_execution_flow(client, db_session):
    # Create workflow
    # Execute workflow
    # Verify results
    pass
```

## Coverage Goals

- Unit Tests: 95%+ coverage
- Integration Tests: All critical paths
- E2E Tests: Key user workflows
