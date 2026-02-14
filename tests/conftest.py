"""
Pytest Configuration and Shared Fixtures

This module provides pytest configuration and shared fixtures for all tests.
"""

import sys
import os
import pytest
import asyncio
from typing import AsyncGenerator, Generator
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.api.src.main import app
from core.config import get_config


# Configure pytest
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Create async HTTP client for testing API"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def db_engine():
    """Create in-memory SQLite database engine for testing"""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    
    # Create tables
    # Note: In production, you'd import your models and use metadata.create_all
    # For now, we'll use a simplified approach
    
    yield engine
    
    await engine.dispose()


@pytest.fixture
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create database session for testing"""
    async_session = async_sessionmaker(
        db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()


@pytest.fixture
def mock_config(monkeypatch):
    """Mock configuration for testing"""
    config = get_config()
    
    # Override settings for testing
    monkeypatch.setattr(config, "environment", "test")
    monkeypatch.setattr(config.api, "debug", True)
    
    return config


@pytest.fixture
def test_user_data():
    """Sample user data for testing"""
    return {
        "email": "test@example.com",
        "password": "testpassword123",
        "full_name": "Test User",
    }


@pytest.fixture
def test_workflow_data():
    """Sample workflow data for testing"""
    return {
        "workflow_id": "test-workflow",
        "version": "1.0.0",
        "name": "Test Workflow",
        "description": "A test workflow",
        "steps": [
            {
                "step_id": "step-1",
                "name": "First Step",
                "type": "task",
                "config": {}
            }
        ],
        "metadata": {}
    }
