"""
Test Fixtures for Users

Provides fixtures and factories for user-related testing.
"""

import pytest
from typing import Dict, Any


@pytest.fixture
def valid_user_registration():
    """Valid user registration data"""
    return {
        "email": "newuser@example.com",
        "password": "SecurePass123!",
        "full_name": "New Test User",
    }


@pytest.fixture
def valid_login_credentials():
    """Valid login credentials"""
    return {
        "email": "test@example.com",
        "password": "testpassword123",
    }


@pytest.fixture
def invalid_email_user():
    """User data with invalid email"""
    return {
        "email": "not-an-email",
        "password": "SecurePass123!",
        "full_name": "Invalid Email User",
    }


@pytest.fixture
def weak_password_user():
    """User data with weak password"""
    return {
        "email": "user@example.com",
        "password": "weak",
        "full_name": "Weak Password User",
    }


@pytest.fixture
def admin_user_data():
    """Admin user data"""
    return {
        "email": "admin@example.com",
        "password": "AdminPass123!",
        "full_name": "Admin User",
        "roles": ["admin", "user"],
    }


@pytest.fixture
def mock_jwt_token():
    """Mock JWT token for testing"""
    return "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyLTEyMyIsImV4cCI6OTk5OTk5OTk5OX0.mock_signature"


class UserFactory:
    """Factory for creating test users"""
    
    @staticmethod
    def create_user(
        email: str = "factory@example.com",
        password: str = "FactoryPass123!",
        full_name: str = "Factory User",
        roles: list = None,
    ) -> Dict[str, Any]:
        """Create a user with specified attributes"""
        return {
            "email": email,
            "password": password,
            "full_name": full_name,
            "roles": roles or ["user"],
        }
    
    @staticmethod
    def create_batch(count: int = 5) -> list:
        """Create multiple test users"""
        return [
            UserFactory.create_user(
                email=f"user{i}@example.com",
                full_name=f"Test User {i}",
            )
            for i in range(count)
        ]


@pytest.fixture
def user_factory():
    """User factory fixture"""
    return UserFactory
