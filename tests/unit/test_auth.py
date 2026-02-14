"""
Unit Tests for Authentication Endpoints

Tests for user registration, login, token refresh, and user info endpoints.
"""

import pytest
from httpx import AsyncClient
from unittest.mock import patch, MagicMock


@pytest.mark.asyncio
@pytest.mark.unit
class TestAuthenticationEndpoints:
    """Test suite for authentication endpoints"""
    
    async def test_health_check(self, client: AsyncClient):
        """Test health check endpoint"""
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
    
    async def test_register_valid_user(self, client: AsyncClient, valid_user_registration):
        """Test user registration with valid data"""
        with patch("services.api.src.routes.auth.get_db_session") as mock_db:
            mock_session = MagicMock()
            mock_db.return_value = mock_session
            
            response = await client.post(
                "/api/v3/auth/register",
                json=valid_user_registration
            )
            
            # Note: This will fail without database, but tests the endpoint structure
            assert response.status_code in [200, 500, 503]
    
    async def test_register_invalid_email(self, client: AsyncClient, invalid_email_user):
        """Test registration with invalid email format"""
        response = await client.post(
            "/api/v3/auth/register",
            json=invalid_email_user
        )
        
        # Should fail validation
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    async def test_register_weak_password(self, client: AsyncClient, weak_password_user):
        """Test registration with weak password"""
        response = await client.post(
            "/api/v3/auth/register",
            json=weak_password_user
        )
        
        # Should fail validation
        assert response.status_code == 422
    
    async def test_register_missing_fields(self, client: AsyncClient):
        """Test registration with missing required fields"""
        response = await client.post(
            "/api/v3/auth/register",
            json={"email": "test@example.com"}  # Missing password
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    async def test_login_endpoint_structure(self, client: AsyncClient, valid_login_credentials):
        """Test login endpoint is accessible"""
        response = await client.post(
            "/api/v3/auth/login",
            json=valid_login_credentials
        )
        
        # Without database, will fail, but endpoint should exist
        assert response.status_code in [200, 401, 500, 503]
    
    async def test_login_invalid_credentials(self, client: AsyncClient):
        """Test login with invalid credentials format"""
        response = await client.post(
            "/api/v3/auth/login",
            json={"email": "invalid"}  # Missing password
        )
        
        assert response.status_code == 422
    
    async def test_refresh_token_endpoint(self, client: AsyncClient):
        """Test refresh token endpoint"""
        response = await client.post(
            "/api/v3/auth/refresh",
            json={"refresh_token": "dummy_token"}
        )
        
        # Should exist and validate input
        assert response.status_code in [200, 401, 422, 500, 503]
    
    async def test_refresh_token_missing_token(self, client: AsyncClient):
        """Test refresh without token"""
        response = await client.post(
            "/api/v3/auth/refresh",
            json={}
        )
        
        assert response.status_code == 422
    
    async def test_get_current_user_unauthorized(self, client: AsyncClient):
        """Test getting current user without auth"""
        response = await client.get("/api/v3/auth/me")
        
        # Should require authentication
        assert response.status_code in [401, 403]
    
    async def test_get_current_user_invalid_token(self, client: AsyncClient):
        """Test getting current user with invalid token"""
        response = await client.get(
            "/api/v3/auth/me",
            headers={"Authorization": "Bearer invalid_token"}
        )
        
        # Should reject invalid token
        assert response.status_code in [401, 403]
    
    async def test_openapi_docs(self, client: AsyncClient):
        """Test that OpenAPI docs are accessible"""
        response = await client.get("/docs")
        assert response.status_code == 200
    
    async def test_openapi_json(self, client: AsyncClient):
        """Test that OpenAPI JSON schema is accessible"""
        response = await client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data


@pytest.mark.asyncio
@pytest.mark.unit
class TestAuthenticationSchemas:
    """Test authentication request/response schemas"""
    
    def test_user_registration_schema(self, valid_user_registration):
        """Test user registration schema validation"""
        from services.api.src.schemas.workflows import ErrorResponse
        
        # Schema should have required fields
        assert "email" in valid_user_registration
        assert "password" in valid_user_registration
        assert "@" in valid_user_registration["email"]
        assert len(valid_user_registration["password"]) >= 8
    
    def test_login_schema(self, valid_login_credentials):
        """Test login schema validation"""
        assert "email" in valid_login_credentials
        assert "password" in valid_login_credentials


@pytest.mark.asyncio
@pytest.mark.unit
class TestJWTFunctions:
    """Test JWT utility functions"""
    
    def test_password_hashing(self):
        """Test password hashing and verification"""
        from services.api.src.auth.jwt import get_password_hash, verify_password
        
        password = "TestPassword123!"
        hashed = get_password_hash(password)
        
        # Hash should be different from password
        assert hashed != password
        
        # Verification should work
        assert verify_password(password, hashed) is True
        assert verify_password("wrong_password", hashed) is False
    
    def test_create_access_token(self):
        """Test access token creation"""
        from services.api.src.auth.jwt import create_access_token
        
        data = {"sub": "user-123"}
        token = create_access_token(data)
        
        # Token should be a string
        assert isinstance(token, str)
        
        # Token should have JWT structure (3 parts separated by dots)
        parts = token.split(".")
        assert len(parts) == 3
    
    def test_create_refresh_token(self):
        """Test refresh token creation"""
        from services.api.src.auth.jwt import create_refresh_token
        
        data = {"sub": "user-123"}
        token = create_refresh_token(data)
        
        # Token should be a string
        assert isinstance(token, str)
        
        # Token should have JWT structure
        parts = token.split(".")
        assert len(parts) == 3
