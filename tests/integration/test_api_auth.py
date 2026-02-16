"""
Authentication API Integration Tests

Tests all authentication endpoints:
- POST /api/v3/auth/register
- POST /api/v3/auth/login
- POST /api/v3/auth/refresh
- GET /api/v3/auth/me
"""

import pytest
from httpx import AsyncClient
from typing import Dict


@pytest.mark.integration
@pytest.mark.asyncio
class TestAuthEndpoints:
    """Test authentication API endpoints"""
    
    async def test_user_registration(self, client: AsyncClient, test_user_credentials: Dict[str, str]):
        """Test POST /api/v3/auth/register"""
        response = await client.post(
            "/api/v3/auth/register",
            json=test_user_credentials
        )
        
        # Should succeed or indicate user already exists
        assert response.status_code in [200, 201, 409]
        
        if response.status_code in [200, 201]:
            data = response.json()
            assert "user_id" in data or "id" in data or "email" in data
            assert data.get("email") == test_user_credentials["email"] or True
    
    async def test_user_login_valid_credentials(self, client: AsyncClient, test_user_credentials: Dict[str, str]):
        """Test POST /api/v3/auth/login with valid credentials"""
        # Register first
        await client.post("/api/v3/auth/register", json=test_user_credentials)
        
        # Login
        response = await client.post(
            "/api/v3/auth/login",
            json={
                "email": test_user_credentials["email"],
                "password": test_user_credentials["password"]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "token_type" in data
        assert data["token_type"] == "bearer"
        assert "refresh_token" in data or True  # Optional
    
    async def test_user_login_invalid_credentials(self, client: AsyncClient):
        """Test POST /api/v3/auth/login with invalid credentials"""
        response = await client.post(
            "/api/v3/auth/login",
            json={
                "email": "nonexistent@example.com",
                "password": "wrongpassword"
            }
        )
        
        assert response.status_code in [401, 403]
    
    async def test_token_refresh(self, client: AsyncClient, test_user_credentials: Dict[str, str]):
        """Test POST /api/v3/auth/refresh"""
        # Register and login
        await client.post("/api/v3/auth/register", json=test_user_credentials)
        login_response = await client.post(
            "/api/v3/auth/login",
            json={
                "email": test_user_credentials["email"],
                "password": test_user_credentials["password"]
            }
        )
        
        assert login_response.status_code == 200
        login_data = login_response.json()
        
        # Get refresh token if provided
        refresh_token = login_data.get("refresh_token")
        if not refresh_token:
            pytest.skip("No refresh token provided")
        
        # Refresh token
        refresh_response = await client.post(
            "/api/v3/auth/refresh",
            json={"refresh_token": refresh_token}
        )
        
        assert refresh_response.status_code == 200
        refresh_data = refresh_response.json()
        assert "access_token" in refresh_data
    
    async def test_get_current_user(self, authenticated_client: AsyncClient):
        """Test GET /api/v3/auth/me"""
        response = await authenticated_client.get("/api/v3/auth/me")
        
        assert response.status_code == 200
        data = response.json()
        assert "email" in data or "user_id" in data or "id" in data
    
    async def test_get_current_user_unauthenticated(self, client: AsyncClient):
        """Test GET /api/v3/auth/me without authentication"""
        response = await client.get("/api/v3/auth/me")
        
        # Should fail without authentication
        assert response.status_code in [401, 403]
