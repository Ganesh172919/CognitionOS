"""
Authentication API Routes

Provides endpoints for user registration, login, and token management.
"""

import sys
import os
from datetime import timedelta
from typing import Optional

# Add core modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field

from services.api.src.auth import (
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    verify_token,
    CurrentUser,
    get_current_user,
)


router = APIRouter(prefix="/api/v3/auth", tags=["Authentication"])


# ==================== Request/Response Schemas ====================

class RegisterRequest(BaseModel):
    """User registration request"""
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="User password (min 8 characters)")
    full_name: Optional[str] = Field(default=None, description="User's full name")


class LoginRequest(BaseModel):
    """User login request"""
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="User password")


class TokenResponse(BaseModel):
    """Token response"""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")


class RefreshTokenRequest(BaseModel):
    """Refresh token request"""
    refresh_token: str = Field(..., description="JWT refresh token")


class UserResponse(BaseModel):
    """User information response"""
    user_id: str = Field(..., description="User ID")
    email: str = Field(..., description="User email")
    roles: list[str] = Field(..., description="User roles")
    full_name: Optional[str] = Field(default=None, description="User's full name")


# ==================== In-Memory User Store (Temporary) ====================
# TODO: Replace with database persistence

_users_db: dict[str, dict] = {}
_user_counter = 1


def _get_next_user_id() -> str:
    """Get next user ID"""
    global _user_counter
    user_id = f"user_{_user_counter}"
    _user_counter += 1
    return user_id


def _get_user_by_email(email: str) -> Optional[dict]:
    """Get user by email"""
    for user in _users_db.values():
        if user["email"] == email:
            return user
    return None


def _get_user_by_id(user_id: str) -> Optional[dict]:
    """Get user by ID"""
    return _users_db.get(user_id)


# ==================== Authentication Endpoints ====================

@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description="Create a new user account with email and password",
)
async def register(request: RegisterRequest) -> UserResponse:
    """Register a new user"""
    
    # Check if user already exists
    if _get_user_by_email(request.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists"
        )
    
    # Create new user
    user_id = _get_next_user_id()
    hashed_password = get_password_hash(request.password)
    
    user = {
        "user_id": user_id,
        "email": request.email,
        "hashed_password": hashed_password,
        "full_name": request.full_name,
        "roles": ["user"],  # Default role
        "is_active": True,
    }
    
    _users_db[user_id] = user
    
    return UserResponse(
        user_id=user_id,
        email=request.email,
        roles=user["roles"],
        full_name=request.full_name,
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="User login",
    description="Authenticate user and receive access and refresh tokens",
)
async def login(request: LoginRequest) -> TokenResponse:
    """Authenticate user and return tokens"""
    
    # Find user
    user = _get_user_by_email(request.email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify password
    if not verify_password(request.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user is active
    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    
    # Create tokens
    token_data = {
        "sub": user["user_id"],
        "email": user["email"],
        "roles": user["roles"],
    }
    
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)
    
    from core.config import get_config
    config = get_config()
    expires_in = config.security.access_token_expire_minutes * 60
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=expires_in,
    )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh access token",
    description="Use refresh token to obtain a new access token",
)
async def refresh_token(request: RefreshTokenRequest) -> TokenResponse:
    """Refresh access token"""
    
    # Verify refresh token
    payload = verify_token(request.refresh_token, token_type="refresh")
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user
    user_id = payload.get("sub")
    user = _get_user_by_id(user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    # Check if user is active
    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    
    # Create new tokens
    token_data = {
        "sub": user["user_id"],
        "email": user["email"],
        "roles": user["roles"],
    }
    
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)
    
    from core.config import get_config
    config = get_config()
    expires_in = config.security.access_token_expire_minutes * 60
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=expires_in,
    )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user",
    description="Get information about the currently authenticated user",
)
async def get_me(current_user: CurrentUser = Depends(get_current_user)) -> UserResponse:
    """Get current user information"""
    
    # Get full user data from database
    user = _get_user_by_id(current_user.user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserResponse(
        user_id=user["user_id"],
        email=user["email"],
        roles=user["roles"],
        full_name=user.get("full_name"),
    )
