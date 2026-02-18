"""
Authentication API Routes

Provides endpoints for user registration, login, and token management.
"""

import os
from datetime import timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.src.auth import (
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    verify_token,
    CurrentUser,
    get_current_user,
)
from services.api.src.dependencies.injection import get_db_session
from core.domain.auth.entities import User
from infrastructure.persistence.auth_repository import PostgresUserRepository


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


# ==================== Database Dependency ====================

async def get_user_repository(session: AsyncSession = Depends(get_db_session)) -> PostgresUserRepository:
    """Get user repository"""
    return PostgresUserRepository(session)


# ==================== Authentication Endpoints ====================

@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description="Create a new user account with email and password",
)
async def register(
    request: RegisterRequest,
    user_repo: PostgresUserRepository = Depends(get_user_repository),
    session: AsyncSession = Depends(get_db_session),
) -> UserResponse:
    """Register a new user"""
    
    # Check if user already exists
    if await user_repo.exists_by_email(request.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists"
        )
    
    # Create new user
    hashed_password = get_password_hash(request.password)
    user = User.create(
        email=request.email,
        password_hash=hashed_password,
        full_name=request.full_name,
    )
    
    # Save to database
    created_user = await user_repo.create(user)
    await session.commit()
    
    return UserResponse(
        user_id=str(created_user.user_id),
        email=created_user.email,
        roles=created_user.roles,
        full_name=created_user.full_name,
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="User login",
    description="Authenticate user and receive access and refresh tokens",
)
async def login(
    request: LoginRequest,
    user_repo: PostgresUserRepository = Depends(get_user_repository),
    session: AsyncSession = Depends(get_db_session),
) -> TokenResponse:
    """Authenticate user and return tokens"""
    
    # Find user
    user = await user_repo.find_by_email(request.email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify password
    if not verify_password(request.password, user.password_hash):
        # Record failed login attempt
        user.record_failed_login()
        await user_repo.update(user)
        await session.commit()
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user is active
    if not user.is_active():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive or locked"
        )
    
    # Record successful login
    user.record_login()
    await user_repo.update(user)
    await session.commit()
    
    # Create tokens
    token_data = {
        "sub": str(user.user_id),
        "email": user.email,
        "roles": user.roles,
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
async def refresh_token(
    request: RefreshTokenRequest,
    user_repo: PostgresUserRepository = Depends(get_user_repository),
) -> TokenResponse:
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
    from uuid import UUID
    user_id = UUID(payload.get("sub"))
    user = await user_repo.find_by_id(user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    # Check if user is active
    if not user.is_active():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive or locked"
        )
    
    # Create new tokens
    token_data = {
        "sub": str(user.user_id),
        "email": user.email,
        "roles": user.roles,
    }
    
    access_token = create_access_token(token_data)
    new_refresh_token = create_refresh_token(token_data)
    
    from core.config import get_config
    config = get_config()
    expires_in = config.security.access_token_expire_minutes * 60
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=expires_in,
    )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user",
    description="Get information about the currently authenticated user",
)
async def get_me(
    current_user: CurrentUser = Depends(get_current_user),
    user_repo: PostgresUserRepository = Depends(get_user_repository),
) -> UserResponse:
    """Get current user information"""
    
    # Get full user data from database
    from uuid import UUID
    user = await user_repo.find_by_id(UUID(current_user.user_id))
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserResponse(
        user_id=str(user.user_id),
        email=user.email,
        roles=user.roles,
        full_name=user.full_name,
    )
