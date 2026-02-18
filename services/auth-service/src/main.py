"""
Auth Service main application.

Handles user authentication, JWT token generation, and session management.
"""

import os

# Add shared libs to path

from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

import bcrypt
import jwt
from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, validator

from shared.libs.config import AuthServiceConfig, load_config
from shared.libs.logger import setup_logger, get_contextual_logger
from shared.libs.models import User, UserRole, ErrorResponse
from shared.libs.middleware import (
    TracingMiddleware,
    LoggingMiddleware,
    RateLimitMiddleware,
    ErrorHandlingMiddleware
)
from shared.libs.utils import generate_token, normalize_email


# Configuration
config = load_config(AuthServiceConfig)
logger = setup_logger(__name__, level=config.log_level)

# FastAPI app
app = FastAPI(
    title="CognitionOS Auth Service",
    version=config.service_version,
    description="Authentication and authorization service"
)

# Add middleware
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(TracingMiddleware)
app.add_middleware(
    RateLimitMiddleware,
    capacity=config.rate_limit_per_minute if hasattr(config, 'rate_limit_per_minute') else 60,
    refill_rate=1.0
)


# ============================================================================
# Request/Response Models
# ============================================================================

class RegisterRequest(BaseModel):
    """User registration request."""
    email: EmailStr
    username: str
    password: str

    @validator('username')
    def validate_username(cls, v):
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters')
        if not v.isalnum():
            raise ValueError('Username must be alphanumeric')
        return v

    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v


class LoginRequest(BaseModel):
    """User login request."""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class UserResponse(BaseModel):
    """User information response."""
    id: UUID
    email: str
    username: str
    role: UserRole
    created_at: datetime
    last_login: Optional[datetime]


class RefreshRequest(BaseModel):
    """Refresh token request."""
    refresh_token: str


# ============================================================================
# In-Memory Storage (Replace with Database in Production)
# ============================================================================

# In production, use PostgreSQL + Redis
users_db = {}  # email -> User
refresh_tokens_db = {}  # refresh_token -> user_id


# ============================================================================
# Authentication Logic
# ============================================================================

def hash_password(password: str) -> str:
    """Hash password using bcrypt."""
    salt = bcrypt.gensalt(rounds=config.bcrypt_rounds)
    return bcrypt.hashpw(password.encode(), salt).decode()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against hash."""
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def create_access_token(user_id: UUID, role: UserRole) -> str:
    """Create JWT access token."""
    payload = {
        "sub": str(user_id),
        "role": role.value,
        "type": "access",
        "exp": datetime.utcnow() + timedelta(
            minutes=config.jwt_access_token_expire_minutes
        ),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, config.jwt_secret, algorithm=config.jwt_algorithm)


def create_refresh_token(user_id: UUID) -> str:
    """Create JWT refresh token."""
    token = generate_token(32)
    refresh_tokens_db[token] = {
        "user_id": user_id,
        "expires_at": datetime.utcnow() + timedelta(
            days=config.jwt_refresh_token_expire_days
        )
    }
    return token


def decode_access_token(token: str) -> dict:
    """Decode and validate JWT access token."""
    try:
        payload = jwt.decode(
            token,
            config.jwt_secret,
            algorithms=[config.jwt_algorithm]
        )
        if payload.get("type") != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


def get_current_user(authorization: str = Header(...)) -> User:
    """
    Dependency to get current authenticated user from JWT.

    Args:
        authorization: Authorization header (Bearer token)

    Returns:
        Current user

    Raises:
        HTTPException: If token is invalid or user not found
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header"
        )

    token = authorization[7:]  # Remove "Bearer " prefix
    payload = decode_access_token(token)

    user_id = UUID(payload["sub"])

    # In production, fetch from database
    # For now, search in-memory
    user = None
    for u in users_db.values():
        if u.id == user_id:
            user = u
            break

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    return user


# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(request: RegisterRequest):
    """
    Register a new user.

    Creates a new user account with hashed password.
    """
    log = get_contextual_logger(__name__, action="register")

    # Normalize email
    email = normalize_email(request.email)

    # Check if user already exists
    if email in users_db:
        log.warning("Registration failed: email already exists", extra={"email": email})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Hash password
    password_hash = hash_password(request.password)

    # Create user
    user = User(
        email=email,
        username=request.username,
        password_hash=password_hash,
        role=UserRole.USER
    )

    # Save to database (in-memory for now)
    users_db[email] = user

    log.info("User registered successfully", extra={"user_id": str(user.id), "email": email})

    return UserResponse(
        id=user.id,
        email=user.email,
        username=user.username,
        role=user.role,
        created_at=user.created_at,
        last_login=user.last_login
    )


@app.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """
    Login and receive JWT tokens.

    Returns access token (short-lived) and refresh token (long-lived).
    """
    log = get_contextual_logger(__name__, action="login")

    # Normalize email
    email = normalize_email(request.email)

    # Get user
    user = users_db.get(email)
    if not user:
        log.warning("Login failed: user not found", extra={"email": email})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    # Verify password
    if not verify_password(request.password, user.password_hash):
        log.warning("Login failed: invalid password", extra={"email": email})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    # Update last login
    user.last_login = datetime.utcnow()

    # Generate tokens
    access_token = create_access_token(user.id, user.role)
    refresh_token = create_refresh_token(user.id)

    log.info("User logged in successfully", extra={"user_id": str(user.id), "email": email})

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=config.jwt_access_token_expire_minutes * 60
    )


@app.post("/refresh", response_model=TokenResponse)
async def refresh(request: RefreshRequest):
    """
    Refresh access token using refresh token.

    Generates a new access token if refresh token is valid.
    """
    log = get_contextual_logger(__name__, action="refresh")

    # Validate refresh token
    token_data = refresh_tokens_db.get(request.refresh_token)
    if not token_data:
        log.warning("Refresh failed: invalid token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

    # Check expiration
    if datetime.utcnow() > token_data["expires_at"]:
        log.warning("Refresh failed: token expired")
        del refresh_tokens_db[request.refresh_token]
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has expired"
        )

    user_id = token_data["user_id"]

    # Find user
    user = None
    for u in users_db.values():
        if u.id == user_id:
            user = u
            break

    if not user:
        log.error("Refresh failed: user not found", extra={"user_id": str(user_id)})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    # Generate new access token
    access_token = create_access_token(user.id, user.role)

    log.info("Token refreshed successfully", extra={"user_id": str(user.id)})

    return TokenResponse(
        access_token=access_token,
        refresh_token=request.refresh_token,  # Reuse same refresh token
        expires_in=config.jwt_access_token_expire_minutes * 60
    )


@app.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """
    Get current user information.

    Requires authentication.
    """
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        username=current_user.username,
        role=current_user.role,
        created_at=current_user.created_at,
        last_login=current_user.last_login
    )


@app.post("/logout")
async def logout(
    request: RefreshRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Logout and invalidate refresh token.

    Requires authentication.
    """
    log = get_contextual_logger(
        __name__,
        action="logout",
        user_id=str(current_user.id)
    )

    # Invalidate refresh token
    if request.refresh_token in refresh_tokens_db:
        del refresh_tokens_db[request.refresh_token]
        log.info("User logged out successfully")
    else:
        log.warning("Logout called with invalid refresh token")

    return {"message": "Logged out successfully"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "auth-service",
        "version": config.service_version,
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize service on startup."""
    logger.info(
        "Auth service starting",
        extra={
            "version": config.service_version,
            "environment": config.environment
        }
    )

    # In production, connect to database and Redis here


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("Auth service shutting down")

    # In production, close database and Redis connections here


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower()
    )
