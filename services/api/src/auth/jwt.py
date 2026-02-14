"""
JWT Authentication Module

Provides JWT token generation, validation, and authentication middleware.
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext

# Add core modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from core.config import get_config


# Configuration
config = get_config()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = config.security.secret_key
ALGORITHM = config.security.algorithm
ACCESS_TOKEN_EXPIRE_MINUTES = config.security.access_token_expire_minutes
REFRESH_TOKEN_EXPIRE_DAYS = config.security.refresh_token_expire_days


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password
        
    Returns:
        True if password matches, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Hash a password.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    return pwd_context.hash(password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Data to encode in the token
        expires_delta: Optional custom expiration time
        
    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any]) -> str:
    """
    Create a JWT refresh token.
    
    Args:
        data: Data to encode in the token
        
    Returns:
        Encoded JWT refresh token
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode and validate a JWT token.
    
    Args:
        token: JWT token to decode
        
    Returns:
        Decoded token payload or None if invalid
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


def verify_token(token: str, token_type: str = "access") -> Optional[Dict[str, Any]]:
    """
    Verify a JWT token and check its type.
    
    Args:
        token: JWT token to verify
        token_type: Expected token type (access or refresh)
        
    Returns:
        Token payload if valid, None otherwise
    """
    payload = decode_token(token)
    
    if payload is None:
        return None
    
    # Check token type
    if payload.get("type") != token_type:
        return None
    
    # Check expiration
    exp = payload.get("exp")
    if exp is None or datetime.utcfromtimestamp(exp) < datetime.utcnow():
        return None
    
    return payload


def extract_user_id(token: str) -> Optional[str]:
    """
    Extract user ID from a JWT token.
    
    Args:
        token: JWT token
        
    Returns:
        User ID if token is valid, None otherwise
    """
    payload = verify_token(token)
    if payload:
        return payload.get("sub")
    return None


def extract_user_roles(token: str) -> list[str]:
    """
    Extract user roles from a JWT token.
    
    Args:
        token: JWT token
        
    Returns:
        List of user roles
    """
    payload = verify_token(token)
    if payload:
        return payload.get("roles", [])
    return []
