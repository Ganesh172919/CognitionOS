"""
Authentication Middleware and Dependencies

Provides FastAPI dependencies for JWT authentication and authorization.
"""

import sys
import os
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Add core modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from services.api.src.auth.jwt import verify_token, extract_user_id, extract_user_roles


# HTTP Bearer token scheme
security = HTTPBearer()


class CurrentUser:
    """Current authenticated user"""
    
    def __init__(self, user_id: str, roles: list[str], email: Optional[str] = None):
        self.user_id = user_id
        self.roles = roles
        self.email = email
    
    def has_role(self, role: str) -> bool:
        """Check if user has a specific role"""
        return role in self.roles
    
    def has_any_role(self, *roles: str) -> bool:
        """Check if user has any of the specified roles"""
        return any(role in self.roles for role in roles)
    
    def has_all_roles(self, *roles: str) -> bool:
        """Check if user has all of the specified roles"""
        return all(role in self.roles for role in roles)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> CurrentUser:
    """
    Dependency to get the current authenticated user.
    
    Args:
        credentials: HTTP Bearer credentials
        
    Returns:
        Current user
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    token = credentials.credentials
    
    # Verify token
    payload = verify_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Extract user information
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    roles = payload.get("roles", [])
    email = payload.get("email")
    
    return CurrentUser(user_id=user_id, roles=roles, email=email)


async def get_current_active_user(
    current_user: CurrentUser = Depends(get_current_user)
) -> CurrentUser:
    """
    Dependency to get the current active user.
    
    This can be extended to check if user is active in the database.
    
    Args:
        current_user: Current user from token
        
    Returns:
        Current active user
    """
    # TODO: Check if user is active in database
    return current_user


def require_role(role: str):
    """
    Dependency factory to require a specific role.
    
    Args:
        role: Required role
        
    Returns:
        Dependency function
    """
    async def role_checker(
        current_user: CurrentUser = Depends(get_current_user)
    ) -> CurrentUser:
        if not current_user.has_role(role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User does not have required role: {role}"
            )
        return current_user
    
    return role_checker


def require_any_role(*roles: str):
    """
    Dependency factory to require any of the specified roles.
    
    Args:
        *roles: Required roles (user must have at least one)
        
    Returns:
        Dependency function
    """
    async def role_checker(
        current_user: CurrentUser = Depends(get_current_user)
    ) -> CurrentUser:
        if not current_user.has_any_role(*roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User does not have any of the required roles: {', '.join(roles)}"
            )
        return current_user
    
    return role_checker


def require_all_roles(*roles: str):
    """
    Dependency factory to require all of the specified roles.
    
    Args:
        *roles: Required roles (user must have all)
        
    Returns:
        Dependency function
    """
    async def role_checker(
        current_user: CurrentUser = Depends(get_current_user)
    ) -> CurrentUser:
        if not current_user.has_all_roles(*roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User does not have all required roles: {', '.join(roles)}"
            )
        return current_user
    
    return role_checker


# Optional authentication - doesn't fail if no token provided
async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> Optional[CurrentUser]:
    """
    Dependency to get the current user optionally.
    
    Returns None if no credentials provided.
    
    Args:
        credentials: Optional HTTP Bearer credentials
        
    Returns:
        Current user or None
    """
    if credentials is None:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None
