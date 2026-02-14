"""
Authentication Module
"""

from .jwt import (
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    verify_token,
    extract_user_id,
    extract_user_roles,
)

from .dependencies import (
    CurrentUser,
    get_current_user,
    get_current_active_user,
    get_current_user_optional,
    require_role,
    require_any_role,
    require_all_roles,
)

__all__ = [
    # JWT functions
    "verify_password",
    "get_password_hash",
    "create_access_token",
    "create_refresh_token",
    "verify_token",
    "extract_user_id",
    "extract_user_roles",
    # Dependencies
    "CurrentUser",
    "get_current_user",
    "get_current_active_user",
    "get_current_user_optional",
    "require_role",
    "require_any_role",
    "require_all_roles",
]
