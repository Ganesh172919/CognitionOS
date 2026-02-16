"""
Authentication Domain Entities

Core domain entities for user authentication and authorization.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import UUID, uuid4


class UserStatus(str, Enum):
    """User account status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"


@dataclass
class User:
    """
    User entity representing an authenticated user in the system.
    
    Invariants:
    - Email must be unique
    - Password hash must be provided
    - At least one role must be assigned
    - Created timestamp is immutable
    """
    
    user_id: UUID
    email: str
    password_hash: str
    roles: List[str]
    status: UserStatus
    created_at: datetime
    updated_at: datetime
    full_name: Optional[str] = None
    last_login_at: Optional[datetime] = None
    email_verified: bool = False
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    
    @staticmethod
    def create(
        email: str,
        password_hash: str,
        full_name: Optional[str] = None,
        roles: Optional[List[str]] = None,
    ) -> "User":
        """
        Create a new user.
        
        Args:
            email: User's email address
            password_hash: Hashed password
            full_name: Optional full name
            roles: List of roles (defaults to ["user"])
            
        Returns:
            New User instance
        """
        now = datetime.utcnow()
        return User(
            user_id=uuid4(),
            email=email.lower(),  # Normalize email
            password_hash=password_hash,
            full_name=full_name,
            roles=roles or ["user"],
            status=UserStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            email_verified=False,
            failed_login_attempts=0,
        )
    
    def activate(self) -> None:
        """Activate the user account"""
        self.status = UserStatus.ACTIVE
        self.updated_at = datetime.utcnow()
    
    def deactivate(self) -> None:
        """Deactivate the user account"""
        self.status = UserStatus.INACTIVE
        self.updated_at = datetime.utcnow()
    
    def suspend(self, until: Optional[datetime] = None) -> None:
        """
        Suspend the user account.
        
        Args:
            until: Optional datetime until which user is suspended
        """
        self.status = UserStatus.SUSPENDED
        self.locked_until = until
        self.updated_at = datetime.utcnow()
    
    def verify_email(self) -> None:
        """Mark email as verified"""
        self.email_verified = True
        if self.status == UserStatus.PENDING_VERIFICATION:
            self.status = UserStatus.ACTIVE
        self.updated_at = datetime.utcnow()
    
    def record_login(self) -> None:
        """Record successful login"""
        self.last_login_at = datetime.utcnow()
        self.failed_login_attempts = 0
        self.updated_at = datetime.utcnow()
    
    def record_failed_login(self) -> None:
        """Record failed login attempt"""
        self.failed_login_attempts += 1
        self.updated_at = datetime.utcnow()
        
        # Lock account after 5 failed attempts
        if self.failed_login_attempts >= 5:
            from datetime import timedelta
            self.suspend(until=datetime.utcnow() + timedelta(minutes=30))
    
    def update_password(self, new_password_hash: str) -> None:
        """Update user password"""
        self.password_hash = new_password_hash
        self.updated_at = datetime.utcnow()
    
    def add_role(self, role: str) -> None:
        """Add a role to the user"""
        if role not in self.roles:
            self.roles.append(role)
            self.updated_at = datetime.utcnow()
    
    def remove_role(self, role: str) -> None:
        """Remove a role from the user"""
        if role in self.roles and len(self.roles) > 1:  # Keep at least one role
            self.roles.remove(role)
            self.updated_at = datetime.utcnow()
    
    def is_active(self) -> bool:
        """Check if user is active and not locked"""
        if self.status != UserStatus.ACTIVE:
            return False
        
        if self.locked_until and datetime.utcnow() < self.locked_until:
            return False
        
        # Auto-unlock if lock period expired
        if self.locked_until and datetime.utcnow() >= self.locked_until:
            self.locked_until = None
            self.failed_login_attempts = 0
        
        return True
    
    def has_role(self, role: str) -> bool:
        """Check if user has a specific role"""
        return role in self.roles
    
    def __post_init__(self):
        """Validate invariants after initialization"""
        if not self.email:
            raise ValueError("Email is required")
        if not self.password_hash:
            raise ValueError("Password hash is required")
        if not self.roles:
            raise ValueError("At least one role is required")
