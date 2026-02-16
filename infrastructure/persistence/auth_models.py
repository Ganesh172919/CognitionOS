"""
Authentication SQLAlchemy Models

Database models for authentication entities.
"""

from datetime import datetime
from sqlalchemy import Column, String, Boolean, DateTime, Integer, ARRAY, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
import uuid

from infrastructure.persistence.database import Base
from core.domain.auth.entities import UserStatus


class UserModel(Base):
    """SQLAlchemy model for User entity"""
    
    __tablename__ = "users"
    
    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), nullable=False, unique=True, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    roles = Column(ARRAY(String), nullable=False, default=["user"])
    status = Column(
        SQLEnum(UserStatus, name="user_status_enum", create_type=True),
        nullable=False,
        default=UserStatus.ACTIVE,
        index=True
    )
    email_verified = Column(Boolean, nullable=False, default=False)
    failed_login_attempts = Column(Integer, nullable=False, default=0)
    locked_until = Column(DateTime(timezone=True), nullable=True)
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<UserModel(user_id={self.user_id}, email={self.email}, status={self.status})>"
