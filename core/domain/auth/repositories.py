"""
Authentication Repository Interfaces

Repository interfaces for user persistence.
"""

from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID

from core.domain.auth.entities import User


class UserRepository(ABC):
    """Repository interface for User persistence"""
    
    @abstractmethod
    async def find_by_id(self, user_id: UUID) -> Optional[User]:
        """
        Find user by ID.
        
        Args:
            user_id: User ID to find
            
        Returns:
            User if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def find_by_email(self, email: str) -> Optional[User]:
        """
        Find user by email.
        
        Args:
            email: Email address to find
            
        Returns:
            User if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def create(self, user: User) -> User:
        """
        Create a new user.
        
        Args:
            user: User entity to create
            
        Returns:
            Created user
        """
        pass
    
    @abstractmethod
    async def update(self, user: User) -> User:
        """
        Update an existing user.
        
        Args:
            user: User entity to update
            
        Returns:
            Updated user
        """
        pass
    
    @abstractmethod
    async def delete(self, user_id: UUID) -> bool:
        """
        Delete a user.
        
        Args:
            user_id: ID of user to delete
            
        Returns:
            True if deleted, False if not found
        """
        pass
    
    @abstractmethod
    async def exists_by_email(self, email: str) -> bool:
        """
        Check if a user exists with the given email.
        
        Args:
            email: Email to check
            
        Returns:
            True if exists, False otherwise
        """
        pass
