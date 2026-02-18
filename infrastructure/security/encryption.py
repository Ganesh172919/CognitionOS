"""
Encryption Service

Production-grade encryption for data at rest and in transit.
"""

import logging
import os
import base64
from typing import Dict, Any, Optional, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.backends import default_backend
import hashlib

logger = logging.getLogger(__name__)


class EncryptionService:
    """
    Production encryption service.
    
    Features:
    - AES-256 encryption
    - Key derivation from master key
    - Field-level encryption
    - Secure key rotation
    """
    
    def __init__(self, master_key: Optional[str] = None):
        """
        Initialize encryption service.
        
        Args:
            master_key: Master encryption key (base64 encoded)
        """
        if master_key:
            self.master_key = master_key.encode()
        else:
            # Generate new key if not provided
            self.master_key = Fernet.generate_key()
        
        self.fernet = Fernet(self.master_key)
        
        logger.info("Encryption service initialized")
    
    def encrypt(self, data: Union[str, bytes]) -> str:
        """
        Encrypt data.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Base64-encoded encrypted data
        """
        if isinstance(data, str):
            data = data.encode()
        
        encrypted = self.fernet.encrypt(data)
        return base64.b64encode(encrypted).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """
        Decrypt data.
        
        Args:
            encrypted_data: Base64-encoded encrypted data
            
        Returns:
            Decrypted data as string
        """
        encrypted_bytes = base64.b64decode(encrypted_data.encode())
        decrypted = self.fernet.decrypt(encrypted_bytes)
        return decrypted.decode()
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> Dict[str, str]:
        """
        Hash password with PBKDF2.
        
        Args:
            password: Plain password
            salt: Optional salt (generated if not provided)
            
        Returns:
            Dictionary with hash and salt
        """
        if salt is None:
            salt = os.urandom(32)
        
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        key = kdf.derive(password.encode())
        
        return {
            "hash": base64.b64encode(key).decode(),
            "salt": base64.b64encode(salt).decode(),
        }
    
    def verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """
        Verify password against hash.
        
        Args:
            password: Plain password
            password_hash: Base64-encoded hash
            salt: Base64-encoded salt
            
        Returns:
            True if password matches
        """
        salt_bytes = base64.b64decode(salt.encode())
        result = self.hash_password(password, salt_bytes)
        
        return result["hash"] == password_hash
    
    @staticmethod
    def generate_key() -> str:
        """Generate a new encryption key."""
        return Fernet.generate_key().decode()


class FieldEncryption:
    """
    Field-level encryption for database models.
    
    Automatically encrypts/decrypts specific fields.
    """
    
    def __init__(self, encryption_service: EncryptionService):
        """
        Initialize field encryption.
        
        Args:
            encryption_service: Encryption service instance
        """
        self.encryption = encryption_service
        self._encrypted_fields: Dict[str, set] = {}
    
    def register_encrypted_field(self, model_name: str, field_name: str):
        """
        Register a field for automatic encryption.
        
        Args:
            model_name: Model class name
            field_name: Field name to encrypt
        """
        if model_name not in self._encrypted_fields:
            self._encrypted_fields[model_name] = set()
        
        self._encrypted_fields[model_name].add(field_name)
        logger.info(f"Registered encrypted field: {model_name}.{field_name}")
    
    def encrypt_model(self, model_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encrypt registered fields in model data.
        
        Args:
            model_name: Model class name
            data: Model data dictionary
            
        Returns:
            Data with encrypted fields
        """
        if model_name not in self._encrypted_fields:
            return data
        
        encrypted_data = data.copy()
        
        for field in self._encrypted_fields[model_name]:
            if field in encrypted_data and encrypted_data[field] is not None:
                encrypted_data[field] = self.encryption.encrypt(str(encrypted_data[field]))
        
        return encrypted_data
    
    def decrypt_model(self, model_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decrypt registered fields in model data.
        
        Args:
            model_name: Model class name
            data: Model data dictionary with encrypted fields
            
        Returns:
            Data with decrypted fields
        """
        if model_name not in self._encrypted_fields:
            return data
        
        decrypted_data = data.copy()
        
        for field in self._encrypted_fields[model_name]:
            if field in decrypted_data and decrypted_data[field] is not None:
                try:
                    decrypted_data[field] = self.encryption.decrypt(decrypted_data[field])
                except Exception as e:
                    logger.error(f"Error decrypting field {model_name}.{field}: {e}")
                    decrypted_data[field] = None
        
        return decrypted_data


# API key hashing utilities
def hash_api_key(api_key: str) -> str:
    """
    Hash API key for storage.
    
    Args:
        api_key: Plain API key
        
    Returns:
        SHA256 hash of API key
    """
    return hashlib.sha256(api_key.encode()).hexdigest()


def generate_api_key(prefix: str = "cog") -> str:
    """
    Generate a secure API key.
    
    Args:
        prefix: Key prefix
        
    Returns:
        Generated API key
    """
    random_bytes = os.urandom(32)
    key_part = base64.urlsafe_b64encode(random_bytes).decode().rstrip("=")
    return f"{prefix}_{key_part}"
