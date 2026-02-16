#!/usr/bin/env python3
"""
Environment Configuration Validator
Validates that all required environment variables are set and properly configured.
"""

import os
import sys
from typing import Dict, List, Tuple


class EnvironmentValidator:
    """Validates environment configuration."""
    
    REQUIRED_VARS = {
        # Database
        "DB_HOST": "Database host",
        "DB_PORT": "Database port",
        "DB_DATABASE": "Database name",
        "DB_USERNAME": "Database username",
        "DB_PASSWORD": "Database password",
        
        # Redis
        "REDIS_HOST": "Redis host",
        "REDIS_PORT": "Redis port",
        
        # RabbitMQ
        "RABBITMQ_HOST": "RabbitMQ host",
        "RABBITMQ_PORT": "RabbitMQ port",
        
        # Security
        "SECURITY_SECRET_KEY": "Security secret key for JWT",
        "JWT_SECRET": "JWT secret key",
        
        # API
        "API_HOST": "API server host",
        "API_PORT": "API server port",
    }
    
    RECOMMENDED_VARS = {
        # LLM Providers
        "LLM_OPENAI_API_KEY": "OpenAI API key",
        "LLM_ANTHROPIC_API_KEY": "Anthropic API key",
        
        # Observability
        "OBSERVABILITY_ENABLE_TRACING": "Enable distributed tracing",
        "OBSERVABILITY_ENABLE_METRICS": "Enable metrics collection",
        
        # Celery
        "CELERY_BROKER_URL": "Celery broker URL",
        "CELERY_RESULT_BACKEND": "Celery result backend",
    }
    
    def __init__(self, environment: str = "development"):
        self.environment = environment
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_required(self) -> bool:
        """Validate all required environment variables are set."""
        all_present = True
        
        for var, description in self.REQUIRED_VARS.items():
            value = os.getenv(var)
            if not value:
                self.errors.append(f"Missing required variable: {var} ({description})")
                all_present = False
            elif value in ["changeme", "your-key-here", "generate_random"]:
                self.errors.append(f"Variable {var} has placeholder value - must be set")
                all_present = False
        
        return all_present
    
    def validate_recommended(self) -> None:
        """Check for recommended environment variables."""
        for var, description in self.RECOMMENDED_VARS.items():
            value = os.getenv(var)
            if not value:
                self.warnings.append(f"Recommended variable not set: {var} ({description})")
    
    def validate_secrets(self) -> None:
        """Validate that secrets are properly configured."""
        secret_key = os.getenv("SECURITY_SECRET_KEY", "")
        jwt_secret = os.getenv("JWT_SECRET", "")
        
        if len(secret_key) < 32:
            self.errors.append("SECURITY_SECRET_KEY must be at least 32 characters")
        
        if len(jwt_secret) < 32:
            self.errors.append("JWT_SECRET must be at least 32 characters")
        
        if secret_key == jwt_secret and secret_key:
            self.warnings.append("SECURITY_SECRET_KEY and JWT_SECRET should be different")
    
    def validate_ports(self) -> None:
        """Validate port numbers are valid."""
        port_vars = ["DB_PORT", "REDIS_PORT", "RABBITMQ_PORT", "API_PORT"]
        
        for var in port_vars:
            value = os.getenv(var)
            if value:
                try:
                    port = int(value)
                    if not (1 <= port <= 65535):
                        self.errors.append(f"{var} must be between 1 and 65535, got {port}")
                except ValueError:
                    self.errors.append(f"{var} must be a valid integer, got {value}")
    
    def validate_production_settings(self) -> None:
        """Validate production-specific settings."""
        if self.environment == "production":
            debug = os.getenv("API_DEBUG", "false").lower()
            if debug == "true":
                self.errors.append("API_DEBUG must be 'false' in production")
            
            reload = os.getenv("API_RELOAD", "false").lower()
            if reload == "true":
                self.warnings.append("API_RELOAD should be 'false' in production")
    
    def run_validation(self) -> Tuple[bool, Dict[str, List[str]]]:
        """Run all validations and return results."""
        self.validate_required()
        self.validate_recommended()
        self.validate_secrets()
        self.validate_ports()
        self.validate_production_settings()
        
        is_valid = len(self.errors) == 0
        results = {
            "errors": self.errors,
            "warnings": self.warnings
        }
        
        return is_valid, results


def main():
    """Main validation function."""
    environment = os.getenv("ENVIRONMENT", "development")
    
    print(f"🔍 Validating environment configuration for: {environment}")
    print("=" * 70)
    
    validator = EnvironmentValidator(environment)
    is_valid, results = validator.run_validation()
    
    if results["warnings"]:
        print("\n⚠️  Warnings:")
        for warning in results["warnings"]:
            print(f"  - {warning}")
    
    if results["errors"]:
        print("\n❌ Errors:")
        for error in results["errors"]:
            print(f"  - {error}")
        print("\n" + "=" * 70)
        print("❌ Environment validation FAILED")
        sys.exit(1)
    else:
        print("\n" + "=" * 70)
        print("✅ Environment validation PASSED")
        if results["warnings"]:
            print(f"   ({len(results['warnings'])} warnings)")
        sys.exit(0)


if __name__ == "__main__":
    main()
