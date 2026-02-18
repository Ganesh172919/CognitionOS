#!/usr/bin/env python3
"""
CognitionOS Environment Validation Script

Validates all required environment variables and configurations
before starting the application.
"""

import os
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Try to load .env from current directory
    env_path = Path.cwd() / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Try .env.localhost
        env_path = Path.cwd() / '.env.localhost'
        if env_path.exists():
            load_dotenv(env_path)
except ImportError:
    print("Warning: python-dotenv not installed, environment variables may not be loaded")
    pass


class Severity(Enum):
    """Validation issue severity"""
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class ValidationIssue:
    """Represents a validation issue"""
    severity: Severity
    component: str
    message: str
    fix_hint: Optional[str] = None


class EnvironmentValidator:
    """Validates environment configuration"""
    
    def __init__(self):
        self.issues: List[ValidationIssue] = []
    
    def add_issue(
        self,
        severity: Severity,
        component: str,
        message: str,
        fix_hint: Optional[str] = None
    ):
        """Add a validation issue"""
        self.issues.append(
            ValidationIssue(severity, component, message, fix_hint)
        )
    
    def check_required_var(
        self,
        var_name: str,
        component: str,
        description: str
    ) -> Optional[str]:
        """Check if a required environment variable is set"""
        value = os.getenv(var_name)
        if not value:
            self.add_issue(
                Severity.ERROR,
                component,
                f"{var_name} is not set",
                f"Set {var_name} in .env file. Purpose: {description}"
            )
            return None
        return value
    
    def check_optional_var(
        self,
        var_name: str,
        component: str,
        default: str,
        description: str
    ) -> str:
        """Check optional environment variable and warn if using default"""
        value = os.getenv(var_name)
        if not value:
            self.add_issue(
                Severity.WARNING,
                component,
                f"{var_name} not set, using default: {default}",
                f"Consider setting {var_name}. Purpose: {description}"
            )
            return default
        return value
    
    def validate_database_config(self):
        """Validate database configuration"""
        component = "Database"
        
        self.check_required_var("DB_HOST", component, "PostgreSQL host")
        self.check_required_var("DB_PORT", component, "PostgreSQL port")
        self.check_required_var("DB_DATABASE", component, "Database name")
        self.check_required_var("DB_USERNAME", component, "Database username")
        password = self.check_required_var("DB_PASSWORD", component, "Database password")
        
        # Check weak passwords in production
        env = os.getenv("ENVIRONMENT", "development")
        if password and env == "production" and (
            password == "changeme" or 
            password == "dev_password_local" or 
            len(password) < 12
        ):
            self.add_issue(
                Severity.ERROR,
                component,
                "Database password is weak or using default",
                "Use a strong password (12+ characters) in production"
            )
    
    def validate_redis_config(self):
        """Validate Redis configuration"""
        component = "Redis"
        
        self.check_required_var("REDIS_HOST", component, "Redis host")
        self.check_required_var("REDIS_PORT", component, "Redis port")
        self.check_optional_var("REDIS_DB", component, "0", "Redis database number")
        self.check_optional_var("REDIS_PASSWORD", component, "", "Redis password")
    
    def validate_rabbitmq_config(self):
        """Validate RabbitMQ configuration"""
        component = "RabbitMQ"
        
        self.check_required_var("RABBITMQ_HOST", component, "RabbitMQ host")
        self.check_required_var("RABBITMQ_PORT", component, "RabbitMQ port")
        self.check_required_var("RABBITMQ_USERNAME", component, "RabbitMQ username")
        password = self.check_required_var("RABBITMQ_PASSWORD", component, "RabbitMQ password")
        
        # Check weak credentials in production
        env = os.getenv("ENVIRONMENT", "development")
        if password and env == "production" and password == "guest":
            self.add_issue(
                Severity.ERROR,
                component,
                "RabbitMQ using default 'guest' password",
                "Change RabbitMQ credentials in production"
            )
    
    def validate_api_config(self):
        """Validate API configuration"""
        component = "API"
        
        self.check_optional_var("API_HOST", component, "0.0.0.0", "API bind address")
        self.check_optional_var("API_PORT", component, "8100", "API port")
        self.check_optional_var("API_LOG_LEVEL", component, "info", "Log level")
    
    def validate_llm_config(self):
        """Validate LLM provider configuration"""
        component = "LLM"
        
        openai_key = os.getenv("LLM_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("LLM_ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        
        if not openai_key and not anthropic_key:
            self.add_issue(
                Severity.WARNING,
                component,
                "No LLM API keys configured",
                "Set LLM_OPENAI_API_KEY or LLM_ANTHROPIC_API_KEY for AI features"
            )
        
        # Check for placeholder keys
        if openai_key and ("test" in openai_key or "replace" in openai_key.lower()):
            self.add_issue(
                Severity.WARNING,
                component,
                "OpenAI API key appears to be a placeholder",
                "Set a valid OpenAI API key for AI features"
            )
        
        if anthropic_key and ("test" in anthropic_key or "replace" in anthropic_key.lower()):
            self.add_issue(
                Severity.WARNING,
                component,
                "Anthropic API key appears to be a placeholder",
                "Set a valid Anthropic API key for AI features"
            )
    
    def validate_security_config(self):
        """Validate security configuration"""
        component = "Security"
        
        secret_key = self.check_required_var(
            "SECURITY_SECRET_KEY",
            component,
            "Application secret key"
        )
        jwt_secret = self.check_required_var(
            "JWT_SECRET",
            component,
            "JWT signing secret"
        )
        
        env = os.getenv("ENVIRONMENT", "development")
        
        # Check weak secrets in production
        if secret_key and env == "production":
            if (
                "localhost" in secret_key or
                "change_in_production" in secret_key or
                len(secret_key) < 32
            ):
                self.add_issue(
                    Severity.ERROR,
                    component,
                    "SECURITY_SECRET_KEY is weak or using default",
                    "Generate a strong secret: openssl rand -hex 32"
                )
        
        if jwt_secret and env == "production":
            if (
                "localhost" in jwt_secret or
                "change_in_production" in jwt_secret or
                len(jwt_secret) < 32
            ):
                self.add_issue(
                    Severity.ERROR,
                    component,
                    "JWT_SECRET is weak or using default",
                    "Generate a strong secret: openssl rand -hex 32"
                )
    
    def validate_environment(self):
        """Validate environment setting"""
        component = "Environment"
        
        env = os.getenv("ENVIRONMENT", "development")
        valid_envs = ["development", "staging", "production"]
        
        if env not in valid_envs:
            self.add_issue(
                Severity.WARNING,
                component,
                f"ENVIRONMENT='{env}' is not standard",
                f"Use one of: {', '.join(valid_envs)}"
            )
        
        if env == "production":
            # Production-specific checks
            debug = os.getenv("DEBUG", "false").lower()
            if debug == "true":
                self.add_issue(
                    Severity.ERROR,
                    component,
                    "DEBUG=true in production",
                    "Set DEBUG=false in production"
                )
    
    def run_validation(self) -> bool:
        """Run all validations. Returns True if no errors."""
        print("🔍 CognitionOS Environment Validation")
        print("=" * 70)
        
        # Check if .env file exists
        if not os.path.exists(".env") and not os.path.exists(".env.localhost"):
            self.add_issue(
                Severity.ERROR,
                "Configuration",
                "No .env file found",
                "Copy .env.localhost to .env or run ./scripts/setup-localhost.sh"
            )
        
        # Run all validations
        self.validate_environment()
        self.validate_database_config()
        self.validate_redis_config()
        self.validate_rabbitmq_config()
        self.validate_api_config()
        self.validate_llm_config()
        self.validate_security_config()
        
        # Report issues
        errors = [i for i in self.issues if i.severity == Severity.ERROR]
        warnings = [i for i in self.issues if i.severity == Severity.WARNING]
        infos = [i for i in self.issues if i.severity == Severity.INFO]
        
        # Print by severity
        if errors:
            print("\n❌ ERRORS (must be fixed):")
            for issue in errors:
                print(f"\n  [{issue.component}] {issue.message}")
                if issue.fix_hint:
                    print(f"  💡 {issue.fix_hint}")
        
        if warnings:
            print("\n⚠️  WARNINGS (should be addressed):")
            for issue in warnings:
                print(f"\n  [{issue.component}] {issue.message}")
                if issue.fix_hint:
                    print(f"  💡 {issue.fix_hint}")
        
        if infos:
            print("\nℹ️  INFO:")
            for issue in infos:
                print(f"\n  [{issue.component}] {issue.message}")
                if issue.fix_hint:
                    print(f"  💡 {issue.fix_hint}")
        
        print("\n" + "=" * 70)
        print(f"Validation Summary:")
        print(f"  ❌ Errors:   {len(errors)}")
        print(f"  ⚠️  Warnings: {len(warnings)}")
        print(f"  ℹ️  Info:     {len(infos)}")
        
        if errors:
            print("\n❌ Validation FAILED - Fix errors before starting")
            return False
        elif warnings:
            print("\n⚠️  Validation PASSED with warnings")
            return True
        else:
            print("\n✅ Validation PASSED - All checks successful")
            return True


def main():
    """Main entry point"""
    validator = EnvironmentValidator()
    success = validator.run_validation()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
