#!/usr/bin/env python3
"""
Environment Configuration Validator
Ensures all required environment variables are properly configured for localhost operation.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_success(msg: str):
    print(f"{Colors.GREEN}✓{Colors.END} {msg}")

def print_error(msg: str):
    print(f"{Colors.RED}✗{Colors.END} {msg}")

def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠{Colors.END} {msg}")

def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ{Colors.END} {msg}")

# Define required environment variables with their validation rules
REQUIRED_VARS = {
    # Database
    "DB_HOST": {"default": "localhost", "example": "localhost", "required": True},
    "DB_PORT": {"default": "5432", "example": "5432", "required": True},
    "DB_DATABASE": {"default": "cognitionos_dev", "example": "cognitionos_dev", "required": True},
    "DB_USERNAME": {"default": "cognition_dev", "example": "cognition_dev", "required": True},
    "DB_PASSWORD": {"default": "dev_password_local", "example": "dev_password_local", "required": True},

    # Redis
    "REDIS_HOST": {"default": "localhost", "example": "localhost", "required": True},
    "REDIS_PORT": {"default": "6379", "example": "6379", "required": True},

    # RabbitMQ
    "RABBITMQ_HOST": {"default": "localhost", "example": "localhost", "required": True},
    "RABBITMQ_PORT": {"default": "5672", "example": "5672", "required": True},
    "RABBITMQ_USERNAME": {"default": "guest", "example": "guest", "required": True},
    "RABBITMQ_PASSWORD": {"default": "guest", "example": "guest", "required": True},

    # API Configuration
    "API_HOST": {"default": "0.0.0.0", "example": "0.0.0.0", "required": True},
    "API_PORT": {"default": "8100", "example": "8100", "required": True},

    # Security
    "SECURITY_SECRET_KEY": {"default": None, "example": "your-secret-key-here-change-in-production", "required": True},
    "JWT_SECRET": {"default": None, "example": "your-jwt-secret-here-change-in-production", "required": True},

    # Environment
    "ENVIRONMENT": {"default": "development", "example": "development", "required": True},

    # LLM Providers (optional for local development)
    "LLM_OPENAI_API_KEY": {"default": None, "example": "sk-...", "required": False},
    "LLM_ANTHROPIC_API_KEY": {"default": None, "example": "sk-ant-...", "required": False},
}

def load_env_file(env_file: str = ".env.localhost") -> Dict[str, str]:
    """Load environment variables from file"""
    env_vars = {}
    env_path = Path(env_file)

    if not env_path.exists():
        return env_vars

    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip()

    return env_vars

def validate_environment() -> Tuple[int, int]:
    """Validate environment configuration"""
    print("=" * 60)
    print("CognitionOS Environment Validator")
    print("=" * 60)
    print()

    # Load environment file
    env_vars = load_env_file()

    if not env_vars:
        print_error(".env.localhost file not found or empty")
        print_info("Create one by copying: cp .env.example .env.localhost")
        return 0, 1

    print_success(f"Loaded {len(env_vars)} variables from .env.localhost")
    print()

    passed = 0
    failed = 0
    warnings = 0

    # Validate each required variable
    print("Validating Required Variables:")
    print("-" * 60)

    for var_name, rules in REQUIRED_VARS.items():
        value = env_vars.get(var_name)
        required = rules.get("required", False)
        default = rules.get("default")
        example = rules.get("example")

        if value is None or value == "":
            if required:
                print_error(f"{var_name}: Missing (required)")
                if example:
                    print(f"         Example: {var_name}={example}")
                failed += 1
            else:
                print_warning(f"{var_name}: Not set (optional)")
                warnings += 1
        else:
            # Check for placeholder values
            placeholders = ["changeme", "replace", "your-", "test-key", "example"]
            is_placeholder = any(placeholder in value.lower() for placeholder in placeholders)

            if is_placeholder and var_name in ["LLM_OPENAI_API_KEY", "LLM_ANTHROPIC_API_KEY"]:
                print_warning(f"{var_name}: Using placeholder (API features will not work)")
                warnings += 1
            elif is_placeholder and var_name in ["SECURITY_SECRET_KEY", "JWT_SECRET"]:
                if "development" in env_vars.get("ENVIRONMENT", "").lower():
                    print_success(f"{var_name}: Set (using development placeholder)")
                    passed += 1
                else:
                    print_error(f"{var_name}: Insecure placeholder in non-development environment")
                    failed += 1
            else:
                print_success(f"{var_name}: Configured")
                passed += 1

    print()
    print("=" * 60)
    print("Validation Summary")
    print("=" * 60)
    print(f"{Colors.GREEN}Passed:{Colors.END}   {passed}")
    print(f"{Colors.RED}Failed:{Colors.END}   {failed}")
    print(f"{Colors.YELLOW}Warnings:{Colors.END} {warnings}")
    print()

    if failed > 0:
        print_error("Environment validation failed")
        print_info("Please fix the errors above before starting the system")
        return passed, failed
    else:
        print_success("Environment validation passed")
        if warnings > 0:
            print_warning(f"There are {warnings} warnings - some features may not work")
        print()
        print("You can now start the system with: docker-compose up -d")
        return passed, 0

def generate_env_template():
    """Generate a template .env file"""
    print("Generating .env.localhost template...")

    template_lines = [
        "# CognitionOS - Localhost Development Environment",
        "# Auto-generated configuration template",
        "#",
        "",
    ]

    sections = {
        "Environment": ["ENVIRONMENT"],
        "Database (PostgreSQL)": ["DB_HOST", "DB_PORT", "DB_DATABASE", "DB_USERNAME", "DB_PASSWORD"],
        "Redis Cache": ["REDIS_HOST", "REDIS_PORT"],
        "RabbitMQ Message Broker": ["RABBITMQ_HOST", "RABBITMQ_PORT", "RABBITMQ_USERNAME", "RABBITMQ_PASSWORD"],
        "API Server": ["API_HOST", "API_PORT"],
        "Security": ["SECURITY_SECRET_KEY", "JWT_SECRET"],
        "LLM Providers (Optional)": ["LLM_OPENAI_API_KEY", "LLM_ANTHROPIC_API_KEY"],
    }

    for section_name, var_names in sections.items():
        template_lines.append(f"# {section_name}")
        template_lines.append("# " + "=" * 40)

        for var_name in var_names:
            if var_name in REQUIRED_VARS:
                rules = REQUIRED_VARS[var_name]
                example = rules.get("example") or rules.get("default") or ""
                required = " (required)" if rules.get("required") else " (optional)"
                template_lines.append(f"{var_name}={example}{required}")

        template_lines.append("")

    output_file = ".env.localhost.template"
    with open(output_file, 'w') as f:
        f.write('\n'.join(template_lines))

    print_success(f"Template generated: {output_file}")
    print_info(f"Copy it to .env.localhost and fill in your values")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        generate_env_template()
    else:
        passed, failed = validate_environment()
        sys.exit(0 if failed == 0 else 1)
