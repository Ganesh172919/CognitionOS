"""
Core Package - CognitionOS V3

Clean Architecture implementation with strict dependency rules.

Layers:
- domain/: Core business logic (ZERO external dependencies)
- application/: Use cases and application services
- (infrastructure/ and interface/ live in services/)
"""

__version__ = "3.0.0"
