"""
Infrastructure Package

Infrastructure layer provides concrete implementations of domain repository interfaces.

Components:
- persistence/: Database implementations (PostgreSQL, Redis)
- events/: Event bus implementations
- config/: Configuration management

This layer depends on domain layer interfaces but provides the concrete implementations.
"""

__version__ = "3.0.0"
