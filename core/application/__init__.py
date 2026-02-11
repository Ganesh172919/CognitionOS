"""
Application Layer Package

Application layer contains use cases that orchestrate domain entities.

Following Clean Architecture:
- Depends ONLY on domain layer
- Contains application-specific business rules
- Coordinates domain entities via use cases
- Defines DTOs (Data Transfer Objects) for input/output
- Publishes domain events
"""

from . import workflow
from . import agent

__all__ = [
    "workflow",
    "agent",
]
