# Shared Libraries

Common utilities and models shared across all CognitionOS services.

## Structure

- **config/**: Configuration management and environment handling
- **logger/**: Structured logging with tracing support
- **models/**: Shared data models and type definitions
- **utils/**: Common utility functions
- **middleware/**: Reusable middleware for web services

## Usage

Each service imports only what it needs:

```python
from shared.libs.logger import get_logger
from shared.libs.models import User, Task
from shared.libs.utils import generate_id
```

## Design Principles

1. **No Business Logic**: Only pure utilities and data structures
2. **No External Dependencies**: Minimal dependencies to avoid version conflicts
3. **Well-Tested**: 100% coverage on critical utilities
4. **Backward Compatible**: Semantic versioning for breaking changes
