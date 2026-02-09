# Versioned Prompts

This directory contains versioned prompts for different agent roles in CognitionOS.

## Structure

- `planner/` - Prompts for Planner agents (goal decomposition)
- `executor/` - Prompts for Executor agents (task execution)
- `critic/` - Prompts for Critic agents (quality validation)
- `summarizer/` - Prompts for Summarizer agents (context compression)

## Versioning

Each prompt file is versioned (e.g., `v1.md`, `v2.md`) to enable:
- A/B testing different prompts
- Gradual rollout of prompt changes
- Rollback to previous versions if needed
- Performance comparison across versions

## Usage

```python
from prompts import load_prompt

# Load specific version
prompt = load_prompt("planner", version="v1")

# Load latest version
prompt = load_prompt("planner", version="latest")
```

## Format

Each prompt file contains:
1. System instructions
2. Role definition
3. Output format requirements
4. Examples (few-shot learning)
5. Constraints and safety guidelines
