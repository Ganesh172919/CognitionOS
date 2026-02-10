# AI Runtime

Multi-model LLM routing and execution service.

## Purpose

The AI Runtime manages all interactions with Large Language Models. It routes requests to appropriate models based on role (Planner, Executor, Critic), manages costs, handles failover, and ensures quality.

## Features

- **Multi-Provider Support**: OpenAI, Anthropic, local models
- **Role-Based Routing**: Different models for different agent roles
- **Cost Optimization**: Select cheapest model that meets requirements
- **Fallback Strategy**: Automatic retry with alternative models
- **Prompt Versioning**: A/B testing and gradual rollout
- **Response Validation**: Check for hallucinations and errors
- **Caching**: Cache identical requests to save costs

## Model Routing Strategy

```
Request → AI Runtime → Model Router → [GPT-4 | Claude | Local]
                            ↓
                     Cost Optimizer
                            ↓
                     Response Validator
                            ↓
                          Cache
```

## Model Selection

- **Planner**: GPT-4 or Claude Opus (high reasoning)
- **Executor**: GPT-3.5-turbo (fast, cost-effective)
- **Critic**: GPT-4 (high quality validation)
- **Summarizer**: GPT-3.5-turbo (fast summarization)

## Environment Variables

```
AI_RUNTIME_PORT=8005
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DEFAULT_PLANNER_MODEL=gpt-4
DEFAULT_EXECUTOR_MODEL=gpt-3.5-turbo
CACHE_ENABLED=true
MAX_RETRIES=3
```

## Tech Stack

- Python 3.11+
- LangChain for model abstraction
- Redis for caching
- Prompt templates in versioned files
