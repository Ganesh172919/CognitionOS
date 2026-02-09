# CognitionOS

**A Production-Grade, Backend-Heavy AI Operating System**

CognitionOS is not a chatbot. It's a thinking and execution platform that autonomously decomposes high-level goals into executable task graphs, orchestrates multi-agent workflows, maintains long-term memory, and explains its reasoning.

## 🎯 What is CognitionOS?

CognitionOS transforms natural language goals into autonomous execution through:

- **Goal Decomposition**: Breaks complex objectives into executable DAG workflows
- **Multi-Agent Orchestration**: Coordinates specialized AI agents (Planner, Executor, Critic)
- **Long-Term Memory**: Semantic memory with vector search, not just prompt stuffing
- **Tool Execution**: Sandboxed code execution, API calls, and file operations
- **Explainability**: Every decision is logged, traced, and explainable

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                                │
│               Web UI │ Mobile App │ API Clients                     │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│                       API GATEWAY (Go/Node.js)                     │
│  • Request Routing    • Rate Limiting    • WebSocket Support       │
│  • Authentication     • Circuit Breaker  • Request Tracing         │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
┌──────────────┐       ┌──────────────────┐      ┌──────────────┐
│Auth Service  │       │ Task Planner     │      │Agent Orch.   │
│ • JWT Auth   │       │ • DAG Generation │      │• Lifecycle   │
│ • RBAC       │       │ • Re-planning    │      │• Assignment  │
│ • Sessions   │       │ • Validation     │      │• Monitoring  │
└──────────────┘       └──────────────────┘      └──────────────┘
                                │                        │
                                └────────┬───────────────┘
                                         ▼
        ┌────────────────────────────────────────────────────┐
        │              AI Runtime (Python + LangChain)       │
        │  • Model Router  • Planner LLM  • Cost Tracking    │
        │  • Executor LLM  • Critic LLM   • Fallback Logic   │
        └────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│Tool Runner   │       │Memory Service│       │Audit Log     │
│• Sandboxing  │       │• Vector DB   │       │• Immutable   │
│• Permissions │       │• Embeddings  │       │• Queryable   │
│• Timeouts    │       │• Retrieval   │       │• Compliance  │
└──────────────┘       └──────────────┘       └──────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- OpenAI or Anthropic API key (for AI features)

### Run Locally

```bash
# Clone repository
git clone https://github.com/Ganesh172919/CognitionOS.git
cd CognitionOS

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Start all services
docker-compose up -d

# Check health
curl http://localhost:8000/health
```

### Test the System

```bash
# Register a user
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","username":"testuser","password":"password123"}'

# Login
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"password123"}'

# Create a task plan
curl -X POST http://localhost:8000/tasks/plan \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <access_token>" \
  -d '{"user_id":"<user_id>","goal":"Build a web application with user authentication"}'
```

## 📚 Documentation

- **[Architecture](docs/architecture.md)**: System design and component interaction
- **[Agent Model](docs/agent_model.md)**: Agent types, lifecycle, and orchestration
- **[Memory Model](docs/memory_model.md)**: Multi-layer memory architecture
- **[Security](docs/security.md)**: Threat model and defense strategies
- **[Deployment](docs/deployment.md)**: Production deployment guide

## 🔑 Key Features

### 1. Task Planning with DAG

Converts goals into dependency graphs:
```
Goal: "Deploy a web app"
  ↓
Design DB → Create Schema → Build API → Build Frontend → Tests → Deploy
                              ↓            ↓               ↑
                              └────────────┴───────────────┘
```

### 2. Multi-Agent System

- **Planner Agent**: Breaks down goals (GPT-4)
- **Executor Agent**: Runs code and tools (GPT-3.5-turbo)
- **Critic Agent**: Validates outputs (GPT-4)
- **Summarizer Agent**: Compresses context (GPT-3.5-turbo)

### 3. Long-Term Memory

Not prompt stuffing - actual semantic memory:
```python
# Store
memory_service.store(
    user_id=user_id,
    content="User prefers Python over JavaScript",
    memory_type=MemoryType.PREFERENCE
)

# Retrieve
relevant_memories = memory_service.retrieve(
    user_id=user_id,
    query="programming languages",
    k=5
)
```

### 4. Tool Execution

Sandboxed, permission-controlled tool execution:
```python
result = tool_runner.execute(
    tool="execute_python",
    parameters={"code": "import pandas as pd; ..."},
    permissions=["code_execution"],
    timeout=30
)
```

### 5. Observability

Every action is traced and explainable:
```json
{
  "trace_id": "abc-123",
  "agent_id": "agent-456",
  "action": "tool_execution",
  "tool": "http_request",
  "cost_usd": 0.002,
  "duration_ms": 1234,
  "success": true
}
```

## 🏛️ Services

| Service | Port | Purpose |
|---------|------|---------|
| API Gateway | 8000 | Entry point, routing, rate limiting |
| Auth Service | 8001 | JWT authentication, RBAC |
| Task Planner | 8002 | Goal decomposition, DAG generation |
| Agent Orchestrator | 8003 | Agent lifecycle management |
| Memory Service | 8004 | Long-term memory storage |
| AI Runtime | 8005 | LLM routing and execution |
| Tool Runner | 8006 | Sandboxed tool execution |
| Audit Log | 8007 | Immutable action logging |

## 🛠️ Technology Stack

### Backend
- **API Layer**: FastAPI (Python), Go (API Gateway)
- **Task Planning**: NetworkX for DAG operations
- **Agent Runtime**: LangChain for LLM orchestration
- **Memory**: PostgreSQL + pgvector for semantic search
- **Caching**: Redis for sessions and rate limiting
- **Message Queue**: RabbitMQ for async tasks

### Infrastructure
- **Containerization**: Docker
- **Orchestration**: Kubernetes
- **Monitoring**: Prometheus + Grafana
- **Tracing**: OpenTelemetry
- **CI/CD**: GitHub Actions

## 📊 Design Principles

1. **Backend First**: Frontend visualizes backend intelligence
2. **Microservices**: Independent scaling and deployment
3. **Observability by Default**: Every decision is logged
4. **User Isolation**: Multi-tenant with strict data separation
5. **Failure Resilience**: Circuit breakers, retries, graceful degradation

## 🔒 Security

- JWT authentication with short expiration
- Row-level security in PostgreSQL
- Sandboxed tool execution (Docker)
- Prompt injection detection
- Rate limiting and budget caps
- Comprehensive audit logging

See [Security Documentation](docs/security.md) for details.

## 📈 Scalability

### Horizontal Scaling
- API Gateway: 3+ instances
- Agent Orchestrator: Autoscale based on queue depth
- AI Runtime: GPU worker pool

### Vertical Scaling
- AI Runtime: GPU instances
- Vector DB: High-memory instances

### Database Scaling
- Sharding by user_id
- Read replicas for queries
- Connection pooling

## 🧪 Testing

```bash
# Run unit tests
pytest services/*/tests/

# Run integration tests
pytest tests/integration/

# Load testing
k6 run tests/load/basic-scenario.js
```

## 🌟 Example Use Case

**User Goal**: "Analyze my sales data and create a dashboard"

**System Execution**:
1. Task Planner creates DAG:
   - Load CSV data
   - Clean and validate
   - Perform statistical analysis
   - Generate visualizations
   - Create interactive dashboard

2. Agent Orchestrator spawns:
   - Executor Agent (loads and cleans data)
   - Executor Agent (runs analysis)
   - Executor Agent (generates charts)
   - Summarizer Agent (explains findings)

3. Tool Runner executes:
   - Python code for data processing
   - Matplotlib for visualizations
   - HTML/JS for dashboard

4. Memory Service stores:
   - Data schema understanding
   - User preferences (chart types)
   - Analysis insights

5. User receives:
   - Interactive dashboard
   - Explanation of findings
   - Suggestions for next steps

## 🗺️ Roadmap

### Phase 1 (Current) ✅
- Core services implemented
- Basic agent orchestration
- Task planning with DAG
- Authentication and authorization

### Phase 2 (Next)
- Production database integration
- Real LLM integration (OpenAI, Anthropic)
- Frontend dashboard
- Memory service with vector DB

### Phase 3 (Future)
- Agent learning from feedback
- Multi-agent collaboration
- Custom agent marketplace
- Mobile app

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Built with FastAPI, LangChain, and NetworkX
- Inspired by AutoGPT, BabyAGI, and CrewAI
- Architecture patterns from microservices best practices

## 📞 Contact

- Issues: [GitHub Issues](https://github.com/Ganesh172919/CognitionOS/issues)
- Discussions: [GitHub Discussions](https://github.com/Ganesh172919/CognitionOS/discussions)

---

**Built with 🧠 for autonomous AI execution**
