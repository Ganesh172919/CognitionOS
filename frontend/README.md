# CognitionOS Frontend

Real-time visualization and monitoring interface for CognitionOS agents.

## Features

- **Agent Thinking Visualization**: See agent reasoning steps in real-time
- **Task Graph**: Visualize task dependencies and execution flow
- **Execution Timeline**: Track timing and performance of agent operations
- **Memory Dashboard**: Monitor memory usage and retrieval
- **System Health**: Real-time metrics and alerting

## Setup

```bash
npm install
npm run dev
```

The frontend will be available at http://localhost:3000

## Environment Variables

Create a `.env.local` file:

```
API_GATEWAY_URL=http://localhost:8000
EXPLAINABILITY_URL=http://localhost:8008
OBSERVABILITY_URL=http://localhost:8009
```

## Architecture

- **Framework**: Next.js 14 with TypeScript
- **Styling**: Tailwind CSS
- **State Management**: React Query
- **Visualizations**: Recharts, React Flow

## Components

- `ReasoningVisualization`: Displays agent reasoning traces with confidence scores
- `TaskGraph`: Interactive task dependency graph
- `ExecutionTimeline`: Timeline of execution events
- `MetricsPanel`: Real-time system metrics
- `MemoryDashboard`: Memory usage visualization

## Development

```bash
npm run dev     # Start development server
npm run build   # Production build
npm run lint    # Run ESLint
```
