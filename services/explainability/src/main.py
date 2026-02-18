"""
Explainability Service.

Generates human-readable explanations of agent reasoning, decisions, and actions.
Provides reasoning summaries, confidence scoring, and decision traces.
"""

import os

# Add shared libs to path

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from uuid import UUID, uuid4
from enum import Enum

from fastapi import FastAPI, HTTPException, status, Depends
from pydantic import BaseModel, Field
from sqlalchemy import Column, String, DateTime, Float, Integer, JSON, Text, Index, Boolean
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from shared.libs.config import BaseConfig, load_config
from shared.libs.logger import setup_logger, get_contextual_logger
from shared.libs.models import ErrorResponse
from shared.libs.middleware import (
    TracingMiddleware,
    LoggingMiddleware,
    ErrorHandlingMiddleware
)


# Configuration
class ExplainabilityConfig(BaseConfig):
    """Configuration for Explainability service."""

    service_name: str = "explainability"
    port: int = Field(default=8008, env="PORT")

    # Explanation settings
    max_reasoning_depth: int = Field(default=10, env="MAX_REASONING_DEPTH")
    min_confidence_threshold: float = Field(default=0.5, env="MIN_CONFIDENCE_THRESHOLD")
    enable_verbose_mode: bool = Field(default=False, env="ENABLE_VERBOSE_MODE")


config = load_config(ExplainabilityConfig)
logger = setup_logger(__name__, level=config.log_level)

# FastAPI app
app = FastAPI(
    title="CognitionOS Explainability",
    version=config.service_version,
    description="Agent reasoning and decision explanation service"
)

# Add middleware
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(TracingMiddleware)


# ============================================================================
# Database Models
# ============================================================================

Base = declarative_base()


class ReasoningTrace(Base):
    """
    Stores reasoning trace for agent decisions.
    """
    __tablename__ = "reasoning_traces"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Context
    task_id = Column(PG_UUID(as_uuid=True), nullable=False, index=True)
    agent_id = Column(PG_UUID(as_uuid=True), nullable=False, index=True)
    user_id = Column(PG_UUID(as_uuid=True), nullable=False, index=True)

    # Reasoning step
    step_number = Column(Integer, nullable=False)
    step_type = Column(String(50), nullable=False)  # plan, reason, execute, critique, summarize
    step_description = Column(Text, nullable=False)

    # Input/Output
    input_data = Column(JSON, nullable=True)
    output_data = Column(JSON, nullable=True)
    model_used = Column(String(100), nullable=True)

    # Metrics
    confidence_score = Column(Float, nullable=True)
    reasoning_quality = Column(Float, nullable=True)
    tokens_used = Column(Integer, nullable=True)
    duration_ms = Column(Integer, nullable=True)

    # Decision factors
    factors_considered = Column(JSON, nullable=True)  # List of factors that influenced decision
    alternatives_evaluated = Column(JSON, nullable=True)  # Alternative options considered
    selection_rationale = Column(Text, nullable=True)  # Why this option was chosen

    # Metadata
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    parent_step_id = Column(PG_UUID(as_uuid=True), nullable=True)  # For hierarchical reasoning

    __table_args__ = (
        Index('idx_reasoning_task_step', 'task_id', 'step_number'),
        Index('idx_reasoning_agent', 'agent_id', 'timestamp'),
    )


class ExecutionTimeline(Base):
    """
    Timeline of agent execution events.
    """
    __tablename__ = "execution_timelines"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Context
    task_id = Column(PG_UUID(as_uuid=True), nullable=False, index=True)
    agent_id = Column(PG_UUID(as_uuid=True), nullable=False, index=True)

    # Event
    event_type = Column(String(100), nullable=False)
    event_name = Column(String(200), nullable=False)
    event_description = Column(Text, nullable=True)

    # Timing
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    duration_ms = Column(Integer, nullable=True)

    # Status
    status = Column(String(50), nullable=False)  # started, completed, failed
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)

    # Metrics
    tokens_used = Column(Integer, nullable=True)
    cost_usd = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)

    # Data
    metadata = Column(JSON, nullable=True)

    __table_args__ = (
        Index('idx_timeline_task', 'task_id', 'start_time'),
    )


# ============================================================================
# Enums and Models
# ============================================================================

class ReasoningStepType(str, Enum):
    """Types of reasoning steps."""
    PLAN = "plan"
    REASON = "reason"
    EXECUTE = "execute"
    CRITIQUE = "critique"
    SUMMARIZE = "summarize"
    REFLECT = "reflect"


class ExplanationLevel(str, Enum):
    """Level of detail for explanations."""
    BRIEF = "brief"  # One-sentence summary
    STANDARD = "standard"  # Paragraph summary
    DETAILED = "detailed"  # Full reasoning trace
    VERBOSE = "verbose"  # Everything including internals


class RecordReasoningRequest(BaseModel):
    """Request to record a reasoning step."""
    task_id: UUID
    agent_id: UUID
    user_id: UUID
    step_number: int
    step_type: ReasoningStepType
    step_description: str
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    model_used: Optional[str] = None
    confidence_score: Optional[float] = None
    reasoning_quality: Optional[float] = None
    tokens_used: Optional[int] = None
    duration_ms: Optional[int] = None
    factors_considered: Optional[List[str]] = None
    alternatives_evaluated: Optional[List[Dict[str, Any]]] = None
    selection_rationale: Optional[str] = None
    parent_step_id: Optional[UUID] = None


class RecordTimelineEventRequest(BaseModel):
    """Request to record a timeline event."""
    task_id: UUID
    agent_id: UUID
    event_type: str
    event_name: str
    event_description: Optional[str] = None
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[int] = None
    status: str
    success: bool = True
    error_message: Optional[str] = None
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class ExplainTaskRequest(BaseModel):
    """Request to explain a task's execution."""
    task_id: UUID
    level: ExplanationLevel = ExplanationLevel.STANDARD
    include_timeline: bool = True
    include_reasoning: bool = True
    include_confidence: bool = True


class ReasoningSummary(BaseModel):
    """Summary of reasoning for a task."""
    task_id: UUID
    agent_id: UUID
    total_steps: int
    reasoning_summary: str
    key_decisions: List[Dict[str, Any]]
    confidence_scores: Dict[str, float]
    overall_quality: float


class TimelineSummary(BaseModel):
    """Summary of execution timeline."""
    task_id: UUID
    total_duration_ms: int
    total_tokens: int
    total_cost_usd: float
    events: List[Dict[str, Any]]
    critical_path: List[str]


class TaskExplanation(BaseModel):
    """Complete explanation of task execution."""
    task_id: UUID
    summary: str
    reasoning_summary: Optional[ReasoningSummary] = None
    timeline_summary: Optional[TimelineSummary] = None
    confidence_analysis: Optional[Dict[str, Any]] = None
    generated_at: datetime


# ============================================================================
# Database
# ============================================================================

engine = create_async_engine(
    config.database_url,
    pool_size=config.database_pool_size,
    max_overflow=config.database_max_overflow,
    echo=config.debug
)

async_session_maker = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def get_db() -> AsyncSession:
    """Get database session."""
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


# ============================================================================
# Explainability Engine
# ============================================================================

class ExplainabilityEngine:
    """
    Generates explanations of agent reasoning and behavior.
    """

    def __init__(self):
        self.logger = get_contextual_logger(__name__, component="ExplainabilityEngine")

    async def generate_explanation(
        self,
        task_id: UUID,
        level: ExplanationLevel,
        include_timeline: bool,
        include_reasoning: bool,
        include_confidence: bool,
        db: AsyncSession
    ) -> TaskExplanation:
        """
        Generate comprehensive explanation for a task.

        Args:
            task_id: Task to explain
            level: Level of detail
            include_timeline: Whether to include execution timeline
            include_reasoning: Whether to include reasoning summary
            include_confidence: Whether to include confidence analysis
            db: Database session

        Returns:
            Complete task explanation
        """
        from sqlalchemy import select

        # Get reasoning traces
        reasoning_summary = None
        if include_reasoning:
            reasoning_query = select(ReasoningTrace).where(
                ReasoningTrace.task_id == task_id
            ).order_by(ReasoningTrace.step_number)

            result = await db.execute(reasoning_query)
            reasoning_traces = result.scalars().all()

            if reasoning_traces:
                reasoning_summary = self._summarize_reasoning(
                    task_id,
                    reasoning_traces,
                    level
                )

        # Get timeline events
        timeline_summary = None
        if include_timeline:
            timeline_query = select(ExecutionTimeline).where(
                ExecutionTimeline.task_id == task_id
            ).order_by(ExecutionTimeline.start_time)

            result = await db.execute(timeline_query)
            timeline_events = result.scalars().all()

            if timeline_events:
                timeline_summary = self._summarize_timeline(
                    task_id,
                    timeline_events,
                    level
                )

        # Generate confidence analysis
        confidence_analysis = None
        if include_confidence and reasoning_summary:
            confidence_analysis = self._analyze_confidence(reasoning_summary)

        # Generate overall summary
        summary = self._generate_summary(
            reasoning_summary,
            timeline_summary,
            level
        )

        return TaskExplanation(
            task_id=task_id,
            summary=summary,
            reasoning_summary=reasoning_summary,
            timeline_summary=timeline_summary,
            confidence_analysis=confidence_analysis,
            generated_at=datetime.utcnow()
        )

    def _summarize_reasoning(
        self,
        task_id: UUID,
        traces: List[ReasoningTrace],
        level: ExplanationLevel
    ) -> ReasoningSummary:
        """Generate reasoning summary from traces."""
        if not traces:
            return ReasoningSummary(
                task_id=task_id,
                agent_id=traces[0].agent_id if traces else None,
                total_steps=0,
                reasoning_summary="No reasoning data available",
                key_decisions=[],
                confidence_scores={},
                overall_quality=0.0
            )

        # Extract key decisions (steps with alternatives evaluated)
        key_decisions = []
        for trace in traces:
            if trace.alternatives_evaluated:
                key_decisions.append({
                    "step": trace.step_number,
                    "type": trace.step_type,
                    "description": trace.step_description,
                    "alternatives": trace.alternatives_evaluated,
                    "rationale": trace.selection_rationale,
                    "confidence": trace.confidence_score
                })

        # Collect confidence scores by step type
        confidence_by_type = {}
        for trace in traces:
            if trace.confidence_score is not None:
                step_type = trace.step_type
                if step_type not in confidence_by_type:
                    confidence_by_type[step_type] = []
                confidence_by_type[step_type].append(trace.confidence_score)

        # Average confidence scores
        confidence_scores = {
            step_type: sum(scores) / len(scores)
            for step_type, scores in confidence_by_type.items()
        }

        # Calculate overall quality
        quality_scores = [t.reasoning_quality for t in traces if t.reasoning_quality is not None]
        overall_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

        # Generate narrative summary based on level
        if level == ExplanationLevel.BRIEF:
            reasoning_summary = f"Completed {len(traces)} reasoning steps with {len(key_decisions)} key decisions."
        elif level == ExplanationLevel.STANDARD:
            reasoning_summary = self._generate_standard_summary(traces, key_decisions)
        else:  # DETAILED or VERBOSE
            reasoning_summary = self._generate_detailed_summary(traces, key_decisions)

        return ReasoningSummary(
            task_id=task_id,
            agent_id=traces[0].agent_id,
            total_steps=len(traces),
            reasoning_summary=reasoning_summary,
            key_decisions=key_decisions[:5] if level != ExplanationLevel.VERBOSE else key_decisions,
            confidence_scores=confidence_scores,
            overall_quality=overall_quality
        )

    def _generate_standard_summary(
        self,
        traces: List[ReasoningTrace],
        key_decisions: List[Dict[str, Any]]
    ) -> str:
        """Generate standard-level reasoning summary."""
        summary_parts = []

        # Group by step type
        by_type = {}
        for trace in traces:
            if trace.step_type not in by_type:
                by_type[trace.step_type] = []
            by_type[trace.step_type].append(trace)

        # Describe each phase
        if "plan" in by_type:
            summary_parts.append(
                f"Planning phase: Analyzed requirements and created {len(by_type['plan'])} planning steps."
            )

        if "reason" in by_type:
            summary_parts.append(
                f"Reasoning phase: Evaluated {len(by_type['reason'])} logical steps and decision points."
            )

        if "execute" in by_type:
            summary_parts.append(
                f"Execution phase: Performed {len(by_type['execute'])} execution steps."
            )

        if "critique" in by_type:
            summary_parts.append(
                f"Critique phase: Reviewed and validated results through {len(by_type['critique'])} critiques."
            )

        # Mention key decisions
        if key_decisions:
            summary_parts.append(
                f"Made {len(key_decisions)} key decisions, each considering multiple alternatives."
            )

        return " ".join(summary_parts)

    def _generate_detailed_summary(
        self,
        traces: List[ReasoningTrace],
        key_decisions: List[Dict[str, Any]]
    ) -> str:
        """Generate detailed reasoning summary."""
        summary_lines = ["## Reasoning Trace\n"]

        for trace in traces:
            summary_lines.append(
                f"{trace.step_number}. **{trace.step_type.upper()}**: {trace.step_description}"
            )

            if trace.confidence_score is not None:
                summary_lines.append(f"   - Confidence: {trace.confidence_score:.2f}")

            if trace.alternatives_evaluated:
                summary_lines.append(
                    f"   - Alternatives considered: {len(trace.alternatives_evaluated)}"
                )

            if trace.selection_rationale:
                summary_lines.append(f"   - Rationale: {trace.selection_rationale}")

            summary_lines.append("")

        return "\n".join(summary_lines)

    def _summarize_timeline(
        self,
        task_id: UUID,
        events: List[ExecutionTimeline],
        level: ExplanationLevel
    ) -> TimelineSummary:
        """Generate timeline summary from events."""
        total_duration = sum(e.duration_ms or 0 for e in events)
        total_tokens = sum(e.tokens_used or 0 for e in events)
        total_cost = sum(e.cost_usd or 0 for e in events)

        # Extract event summaries
        event_summaries = []
        for event in events:
            event_summaries.append({
                "name": event.event_name,
                "type": event.event_type,
                "duration_ms": event.duration_ms,
                "status": event.status,
                "success": event.success,
                "start_time": event.start_time.isoformat(),
                "end_time": event.end_time.isoformat() if event.end_time else None
            })

        # Identify critical path (longest running events)
        sorted_events = sorted(events, key=lambda e: e.duration_ms or 0, reverse=True)
        critical_path = [e.event_name for e in sorted_events[:3]]

        return TimelineSummary(
            task_id=task_id,
            total_duration_ms=total_duration,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            events=event_summaries[:10] if level != ExplanationLevel.VERBOSE else event_summaries,
            critical_path=critical_path
        )

    def _analyze_confidence(
        self,
        reasoning_summary: ReasoningSummary
    ) -> Dict[str, Any]:
        """Analyze confidence levels across reasoning."""
        if not reasoning_summary.confidence_scores:
            return {"overall": "insufficient_data"}

        avg_confidence = sum(reasoning_summary.confidence_scores.values()) / len(
            reasoning_summary.confidence_scores
        )

        # Determine confidence level
        if avg_confidence >= 0.8:
            confidence_level = "high"
            recommendation = "Results are highly reliable"
        elif avg_confidence >= 0.6:
            confidence_level = "moderate"
            recommendation = "Results are generally reliable but verify critical aspects"
        else:
            confidence_level = "low"
            recommendation = "Results should be verified independently"

        # Identify weak points
        weak_steps = [
            step_type for step_type, score in reasoning_summary.confidence_scores.items()
            if score < 0.5
        ]

        return {
            "average_confidence": avg_confidence,
            "confidence_level": confidence_level,
            "recommendation": recommendation,
            "by_step_type": reasoning_summary.confidence_scores,
            "weak_points": weak_steps,
            "overall_quality": reasoning_summary.overall_quality
        }

    def _generate_summary(
        self,
        reasoning_summary: Optional[ReasoningSummary],
        timeline_summary: Optional[TimelineSummary],
        level: ExplanationLevel
    ) -> str:
        """Generate overall task summary."""
        parts = []

        if reasoning_summary:
            if level == ExplanationLevel.BRIEF:
                parts.append(
                    f"Task completed with {reasoning_summary.total_steps} reasoning steps."
                )
            else:
                parts.append(reasoning_summary.reasoning_summary)

        if timeline_summary:
            duration_sec = timeline_summary.total_duration_ms / 1000
            parts.append(
                f"Execution took {duration_sec:.1f}s using {timeline_summary.total_tokens} tokens "
                f"(${timeline_summary.total_cost_usd:.4f})."
            )

        return " ".join(parts) if parts else "No data available for explanation."


# Initialize engine
explainability_engine = ExplainabilityEngine()


# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/reasoning/record")
async def record_reasoning_step(
    request: RecordReasoningRequest,
    db: AsyncSession = Depends(get_db)
):
    """Record a reasoning step for later explanation."""
    trace = ReasoningTrace(
        task_id=request.task_id,
        agent_id=request.agent_id,
        user_id=request.user_id,
        step_number=request.step_number,
        step_type=request.step_type.value,
        step_description=request.step_description,
        input_data=request.input_data,
        output_data=request.output_data,
        model_used=request.model_used,
        confidence_score=request.confidence_score,
        reasoning_quality=request.reasoning_quality,
        tokens_used=request.tokens_used,
        duration_ms=request.duration_ms,
        factors_considered=request.factors_considered,
        alternatives_evaluated=request.alternatives_evaluated,
        selection_rationale=request.selection_rationale,
        parent_step_id=request.parent_step_id
    )

    db.add(trace)
    await db.commit()
    await db.refresh(trace)

    return {"id": str(trace.id), "recorded": True}


@app.post("/timeline/record")
async def record_timeline_event(
    request: RecordTimelineEventRequest,
    db: AsyncSession = Depends(get_db)
):
    """Record a timeline event."""
    event = ExecutionTimeline(
        task_id=request.task_id,
        agent_id=request.agent_id,
        event_type=request.event_type,
        event_name=request.event_name,
        event_description=request.event_description,
        start_time=request.start_time,
        end_time=request.end_time,
        duration_ms=request.duration_ms,
        status=request.status,
        success=request.success,
        error_message=request.error_message,
        tokens_used=request.tokens_used,
        cost_usd=request.cost_usd,
        confidence=request.confidence,
        metadata=request.metadata
    )

    db.add(event)
    await db.commit()
    await db.refresh(event)

    return {"id": str(event.id), "recorded": True}


@app.post("/explain", response_model=TaskExplanation)
async def explain_task(
    request: ExplainTaskRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Generate explanation for a task's execution.

    Combines reasoning traces and timeline events into human-readable explanation.
    """
    log = get_contextual_logger(
        __name__,
        action="explain_task",
        task_id=str(request.task_id)
    )

    try:
        explanation = await explainability_engine.generate_explanation(
            task_id=request.task_id,
            level=request.level,
            include_timeline=request.include_timeline,
            include_reasoning=request.include_reasoning,
            include_confidence=request.include_confidence,
            db=db
        )

        return explanation

    except Exception as e:
        log.error(f"Failed to generate explanation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate explanation: {str(e)}"
        )


@app.get("/reasoning/{task_id}")
async def get_reasoning_trace(
    task_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get raw reasoning trace for a task."""
    from sqlalchemy import select

    query = select(ReasoningTrace).where(
        ReasoningTrace.task_id == task_id
    ).order_by(ReasoningTrace.step_number)

    result = await db.execute(query)
    traces = result.scalars().all()

    return {
        "task_id": str(task_id),
        "total_steps": len(traces),
        "traces": [
            {
                "step_number": t.step_number,
                "step_type": t.step_type,
                "description": t.step_description,
                "confidence": t.confidence_score,
                "quality": t.reasoning_quality,
                "model": t.model_used,
                "timestamp": t.timestamp.isoformat()
            }
            for t in traces
        ]
    }


@app.get("/timeline/{task_id}")
async def get_timeline(
    task_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get execution timeline for a task."""
    from sqlalchemy import select

    query = select(ExecutionTimeline).where(
        ExecutionTimeline.task_id == task_id
    ).order_by(ExecutionTimeline.start_time)

    result = await db.execute(query)
    events = result.scalars().all()

    return {
        "task_id": str(task_id),
        "total_events": len(events),
        "total_duration_ms": sum(e.duration_ms or 0 for e in events),
        "events": [
            {
                "name": e.event_name,
                "type": e.event_type,
                "start_time": e.start_time.isoformat(),
                "duration_ms": e.duration_ms,
                "status": e.status,
                "success": e.success
            }
            for e in events
        ]
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "explainability",
        "version": config.service_version,
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize service on startup."""
    logger.info(
        "Explainability service starting",
        extra={"version": config.service_version}
    )

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Explainability service ready")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("Explainability service shutting down")
    await engine.dispose()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower()
    )
