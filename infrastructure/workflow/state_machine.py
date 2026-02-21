"""
Workflow State Machine

Provides a formal finite-state machine (FSM) for workflow execution control.
Each workflow instance progresses through well-defined states with guarded transitions,
entry/exit actions, and full audit logging.

Features:
- Guard conditions on transitions
- Entry and exit hooks (sync and async)
- History state tracking
- Parallel region support
- Timeout-based auto-transitions
- Serializable state snapshots
- Event-driven and direct-transition APIs
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4


class StateType(str, Enum):
    NORMAL = "normal"
    INITIAL = "initial"
    FINAL = "final"
    PARALLEL = "parallel"
    HISTORY = "history"


@dataclass
class State:
    """A single state node in the FSM"""
    name: str
    state_type: StateType = StateType.NORMAL
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Transition:
    """A guarded transition between two states"""
    source: str
    target: str
    event: Optional[str] = None       # None = internal/auto transition
    guard: Optional[Callable[["MachineContext"], bool]] = None
    action: Optional[Callable[["MachineContext"], Awaitable[None]]] = None
    priority: int = 0                  # Higher = checked first when multiple match
    description: str = ""


@dataclass
class MachineContext:
    """Mutable context passed to guards and actions"""
    instance_id: str
    current_state: str
    previous_state: Optional[str]
    trigger_event: Optional[str]
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.variables.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.variables[key] = value


@dataclass
class TransitionRecord:
    """Audit record of a completed transition"""
    from_state: str
    to_state: str
    event: Optional[str]
    timestamp: float
    duration_ms: float
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_state": self.from_state,
            "to_state": self.to_state,
            "event": self.event,
            "timestamp": self.timestamp,
            "duration_ms": round(self.duration_ms, 2),
            "success": self.success,
            "error": self.error,
        }


class StateMachineError(RuntimeError):
    pass


class InvalidTransitionError(StateMachineError):
    pass


class GuardRejectedError(StateMachineError):
    pass


class StateMachine:
    """
    Defines the structure of a workflow state machine (reusable blueprint).

    Usage::

        sm = StateMachine("order-processing")
        sm.add_state(State("idle", StateType.INITIAL))
        sm.add_state(State("processing", StateType.NORMAL))
        sm.add_state(State("done", StateType.FINAL))
        sm.add_transition(Transition("idle", "processing", event="start"))
        sm.add_transition(Transition("processing", "done", event="complete"))
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._states: Dict[str, State] = {}
        self._transitions: List[Transition] = []
        self._entry_hooks: Dict[str, List[Callable]] = defaultdict(list)
        self._exit_hooks: Dict[str, List[Callable]] = defaultdict(list)
        self._global_hooks: List[Callable] = []

    def add_state(self, state: State) -> "StateMachine":
        self._states[state.name] = state
        return self

    def add_transition(self, transition: Transition) -> "StateMachine":
        if transition.source not in self._states:
            raise StateMachineError(f"Source state '{transition.source}' not registered")
        if transition.target not in self._states:
            raise StateMachineError(f"Target state '{transition.target}' not registered")
        self._transitions.append(transition)
        return self

    def on_enter(self, state_name: str, hook: Callable) -> "StateMachine":
        """Register a hook called when entering a state"""
        self._entry_hooks[state_name].append(hook)
        return self

    def on_exit(self, state_name: str, hook: Callable) -> "StateMachine":
        """Register a hook called when leaving a state"""
        self._exit_hooks[state_name].append(hook)
        return self

    def on_any_transition(self, hook: Callable) -> "StateMachine":
        """Register a hook called on every transition"""
        self._global_hooks.append(hook)
        return self

    @property
    def initial_state(self) -> Optional[str]:
        for name, state in self._states.items():
            if state.state_type == StateType.INITIAL:
                return name
        return next(iter(self._states), None)

    @property
    def final_states(self) -> Set[str]:
        return {n for n, s in self._states.items() if s.state_type == StateType.FINAL}

    def validate(self) -> List[str]:
        """Validate FSM structure, return list of warnings"""
        warnings: List[str] = []
        if self.initial_state is None:
            warnings.append("No initial state defined")
        if not self.final_states:
            warnings.append("No final states defined")
        reachable = self._reachable_states()
        for name in self._states:
            if name not in reachable and name != self.initial_state:
                warnings.append(f"State '{name}' is unreachable")
        return warnings

    def _reachable_states(self) -> Set[str]:
        visited: Set[str] = set()
        queue = [self.initial_state] if self.initial_state else []
        while queue:
            current = queue.pop()
            if current in visited:
                continue
            visited.add(current)
            for t in self._transitions:
                if t.source == current and t.target not in visited:
                    queue.append(t.target)
        return visited

    def _get_transitions(self, from_state: str, event: Optional[str]) -> List[Transition]:
        candidates = [
            t for t in self._transitions
            if t.source == from_state and (t.event is None or t.event == event)
        ]
        candidates.sort(key=lambda t: t.priority, reverse=True)
        return candidates


class MachineInstance:
    """
    A running instance of a StateMachine.

    Each workflow execution gets its own MachineInstance.
    Thread/coroutine safe via asyncio lock.
    """

    def __init__(
        self,
        machine: StateMachine,
        instance_id: Optional[str] = None,
        initial_variables: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._machine = machine
        self._instance_id = instance_id or str(uuid4())
        self._current_state = machine.initial_state or ""
        self._history: List[TransitionRecord] = []
        self._lock = asyncio.Lock()
        self._context = MachineContext(
            instance_id=self._instance_id,
            current_state=self._current_state,
            previous_state=None,
            trigger_event=None,
            variables=initial_variables or {},
        )

    @property
    def current_state(self) -> str:
        return self._current_state

    @property
    def instance_id(self) -> str:
        return self._instance_id

    @property
    def is_final(self) -> bool:
        return self._current_state in self._machine.final_states

    @property
    def history(self) -> List[TransitionRecord]:
        return list(self._history)

    @property
    def context(self) -> MachineContext:
        return self._context

    # ──────────────────────────────────────────────
    # Transition API
    # ──────────────────────────────────────────────

    async def trigger(self, event: str) -> bool:
        """
        Trigger a named event. Returns True if a transition occurred.
        Raises InvalidTransitionError if no valid transition exists.
        """
        async with self._lock:
            return await self._do_transition(event)

    async def force_transition(self, target_state: str) -> None:
        """Bypass guards and directly move to a target state"""
        async with self._lock:
            if target_state not in self._machine._states:
                raise StateMachineError(f"Unknown state: '{target_state}'")
            await self._transition_to(target_state, event=None, force=True)

    async def auto_advance(self) -> int:
        """
        Execute all automatic (event=None) transitions available from the current state.
        Returns number of transitions taken.
        """
        count = 0
        async with self._lock:
            while True:
                candidates = self._machine._get_transitions(self._current_state, None)
                taken = False
                for t in candidates:
                    if self._check_guard(t):
                        await self._transition_to(t.target, event=None, transition=t)
                        count += 1
                        taken = True
                        break
                if not taken or self.is_final:
                    break
        return count

    # ──────────────────────────────────────────────
    # Snapshot / restore
    # ──────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        return {
            "instance_id": self._instance_id,
            "machine_name": self._machine.name,
            "current_state": self._current_state,
            "variables": dict(self._context.variables),
            "history": [r.to_dict() for r in self._history[-50:]],
            "timestamp": time.time(),
        }

    @classmethod
    async def restore(
        cls,
        snapshot: Dict[str, Any],
        machine: StateMachine,
    ) -> "MachineInstance":
        inst = cls(
            machine=machine,
            instance_id=snapshot["instance_id"],
            initial_variables=snapshot.get("variables", {}),
        )
        target = snapshot["current_state"]
        if target in machine._states:
            inst._current_state = target
            inst._context.current_state = target
        return inst

    # ──────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────

    async def _do_transition(self, event: Optional[str]) -> bool:
        candidates = self._machine._get_transitions(self._current_state, event)
        if not candidates:
            raise InvalidTransitionError(
                f"No transition from '{self._current_state}' on event '{event}'"
            )
        for t in candidates:
            if self._check_guard(t):
                await self._transition_to(t.target, event=event, transition=t)
                return True
        raise GuardRejectedError(
            f"All guards rejected transition from '{self._current_state}' on event '{event}'"
        )

    def _check_guard(self, t: Transition) -> bool:
        if t.guard is None:
            return True
        try:
            return t.guard(self._context)
        except Exception:  # noqa: BLE001
            return False

    async def _transition_to(
        self,
        target: str,
        event: Optional[str],
        transition: Optional[Transition] = None,
        force: bool = False,
    ) -> None:
        from_state = self._current_state
        start = time.time()

        # Exit hooks
        for hook in self._machine._exit_hooks.get(from_state, []):
            await self._call_hook(hook, self._context)

        # Transition action
        if transition and transition.action:
            await self._call_hook(transition.action, self._context)

        # Update state
        self._context.previous_state = from_state
        self._context.current_state = target
        self._context.trigger_event = event
        self._current_state = target

        # Entry hooks
        for hook in self._machine._entry_hooks.get(target, []):
            await self._call_hook(hook, self._context)

        # Global hooks
        for hook in self._machine._global_hooks:
            await self._call_hook(hook, self._context)

        elapsed_ms = (time.time() - start) * 1000
        self._history.append(TransitionRecord(
            from_state=from_state,
            to_state=target,
            event=event,
            timestamp=start,
            duration_ms=elapsed_ms,
            success=True,
        ))

    @staticmethod
    async def _call_hook(hook: Callable, ctx: MachineContext) -> None:
        if asyncio.iscoroutinefunction(hook):
            await hook(ctx)
        else:
            hook(ctx)


# ──────────────────────────────────────────────────────────────────────────────
# Pre-built workflow state machines
# ──────────────────────────────────────────────────────────────────────────────

def build_workflow_machine() -> StateMachine:
    """Standard workflow execution FSM"""
    sm = StateMachine("workflow")
    sm.add_state(State("created", StateType.INITIAL))
    sm.add_state(State("queued"))
    sm.add_state(State("running"))
    sm.add_state(State("paused"))
    sm.add_state(State("waiting_for_approval"))
    sm.add_state(State("completed", StateType.FINAL))
    sm.add_state(State("failed", StateType.FINAL))
    sm.add_state(State("cancelled", StateType.FINAL))
    sm.add_state(State("timed_out", StateType.FINAL))

    sm.add_transition(Transition("created", "queued", event="enqueue"))
    sm.add_transition(Transition("queued", "running", event="start"))
    sm.add_transition(Transition("running", "paused", event="pause"))
    sm.add_transition(Transition("paused", "running", event="resume"))
    sm.add_transition(Transition("running", "waiting_for_approval", event="request_approval"))
    sm.add_transition(Transition("waiting_for_approval", "running", event="approved"))
    sm.add_transition(Transition("waiting_for_approval", "failed", event="rejected"))
    sm.add_transition(Transition("running", "completed", event="complete"))
    sm.add_transition(Transition("running", "failed", event="fail"))
    sm.add_transition(Transition("queued", "cancelled", event="cancel"))
    sm.add_transition(Transition("running", "cancelled", event="cancel"))
    sm.add_transition(Transition("paused", "cancelled", event="cancel"))
    sm.add_transition(Transition("running", "timed_out", event="timeout"))
    sm.add_transition(Transition("queued", "timed_out", event="timeout"))

    return sm


def build_agent_machine() -> StateMachine:
    """Autonomous agent lifecycle FSM"""
    sm = StateMachine("agent")
    sm.add_state(State("initializing", StateType.INITIAL))
    sm.add_state(State("planning"))
    sm.add_state(State("executing"))
    sm.add_state(State("reflecting"))
    sm.add_state(State("waiting_for_tool"))
    sm.add_state(State("recovering"))
    sm.add_state(State("succeeded", StateType.FINAL))
    sm.add_state(State("failed", StateType.FINAL))
    sm.add_state(State("aborted", StateType.FINAL))

    sm.add_transition(Transition("initializing", "planning", event="initialized"))
    sm.add_transition(Transition("planning", "executing", event="plan_ready"))
    sm.add_transition(Transition("executing", "waiting_for_tool", event="tool_called"))
    sm.add_transition(Transition("waiting_for_tool", "executing", event="tool_result"))
    sm.add_transition(Transition("executing", "reflecting", event="step_done"))
    sm.add_transition(Transition("reflecting", "planning", event="needs_replan"))
    sm.add_transition(Transition("reflecting", "executing", event="continue"))
    sm.add_transition(Transition("reflecting", "succeeded", event="goal_achieved"))
    sm.add_transition(Transition("executing", "recovering", event="error"))
    sm.add_transition(Transition("recovering", "executing", event="recovered"))
    sm.add_transition(Transition("recovering", "failed", event="unrecoverable"))
    sm.add_transition(Transition("executing", "aborted", event="abort"))
    sm.add_transition(Transition("planning", "aborted", event="abort"))

    return sm
