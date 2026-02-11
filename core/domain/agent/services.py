"""
Agent Domain - Domain Services

Domain services for agent-related business logic.
"""

from typing import List, Optional
from uuid import UUID

from .entities import (
    Agent,
    AgentDefinition,
    AgentStatus,
    BudgetLimits,
    Capability
)


class AgentMatcher:
    """
    Domain service for matching agents to tasks.

    Determines which agents are best suited for specific tasks.
    """

    @staticmethod
    def find_best_match(
        agents: List[Agent],
        required_capabilities: List[str],
        prefer_role: Optional[str] = None
    ) -> Optional[Agent]:
        """
        Find the best agent to execute a task.

        Selection criteria (in order of priority):
        1. Can execute task (has capabilities, is idle, has budget)
        2. Matches preferred role if specified
        3. Has lowest budget usage (most capacity remaining)
        4. Least recently used

        Args:
            agents: List of candidate agents
            required_capabilities: Required capabilities for task
            prefer_role: Preferred agent role

        Returns:
            Best matching agent, or None if no match
        """
        # Filter to agents that can execute the task
        capable_agents = [
            agent for agent in agents
            if agent.can_execute_task(required_capabilities)
        ]

        if not capable_agents:
            return None

        # Prefer specific role if specified
        if prefer_role:
            role_match = [
                agent for agent in capable_agents
                if agent.role.value == prefer_role
            ]
            if role_match:
                capable_agents = role_match

        # Score agents
        def score_agent(agent: Agent) -> tuple:
            """
            Score agent for selection.
            Returns tuple for sorting (lower is better):
            - budget_ratio: % of budget used
            - last_active_timestamp: negative so more recent is better
            """
            budget_ratio = (
                agent.budget_usage.tokens_used / agent.budget_limits.max_tokens
                if agent.budget_limits.max_tokens > 0
                else 0
            )
            return (budget_ratio, -agent.last_active.timestamp())

        # Return agent with best score
        return min(capable_agents, key=score_agent)

    @staticmethod
    def find_all_capable_agents(
        agents: List[Agent],
        required_capabilities: List[str]
    ) -> List[Agent]:
        """
        Find all agents capable of executing a task.

        Args:
            agents: List of candidate agents
            required_capabilities: Required capabilities

        Returns:
            List of capable agents, sorted by availability
        """
        capable = [
            agent for agent in agents
            if agent.can_execute_task(required_capabilities)
        ]

        # Sort by budget usage (ascending)
        return sorted(
            capable,
            key=lambda a: a.budget_usage.tokens_used / a.budget_limits.max_tokens
        )


class AgentLoadBalancer:
    """
    Domain service for distributing tasks across agents.

    Ensures fair distribution and prevents agent overload.
    """

    @staticmethod
    def should_create_new_agent(
        active_agents: List[Agent],
        idle_agents: List[Agent],
        required_capabilities: List[str],
        max_agents: int = 10
    ) -> bool:
        """
        Determine if a new agent should be created.

        Create new agent if:
        1. No idle agents can handle the task
        2. All capable agents are approaching budget limits
        3. Haven't reached max agent count

        Args:
            active_agents: Currently active agents
            idle_agents: Currently idle agents
            required_capabilities: Required capabilities
            max_agents: Maximum allowed agents

        Returns:
            True if new agent should be created
        """
        total_agents = len(active_agents) + len(idle_agents)

        # Don't exceed max agents
        if total_agents >= max_agents:
            return False

        # Check if any idle agent can handle task
        capable_idle = AgentMatcher.find_all_capable_agents(
            idle_agents,
            required_capabilities
        )

        if not capable_idle:
            # No idle agents can handle it - create new one
            return True

        # Check if idle agents have sufficient budget
        any_with_budget = any(
            agent.budget_usage.within_limits(agent.budget_limits) and
            not agent.budget_usage.is_approaching_limit(agent.budget_limits)
            for agent in capable_idle
        )

        if not any_with_budget:
            # All capable idle agents are near budget limits
            return True

        return False

    @staticmethod
    def get_agents_to_terminate(
        agents: List[Agent],
        keep_minimum: int = 2,
        idle_timeout_seconds: int = 3600
    ) -> List[Agent]:
        """
        Identify agents that should be terminated to save resources.

        Terminate agents that:
        1. Have been idle for longer than timeout
        2. Are not part of minimum keep count
        3. Have no active tasks

        Args:
            agents: All agents
            keep_minimum: Minimum number of agents to keep
            idle_timeout_seconds: Idle timeout before termination

        Returns:
            List of agents to terminate
        """
        from datetime import datetime, timedelta

        # Filter to idle agents
        idle_agents = [a for a in agents if a.status == AgentStatus.IDLE]

        if len(idle_agents) <= keep_minimum:
            return []

        # Find agents idle beyond timeout
        timeout = timedelta(seconds=idle_timeout_seconds)
        now = datetime.utcnow()

        idle_too_long = [
            agent for agent in idle_agents
            if (now - agent.last_active) > timeout
        ]

        # Sort by uptime (terminate oldest idle agents first)
        idle_too_long.sort(key=lambda a: a.uptime_seconds(), reverse=True)

        # Keep at least minimum number
        can_terminate = max(0, len(idle_agents) - keep_minimum)
        return idle_too_long[:can_terminate]


class AgentHealthMonitor:
    """
    Domain service for monitoring agent health and performance.

    Detects agents that are stuck or performing poorly.
    """

    @staticmethod
    def is_agent_stuck(agent: Agent, max_execution_seconds: int = 600) -> bool:
        """
        Determine if agent is stuck in execution.

        An agent is considered stuck if it's been in an active state
        for longer than expected without progress.

        Args:
            agent: Agent to check
            max_execution_seconds: Maximum expected execution time

        Returns:
            True if agent appears stuck
        """
        from datetime import datetime, timedelta

        if not agent.is_active():
            return False

        timeout = timedelta(seconds=max_execution_seconds)
        time_in_state = datetime.utcnow() - agent.last_active

        return time_in_state > timeout

    @staticmethod
    def is_budget_exceeded(agent: Agent) -> bool:
        """
        Check if agent has exceeded budget limits.

        Args:
            agent: Agent to check

        Returns:
            True if budget exceeded
        """
        return not agent.budget_usage.within_limits(agent.budget_limits)

    @staticmethod
    def is_approaching_budget_limit(
        agent: Agent,
        threshold: float = 0.8
    ) -> bool:
        """
        Check if agent is approaching budget limits.

        Args:
            agent: Agent to check
            threshold: Threshold percentage (0.8 = 80%)

        Returns:
            True if approaching limit
        """
        return agent.budget_usage.is_approaching_limit(
            agent.budget_limits,
            threshold
        )

    @staticmethod
    def get_unhealthy_agents(agents: List[Agent]) -> List[Agent]:
        """
        Get all agents in unhealthy state.

        Unhealthy agents are:
        - Stuck in execution
        - Budget exceeded
        - Failed status

        Args:
            agents: List of agents to check

        Returns:
            List of unhealthy agents
        """
        unhealthy = []

        for agent in agents:
            if agent.status == AgentStatus.FAILED:
                unhealthy.append(agent)
            elif AgentHealthMonitor.is_agent_stuck(agent):
                unhealthy.append(agent)
            elif AgentHealthMonitor.is_budget_exceeded(agent):
                unhealthy.append(agent)

        return unhealthy


class AgentCapabilityRegistry:
    """
    Domain service for managing agent capabilities.

    Provides semantic matching and capability discovery.
    """

    @staticmethod
    def find_agents_with_capability(
        agents: List[Agent],
        capability_name: str
    ) -> List[Agent]:
        """
        Find all agents with a specific capability.

        Args:
            agents: List of agents
            capability_name: Capability to search for

        Returns:
            List of agents with the capability
        """
        return [
            agent for agent in agents
            if agent.has_capability(capability_name)
        ]

    @staticmethod
    def get_all_capabilities(agents: List[Agent]) -> List[str]:
        """
        Get all unique capabilities across agents.

        Args:
            agents: List of agents

        Returns:
            List of unique capability names
        """
        capabilities = set()
        for agent in agents:
            for cap in agent.capabilities:
                capabilities.add(cap.name)
        return sorted(list(capabilities))

    @staticmethod
    def find_definitions_for_capabilities(
        definitions: List[AgentDefinition],
        required_capabilities: List[str]
    ) -> List[AgentDefinition]:
        """
        Find agent definitions that support required capabilities.

        Args:
            definitions: List of agent definitions
            required_capabilities: Required capabilities

        Returns:
            List of compatible definitions
        """
        return [
            defn for defn in definitions
            if defn.is_compatible_with(required_capabilities)
        ]
