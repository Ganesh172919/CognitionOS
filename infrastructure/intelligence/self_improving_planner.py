"""
Self-Improving Planning Engine with Reinforcement Learning

Enables agents to learn from past executions and improve over time.
"""
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict
import asyncio

import numpy as np
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.persistence.workflow_models import (
    WorkflowExecutionModel,
    WorkflowStepExecutionModel
)


class LearningStrategy(str, Enum):
    """Learning strategies for agent improvement"""
    Q_LEARNING = "q_learning"
    POLICY_GRADIENT = "policy_gradient"
    MONTE_CARLO = "monte_carlo"
    TEMPORAL_DIFFERENCE = "temporal_difference"


class RewardSignal(str, Enum):
    """Reward signals for learning"""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL_SUCCESS = "partial_success"
    EFFICIENCY = "efficiency"
    COST = "cost"
    QUALITY = "quality"


@dataclass
class StateAction:
    """State-action pair for learning"""
    state: Dict[str, Any]  # Workflow state
    action: str  # Action taken (tool, LLM call, etc.)
    context: Dict[str, Any]  # Additional context


@dataclass
class LearningEpisode:
    """Single learning episode (execution)"""
    episode_id: str
    workflow_id: str
    execution_id: str
    states_actions: List[StateAction]
    rewards: List[float]
    total_reward: float
    success: bool
    duration_seconds: float
    cost_usd: float
    quality_score: float
    timestamp: datetime


@dataclass
class PolicyImprovement:
    """Policy improvement result"""
    workflow_id: str
    improvement_type: str
    old_success_rate: float
    new_success_rate: float
    improvement_percentage: float
    confidence: float
    recommended_changes: List[Dict[str, Any]]
    episodes_analyzed: int


@dataclass
class AgentPerformance:
    """Agent performance metrics"""
    agent_id: str
    agent_type: str
    executions_count: int
    success_rate: float
    avg_duration: float
    avg_cost: float
    avg_quality: float
    trend: str  # "improving", "declining", "stable"
    learning_rate: float


class SelfImprovingPlanner:
    """
    Self-improving planning engine with reinforcement learning.
    
    Features:
    - Learn from execution history
    - Q-learning for action selection
    - Policy gradient optimization
    - Automatic parameter tuning
    - Success pattern recognition
    - Failure mode analysis
    - Transfer learning across workflows
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
        
        # Learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1  # Exploration rate
        
        # Q-table for state-action values
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Policy parameters
        self.policy_weights: Dict[str, np.ndarray] = {}
    
    async def learn_from_execution(
        self,
        execution_id: str
    ) -> LearningEpisode:
        """Learn from a completed workflow execution"""
        # Fetch execution data
        exec_query = select(WorkflowExecutionModel).where(
            WorkflowExecutionModel.id == execution_id
        )
        exec_result = await self.session.execute(exec_query)
        execution = exec_result.scalar_one_or_none()
        
        if not execution:
            raise ValueError(f"Execution {execution_id} not found")
        
        # Fetch step executions
        steps_query = select(WorkflowStepExecutionModel).where(
            WorkflowStepExecutionModel.execution_id == execution_id
        ).order_by(WorkflowStepExecutionModel.started_at)
        
        steps_result = await self.session.execute(steps_query)
        steps = steps_result.scalars().all()
        
        # Convert to state-action sequence
        states_actions = []
        rewards = []
        
        for i, step in enumerate(steps):
            # Extract state (simplified)
            state = {
                "step_index": i,
                "step_type": step.step_id,
                "execution_status": execution.status,
                "previous_steps": [s.step_id for s in steps[:i]]
            }
            
            # Extract action
            action = step.step_id
            
            # Calculate reward for this step
            reward = self._calculate_step_reward(step, execution)
            
            states_actions.append(StateAction(
                state=state,
                action=action,
                context={"agent_capability": step.assigned_agent or "general"}
            ))
            rewards.append(reward)
        
        # Calculate total reward
        success = execution.status == "completed"
        duration = (execution.completed_at - execution.started_at).total_seconds() if execution.completed_at else 0
        cost = execution.cost_usd or 0
        quality = self._calculate_quality_score(execution, steps)
        
        total_reward = self._calculate_total_reward(
            success=success,
            duration=duration,
            cost=cost,
            quality=quality
        )
        
        episode = LearningEpisode(
            episode_id=f"episode_{execution_id}",
            workflow_id=execution.workflow_id,
            execution_id=execution_id,
            states_actions=states_actions,
            rewards=rewards,
            total_reward=total_reward,
            success=success,
            duration_seconds=duration,
            cost_usd=cost,
            quality_score=quality,
            timestamp=execution.completed_at or datetime.utcnow()
        )
        
        # Update Q-table
        await self._update_q_values(episode)
        
        return episode
    
    async def get_optimal_action(
        self,
        state: Dict[str, Any],
        workflow_id: str,
        available_actions: List[str]
    ) -> str:
        """Get optimal action for current state using learned policy"""
        state_key = self._serialize_state(state)
        
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.choice(available_actions)
        
        # Exploit: best known action
        action_values = {
            action: self.q_table[state_key][action]
            for action in available_actions
        }
        
        if not action_values:
            # No learned values yet, random choice
            return np.random.choice(available_actions)
        
        # Return action with highest Q-value
        best_action = max(action_values, key=action_values.get)
        return best_action
    
    async def analyze_policy_improvement(
        self,
        workflow_id: str,
        lookback_days: int = 90
    ) -> PolicyImprovement:
        """Analyze improvement in workflow policy over time"""
        start_date = datetime.utcnow() - timedelta(days=lookback_days)
        
        # Get all executions
        query = select(WorkflowExecutionModel).where(
            and_(
                WorkflowExecutionModel.workflow_id == workflow_id,
                WorkflowExecutionModel.started_at >= start_date
            )
        ).order_by(WorkflowExecutionModel.started_at)
        
        result = await self.session.execute(query)
        executions = result.scalars().all()
        
        if len(executions) < 10:
            return PolicyImprovement(
                workflow_id=workflow_id,
                improvement_type="insufficient_data",
                old_success_rate=0,
                new_success_rate=0,
                improvement_percentage=0,
                confidence=0,
                recommended_changes=[],
                episodes_analyzed=len(executions)
            )
        
        # Split into old and new periods
        split_idx = len(executions) // 2
        old_executions = executions[:split_idx]
        new_executions = executions[split_idx:]
        
        # Calculate success rates
        old_success_rate = sum(
            1 for e in old_executions if e.status == "completed"
        ) / len(old_executions)
        
        new_success_rate = sum(
            1 for e in new_executions if e.status == "completed"
        ) / len(new_executions)
        
        improvement_pct = (
            ((new_success_rate - old_success_rate) / old_success_rate * 100)
            if old_success_rate > 0 else 0
        )
        
        # Analyze patterns for recommendations
        recommendations = await self._generate_recommendations(
            workflow_id, new_executions
        )
        
        # Calculate confidence based on sample size
        confidence = min(1.0, len(executions) / 100)
        
        improvement_type = "improvement" if improvement_pct > 5 else (
            "decline" if improvement_pct < -5 else "stable"
        )
        
        return PolicyImprovement(
            workflow_id=workflow_id,
            improvement_type=improvement_type,
            old_success_rate=old_success_rate,
            new_success_rate=new_success_rate,
            improvement_percentage=improvement_pct,
            confidence=confidence,
            recommended_changes=recommendations,
            episodes_analyzed=len(executions)
        )
    
    async def get_agent_performance(
        self,
        agent_id: str,
        lookback_days: int = 30
    ) -> AgentPerformance:
        """Get performance metrics for an agent"""
        start_date = datetime.utcnow() - timedelta(days=lookback_days)
        
        # Query executions involving this agent
        # In production, would track agent assignments
        query = select(WorkflowExecutionModel).where(
            WorkflowExecutionModel.started_at >= start_date
        )
        
        result = await self.session.execute(query)
        executions = result.scalars().all()
        
        if not executions:
            return AgentPerformance(
                agent_id=agent_id,
                agent_type="unknown",
                executions_count=0,
                success_rate=0,
                avg_duration=0,
                avg_cost=0,
                avg_quality=0,
                trend="unknown",
                learning_rate=0
            )
        
        # Calculate metrics
        success_count = sum(1 for e in executions if e.status == "completed")
        success_rate = success_count / len(executions)
        
        durations = [
            (e.completed_at - e.started_at).total_seconds()
            for e in executions if e.completed_at
        ]
        avg_duration = np.mean(durations) if durations else 0
        
        costs = [e.cost_usd for e in executions if e.cost_usd]
        avg_cost = np.mean(costs) if costs else 0
        
        # Calculate quality scores
        quality_scores = []
        for execution in executions:
            steps_query = select(WorkflowStepExecutionModel).where(
                WorkflowStepExecutionModel.execution_id == execution.id
            )
            steps_result = await self.session.execute(steps_query)
            steps = steps_result.scalars().all()
            
            quality = self._calculate_quality_score(execution, steps)
            quality_scores.append(quality)
        
        avg_quality = np.mean(quality_scores) if quality_scores else 0
        
        # Determine trend
        if len(executions) >= 10:
            recent_success = sum(
                1 for e in executions[-5:] if e.status == "completed"
            ) / 5
            older_success = sum(
                1 for e in executions[:5] if e.status == "completed"
            ) / 5
            
            if recent_success > older_success + 0.1:
                trend = "improving"
            elif recent_success < older_success - 0.1:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "unknown"
        
        # Calculate learning rate (improvement per execution)
        if len(success_rate) >= 2:
            learning_rate = (success_rate - 0.5) / len(executions)
        else:
            learning_rate = 0
        
        return AgentPerformance(
            agent_id=agent_id,
            agent_type="general",  # Would extract from agent metadata
            executions_count=len(executions),
            success_rate=success_rate,
            avg_duration=avg_duration,
            avg_cost=avg_cost,
            avg_quality=avg_quality,
            trend=trend,
            learning_rate=learning_rate
        )
    
    async def transfer_learning(
        self,
        source_workflow_id: str,
        target_workflow_id: str,
        similarity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Transfer learned policy from one workflow to another"""
        # Get source workflow Q-values
        source_states = {
            k: v for k, v in self.q_table.items()
            if k.startswith(source_workflow_id)
        }
        
        if not source_states:
            return {
                "success": False,
                "reason": "No learned policy for source workflow",
                "transferred_states": 0
            }
        
        # Calculate workflow similarity
        similarity = await self._calculate_workflow_similarity(
            source_workflow_id, target_workflow_id
        )
        
        if similarity < similarity_threshold:
            return {
                "success": False,
                "reason": f"Workflows too dissimilar: {similarity:.2f} < {similarity_threshold}",
                "transferred_states": 0
            }
        
        # Transfer Q-values
        transferred = 0
        for source_state, actions in source_states.items():
            # Map state to target workflow
            target_state = source_state.replace(
                source_workflow_id, target_workflow_id, 1
            )
            
            # Transfer with decay based on similarity
            transfer_factor = similarity
            
            for action, q_value in actions.items():
                self.q_table[target_state][action] = q_value * transfer_factor
                transferred += 1
        
        return {
            "success": True,
            "similarity": similarity,
            "transferred_states": transferred,
            "transfer_factor": transfer_factor
        }
    
    async def optimize_hyperparameters(
        self,
        workflow_id: str,
        trials: int = 10
    ) -> Dict[str, float]:
        """Optimize learning hyperparameters"""
        # Grid search over hyperparameters
        learning_rates = [0.05, 0.1, 0.2]
        discount_factors = [0.8, 0.9, 0.95]
        epsilon_values = [0.05, 0.1, 0.2]
        
        best_params = None
        best_performance = -float('inf')
        
        for lr in learning_rates:
            for df in discount_factors:
                for eps in epsilon_values:
                    # Test parameters
                    self.learning_rate = lr
                    self.discount_factor = df
                    self.epsilon = eps
                    
                    # Evaluate performance
                    performance = await self._evaluate_policy_performance(
                        workflow_id
                    )
                    
                    if performance > best_performance:
                        best_performance = performance
                        best_params = {
                            "learning_rate": lr,
                            "discount_factor": df,
                            "epsilon": eps
                        }
        
        if best_params:
            self.learning_rate = best_params["learning_rate"]
            self.discount_factor = best_params["discount_factor"]
            self.epsilon = best_params["epsilon"]
        
        return best_params or {
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon
        }
    
    def _calculate_step_reward(
        self,
        step: WorkflowStepExecutionModel,
        execution: WorkflowExecutionModel
    ) -> float:
        """Calculate reward for a single step"""
        reward = 0.0
        
        # Success reward
        if step.status == "completed":
            reward += 1.0
        elif step.status == "failed":
            reward -= 1.0
        
        # Efficiency reward (faster is better)
        if step.started_at and step.completed_at:
            duration = (step.completed_at - step.started_at).total_seconds()
            # Normalize to 0-1, penalize slow steps
            if duration < 10:  # Under 10 seconds
                reward += 0.5
            elif duration > 60:  # Over 1 minute
                reward -= 0.2
        
        # Cost reward (cheaper is better)
        if hasattr(step, 'cost_usd') and step.cost_usd:
            if step.cost_usd < 0.01:
                reward += 0.2
            elif step.cost_usd > 0.1:
                reward -= 0.3
        
        return reward
    
    def _calculate_total_reward(
        self,
        success: bool,
        duration: float,
        cost: float,
        quality: float
    ) -> float:
        """Calculate total reward for an episode"""
        reward = 0.0
        
        # Success is primary goal
        if success:
            reward += 10.0
        else:
            reward -= 5.0
        
        # Efficiency bonus
        if duration < 60:
            reward += 3.0
        elif duration < 300:
            reward += 1.0
        elif duration > 600:
            reward -= 2.0
        
        # Cost efficiency
        if cost < 0.1:
            reward += 2.0
        elif cost < 0.5:
            reward += 0.5
        elif cost > 1.0:
            reward -= 1.0
        
        # Quality bonus
        reward += quality * 2.0
        
        return reward
    
    def _calculate_quality_score(
        self,
        execution: WorkflowExecutionModel,
        steps: List[WorkflowStepExecutionModel]
    ) -> float:
        """Calculate quality score for an execution"""
        if not steps:
            return 0.0
        
        # Success rate of steps
        success_rate = sum(
            1 for s in steps if s.status == "completed"
        ) / len(steps)
        
        # Completeness (all steps executed)
        completeness = 1.0 if execution.status == "completed" else 0.5
        
        # Efficiency (no retries)
        efficiency = 1.0  # Would track retries in production
        
        quality = (success_rate * 0.5 + completeness * 0.3 + efficiency * 0.2)
        
        return quality
    
    async def _update_q_values(self, episode: LearningEpisode):
        """Update Q-values using episode data"""
        # Backward pass through episode
        returns = 0
        
        for i in reversed(range(len(episode.states_actions))):
            state_action = episode.states_actions[i]
            reward = episode.rewards[i]
            
            # Calculate return
            returns = reward + self.discount_factor * returns
            
            # Update Q-value
            state_key = self._serialize_state(state_action.state)
            action_key = state_action.action
            
            old_q = self.q_table[state_key][action_key]
            
            # Q-learning update
            self.q_table[state_key][action_key] = (
                old_q + self.learning_rate * (returns - old_q)
            )
    
    def _serialize_state(self, state: Dict[str, Any]) -> str:
        """Serialize state to string key"""
        # Simplified serialization
        return json.dumps(state, sort_keys=True)
    
    async def _generate_recommendations(
        self,
        workflow_id: str,
        executions: List[WorkflowExecutionModel]
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on execution analysis"""
        recommendations = []
        
        # Analyze success patterns
        successful = [e for e in executions if e.status == "completed"]
        failed = [e for e in executions if e.status == "failed"]
        
        if len(failed) > len(successful) * 0.3:
            recommendations.append({
                "type": "high_failure_rate",
                "priority": "high",
                "description": f"High failure rate: {len(failed)}/{len(executions)} executions failed",
                "action": "Review error patterns and add retry logic"
            })
        
        # Analyze duration patterns
        if successful:
            durations = [
                (e.completed_at - e.started_at).total_seconds()
                for e in successful if e.completed_at
            ]
            
            if durations and np.mean(durations) > 300:
                recommendations.append({
                    "type": "slow_execution",
                    "priority": "medium",
                    "description": f"Average execution time: {np.mean(durations):.1f}s",
                    "action": "Consider parallel execution or caching"
                })
        
        # Analyze cost patterns
        costs = [e.cost_usd for e in executions if e.cost_usd]
        if costs and np.mean(costs) > 1.0:
            recommendations.append({
                "type": "high_cost",
                "priority": "medium",
                "description": f"Average cost: ${np.mean(costs):.2f}",
                "action": "Optimize LLM calls or use cheaper models"
            })
        
        return recommendations
    
    async def _calculate_workflow_similarity(
        self,
        workflow_id_1: str,
        workflow_id_2: str
    ) -> float:
        """Calculate similarity between two workflows"""
        # Get workflow data
        query1 = select(WorkflowExecutionModel).where(
            WorkflowExecutionModel.workflow_id == workflow_id_1
        ).limit(10)
        
        query2 = select(WorkflowExecutionModel).where(
            WorkflowExecutionModel.workflow_id == workflow_id_2
        ).limit(10)
        
        result1 = await self.session.execute(query1)
        result2 = await self.session.execute(query2)
        
        execs1 = result1.scalars().all()
        execs2 = result2.scalars().all()
        
        if not execs1 or not execs2:
            return 0.0
        
        # Compare step counts (simplified)
        # In production, would compare step types, dependencies, etc.
        avg_steps1 = np.mean([len(e.step_executions) for e in execs1])
        avg_steps2 = np.mean([len(e.step_executions) for e in execs2])
        
        if max(avg_steps1, avg_steps2) == 0:
            return 0.0
        
        step_similarity = 1 - abs(avg_steps1 - avg_steps2) / max(avg_steps1, avg_steps2)
        
        # Would add more sophisticated similarity measures in production
        return step_similarity
    
    async def _evaluate_policy_performance(
        self,
        workflow_id: str
    ) -> float:
        """Evaluate current policy performance"""
        # Get recent executions
        query = select(WorkflowExecutionModel).where(
            WorkflowExecutionModel.workflow_id == workflow_id
        ).order_by(WorkflowExecutionModel.started_at.desc()).limit(20)
        
        result = await self.session.execute(query)
        executions = result.scalars().all()
        
        if not executions:
            return 0.0
        
        # Calculate success rate
        success_rate = sum(
            1 for e in executions if e.status == "completed"
        ) / len(executions)
        
        # Calculate efficiency
        durations = [
            (e.completed_at - e.started_at).total_seconds()
            for e in executions if e.completed_at
        ]
        avg_duration = np.mean(durations) if durations else 300
        efficiency = max(0, 1 - avg_duration / 600)  # Normalize to 0-1
        
        # Combined performance score
        performance = success_rate * 0.7 + efficiency * 0.3
        
        return performance
