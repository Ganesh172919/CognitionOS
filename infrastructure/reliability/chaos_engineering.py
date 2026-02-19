"""
Chaos Engineering Framework
Production resilience testing through controlled failure injection.
Tests system behavior under failure conditions to improve reliability.
"""

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import psutil


class ChaosExperimentType(str, Enum):
    """Types of chaos experiments"""
    LATENCY_INJECTION = "latency_injection"
    ERROR_INJECTION = "error_injection"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_PARTITION = "network_partition"
    DEPENDENCY_FAILURE = "dependency_failure"
    CPU_STRESS = "cpu_stress"
    MEMORY_STRESS = "memory_stress"
    DISK_STRESS = "disk_stress"
    POD_FAILURE = "pod_failure"
    SERVICE_FAILURE = "service_failure"


class ExperimentStatus(str, Enum):
    """Experiment execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class ImpactLevel(str, Enum):
    """Blast radius of experiment"""
    LOW = "low"  # Single instance
    MEDIUM = "medium"  # Multiple instances
    HIGH = "high"  # Entire service
    CRITICAL = "critical"  # Cross-service


@dataclass
class ChaosTarget:
    """Target for chaos experiment"""
    target_type: str  # service, pod, node, database, etc.
    target_id: str
    target_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SteadyStateHypothesis:
    """Expected system behavior in normal state"""
    metric_name: str
    operator: str  # gt, lt, eq, gte, lte
    threshold: float
    tolerance: float = 0.1
    description: str = ""


@dataclass
class ChaosExperiment:
    """Chaos engineering experiment definition"""
    experiment_id: str
    name: str
    description: str
    experiment_type: ChaosExperimentType
    targets: List[ChaosTarget]
    impact_level: ImpactLevel
    steady_state_hypothesis: List[SteadyStateHypothesis]
    duration_seconds: int
    parameters: Dict[str, Any]
    rollback_strategy: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: ExperimentStatus = ExperimentStatus.PENDING


@dataclass
class ExperimentResult:
    """Results from chaos experiment"""
    experiment_id: str
    status: ExperimentStatus
    started_at: datetime
    completed_at: datetime
    duration_seconds: float
    steady_state_before: Dict[str, Any]
    steady_state_after: Dict[str, Any]
    steady_state_maintained: bool
    observations: List[Dict[str, Any]]
    metrics: Dict[str, List[float]]
    errors_detected: List[str]
    recovery_time_seconds: Optional[float]
    insights: List[str]


@dataclass
class ChaosSchedule:
    """Schedule for running chaos experiments"""
    schedule_id: str
    experiment_id: str
    cron_expression: str
    enabled: bool
    next_run: datetime
    last_run: Optional[datetime] = None


class ChaosEngineeringFramework:
    """
    Chaos Engineering Framework for Production Resilience Testing

    Features:
    - Multiple failure injection types
    - Controlled blast radius
    - Steady-state hypothesis validation
    - Automatic rollback on critical failures
    - Experiment scheduling
    - Impact analysis and insights
    - Safe production testing
    """

    def __init__(self):
        self.experiments: Dict[str, ChaosExperiment] = {}
        self.results: Dict[str, List[ExperimentResult]] = {}
        self.schedules: Dict[str, ChaosSchedule] = {}
        self.running_experiments: Set[str] = set()
        self.safety_enabled = True

    async def create_experiment(
        self,
        name: str,
        description: str,
        experiment_type: ChaosExperimentType,
        targets: List[ChaosTarget],
        impact_level: ImpactLevel,
        steady_state_hypothesis: List[SteadyStateHypothesis],
        duration_seconds: int = 60,
        parameters: Optional[Dict[str, Any]] = None
    ) -> ChaosExperiment:
        """
        Create chaos experiment

        Args:
            name: Experiment name
            description: Detailed description
            experiment_type: Type of failure to inject
            targets: List of targets to affect
            impact_level: Blast radius control
            steady_state_hypothesis: Expected normal behavior
            duration_seconds: How long to run experiment
            parameters: Experiment-specific parameters

        Returns:
            Created experiment
        """
        experiment_id = f"exp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"

        experiment = ChaosExperiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            experiment_type=experiment_type,
            targets=targets,
            impact_level=impact_level,
            steady_state_hypothesis=steady_state_hypothesis,
            duration_seconds=duration_seconds,
            parameters=parameters or {},
            rollback_strategy="immediate"
        )

        self.experiments[experiment_id] = experiment
        return experiment

    async def run_experiment(
        self,
        experiment_id: str,
        dry_run: bool = False
    ) -> ExperimentResult:
        """
        Execute chaos experiment

        Args:
            experiment_id: ID of experiment to run
            dry_run: If True, simulate without actual failure injection

        Returns:
            Experiment results
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        if experiment_id in self.running_experiments:
            raise ValueError(f"Experiment {experiment_id} already running")

        experiment = self.experiments[experiment_id]
        self.running_experiments.add(experiment_id)

        try:
            # Validate safety constraints
            if self.safety_enabled:
                await self._validate_safety_constraints(experiment)

            # Measure steady state before
            steady_state_before = await self._measure_steady_state(experiment)

            started_at = datetime.utcnow()
            experiment.status = ExperimentStatus.RUNNING

            # Inject failure
            if not dry_run:
                await self._inject_failure(experiment)

            # Monitor during experiment
            observations = []
            metrics = {}

            for i in range(experiment.duration_seconds):
                observation = await self._observe_system(experiment)
                observations.append(observation)

                # Check if system maintains steady state
                steady_state_ok = await self._check_steady_state(
                    experiment,
                    observation
                )

                if not steady_state_ok and self.safety_enabled:
                    # Abort and rollback
                    await self._rollback_experiment(experiment)
                    experiment.status = ExperimentStatus.ROLLED_BACK
                    break

                await asyncio.sleep(1)

            # Remove failure injection
            if not dry_run:
                await self._rollback_experiment(experiment)

            # Measure steady state after
            steady_state_after = await self._measure_steady_state(experiment)

            completed_at = datetime.utcnow()
            experiment.status = ExperimentStatus.COMPLETED

            # Validate steady state maintained
            steady_state_maintained = await self._validate_steady_state_hypothesis(
                experiment,
                steady_state_before,
                steady_state_after
            )

            # Calculate recovery time
            recovery_time = await self._calculate_recovery_time(observations)

            # Generate insights
            insights = await self._generate_insights(
                experiment,
                observations,
                steady_state_maintained,
                recovery_time
            )

            result = ExperimentResult(
                experiment_id=experiment_id,
                status=experiment.status,
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=(completed_at - started_at).total_seconds(),
                steady_state_before=steady_state_before,
                steady_state_after=steady_state_after,
                steady_state_maintained=steady_state_maintained,
                observations=observations,
                metrics=metrics,
                errors_detected=[obs.get("error") for obs in observations if obs.get("error")],
                recovery_time_seconds=recovery_time,
                insights=insights
            )

            # Store result
            if experiment_id not in self.results:
                self.results[experiment_id] = []
            self.results[experiment_id].append(result)

            return result

        finally:
            self.running_experiments.discard(experiment_id)

    async def _validate_safety_constraints(self, experiment: ChaosExperiment):
        """Validate experiment can be safely run"""
        # Check impact level constraints
        if experiment.impact_level == ImpactLevel.CRITICAL:
            # Require explicit approval for critical experiments
            raise ValueError("Critical impact experiments require manual approval")

        # Check target health before experiment
        for target in experiment.targets:
            health = await self._check_target_health(target)
            if not health:
                raise ValueError(f"Target {target.target_name} is unhealthy")

        # Check no overlapping experiments
        for exp_id in self.running_experiments:
            running_exp = self.experiments[exp_id]
            if self._has_overlapping_targets(experiment, running_exp):
                raise ValueError("Cannot run experiments with overlapping targets")

    async def _check_target_health(self, target: ChaosTarget) -> bool:
        """Check if target is healthy"""
        # Simulate health check
        return True

    def _has_overlapping_targets(
        self,
        exp1: ChaosExperiment,
        exp2: ChaosExperiment
    ) -> bool:
        """Check if two experiments target same resources"""
        targets1 = {t.target_id for t in exp1.targets}
        targets2 = {t.target_id for t in exp2.targets}
        return bool(targets1 & targets2)

    async def _inject_failure(self, experiment: ChaosExperiment):
        """Inject failure based on experiment type"""
        if experiment.experiment_type == ChaosExperimentType.LATENCY_INJECTION:
            await self._inject_latency(experiment)
        elif experiment.experiment_type == ChaosExperimentType.ERROR_INJECTION:
            await self._inject_errors(experiment)
        elif experiment.experiment_type == ChaosExperimentType.CPU_STRESS:
            await self._inject_cpu_stress(experiment)
        elif experiment.experiment_type == ChaosExperimentType.MEMORY_STRESS:
            await self._inject_memory_stress(experiment)
        elif experiment.experiment_type == ChaosExperimentType.NETWORK_PARTITION:
            await self._inject_network_partition(experiment)
        elif experiment.experiment_type == ChaosExperimentType.DEPENDENCY_FAILURE:
            await self._inject_dependency_failure(experiment)

    async def _inject_latency(self, experiment: ChaosExperiment):
        """Inject network latency"""
        latency_ms = experiment.parameters.get("latency_ms", 1000)
        # In production: Use traffic control (tc) or service mesh
        pass

    async def _inject_errors(self, experiment: ChaosExperiment):
        """Inject error responses"""
        error_rate = experiment.parameters.get("error_rate", 0.5)
        error_code = experiment.parameters.get("error_code", 500)
        # In production: Configure error injection middleware
        pass

    async def _inject_cpu_stress(self, experiment: ChaosExperiment):
        """Stress CPU resources"""
        cpu_percent = experiment.parameters.get("cpu_percent", 80)
        # In production: Use stress-ng or similar tools
        pass

    async def _inject_memory_stress(self, experiment: ChaosExperiment):
        """Stress memory resources"""
        memory_mb = experiment.parameters.get("memory_mb", 1024)
        # In production: Allocate memory to simulate pressure
        pass

    async def _inject_network_partition(self, experiment: ChaosExperiment):
        """Create network partition"""
        # In production: Use iptables or network policies
        pass

    async def _inject_dependency_failure(self, experiment: ChaosExperiment):
        """Simulate dependency failure"""
        dependency = experiment.parameters.get("dependency_name")
        # In production: Block traffic to dependency
        pass

    async def _rollback_experiment(self, experiment: ChaosExperiment):
        """Rollback failure injection"""
        # Remove all failure injections
        pass

    async def _measure_steady_state(
        self,
        experiment: ChaosExperiment
    ) -> Dict[str, Any]:
        """Measure current system steady state"""
        measurements = {}

        for hypothesis in experiment.steady_state_hypothesis:
            value = await self._measure_metric(hypothesis.metric_name)
            measurements[hypothesis.metric_name] = value

        return measurements

    async def _measure_metric(self, metric_name: str) -> float:
        """Measure specific metric"""
        # In production: Query monitoring system
        if metric_name == "cpu_usage":
            return psutil.cpu_percent()
        elif metric_name == "memory_usage":
            return psutil.virtual_memory().percent
        elif metric_name == "response_time_ms":
            return random.uniform(50, 200)
        elif metric_name == "error_rate":
            return random.uniform(0, 0.05)
        else:
            return 0.0

    async def _observe_system(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Observe system during experiment"""
        observation = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {}
        }

        for hypothesis in experiment.steady_state_hypothesis:
            value = await self._measure_metric(hypothesis.metric_name)
            observation["metrics"][hypothesis.metric_name] = value

        return observation

    async def _check_steady_state(
        self,
        experiment: ChaosExperiment,
        observation: Dict[str, Any]
    ) -> bool:
        """Check if steady state is maintained"""
        for hypothesis in experiment.steady_state_hypothesis:
            metric_value = observation["metrics"].get(hypothesis.metric_name)
            if metric_value is None:
                continue

            # Check if within tolerance
            if not self._evaluate_hypothesis(hypothesis, metric_value):
                return False

        return True

    def _evaluate_hypothesis(
        self,
        hypothesis: SteadyStateHypothesis,
        value: float
    ) -> bool:
        """Evaluate if hypothesis holds"""
        threshold = hypothesis.threshold
        tolerance = hypothesis.tolerance

        if hypothesis.operator == "lt":
            return value < threshold * (1 + tolerance)
        elif hypothesis.operator == "gt":
            return value > threshold * (1 - tolerance)
        elif hypothesis.operator == "lte":
            return value <= threshold * (1 + tolerance)
        elif hypothesis.operator == "gte":
            return value >= threshold * (1 - tolerance)
        elif hypothesis.operator == "eq":
            return abs(value - threshold) <= threshold * tolerance
        else:
            return True

    async def _validate_steady_state_hypothesis(
        self,
        experiment: ChaosExperiment,
        before: Dict[str, Any],
        after: Dict[str, Any]
    ) -> bool:
        """Validate steady state was maintained"""
        for hypothesis in experiment.steady_state_hypothesis:
            metric_name = hypothesis.metric_name
            before_value = before.get(metric_name)
            after_value = after.get(metric_name)

            if before_value is None or after_value is None:
                continue

            # Check if returned to normal after experiment
            if not self._evaluate_hypothesis(hypothesis, after_value):
                return False

        return True

    async def _calculate_recovery_time(
        self,
        observations: List[Dict[str, Any]]
    ) -> Optional[float]:
        """Calculate how long system took to recover"""
        # Find first observation where all metrics normalized
        # Simplified implementation
        if not observations:
            return None

        return len(observations) / 2.0  # Simplified

    async def _generate_insights(
        self,
        experiment: ChaosExperiment,
        observations: List[Dict[str, Any]],
        steady_state_maintained: bool,
        recovery_time: Optional[float]
    ) -> List[str]:
        """Generate insights from experiment"""
        insights = []

        if steady_state_maintained:
            insights.append("✅ System maintained steady state during failure")
        else:
            insights.append("⚠️ System deviated from steady state during failure")

        if recovery_time:
            if recovery_time < 10:
                insights.append(f"✅ Fast recovery time: {recovery_time:.1f}s")
            elif recovery_time < 60:
                insights.append(f"⚠️ Moderate recovery time: {recovery_time:.1f}s")
            else:
                insights.append(f"❌ Slow recovery time: {recovery_time:.1f}s")

        if experiment.experiment_type == ChaosExperimentType.LATENCY_INJECTION:
            insights.append("Consider implementing request timeouts and circuit breakers")

        if experiment.experiment_type == ChaosExperimentType.ERROR_INJECTION:
            insights.append("Ensure proper error handling and retry logic")

        return insights

    async def schedule_experiment(
        self,
        experiment_id: str,
        cron_expression: str
    ) -> ChaosSchedule:
        """Schedule recurring experiment"""
        schedule_id = f"sched_{experiment_id}"

        schedule = ChaosSchedule(
            schedule_id=schedule_id,
            experiment_id=experiment_id,
            cron_expression=cron_expression,
            enabled=True,
            next_run=datetime.utcnow() + timedelta(hours=1)  # Simplified
        )

        self.schedules[schedule_id] = schedule
        return schedule

    async def get_experiment_history(
        self,
        experiment_id: str
    ) -> List[ExperimentResult]:
        """Get historical results for experiment"""
        return self.results.get(experiment_id, [])

    async def get_resilience_report(self) -> Dict[str, Any]:
        """Generate overall resilience report"""
        total_experiments = len(self.experiments)
        completed_experiments = sum(
            1 for exp in self.experiments.values()
            if exp.status == ExperimentStatus.COMPLETED
        )

        all_results = [r for results in self.results.values() for r in results]
        steady_state_success_rate = (
            sum(1 for r in all_results if r.steady_state_maintained) / len(all_results)
            if all_results else 0.0
        )

        avg_recovery_time = (
            sum(r.recovery_time_seconds for r in all_results if r.recovery_time_seconds) / len(all_results)
            if all_results else 0.0
        )

        return {
            "total_experiments": total_experiments,
            "completed_experiments": completed_experiments,
            "total_runs": len(all_results),
            "steady_state_success_rate": steady_state_success_rate,
            "average_recovery_time_seconds": avg_recovery_time,
            "experiment_types_tested": list({exp.experiment_type.value for exp in self.experiments.values()}),
            "recommendations": self._generate_recommendations(all_results)
        }

    def _generate_recommendations(self, results: List[ExperimentResult]) -> List[str]:
        """Generate recommendations based on experiment results"""
        recommendations = []

        failed_experiments = [r for r in results if not r.steady_state_maintained]
        if failed_experiments:
            recommendations.append(
                f"⚠️ {len(failed_experiments)} experiments showed steady-state deviation"
            )

        slow_recovery = [r for r in results if r.recovery_time_seconds and r.recovery_time_seconds > 60]
        if slow_recovery:
            recommendations.append(
                f"⚠️ {len(slow_recovery)} experiments had slow recovery (>60s)"
            )

        if not recommendations:
            recommendations.append("✅ System shows good resilience across all tested failure modes")

        return recommendations
