"""
ML Pipeline & Model Registry System
====================================
Production-grade ML lifecycle management including:
- Model versioning and artifact storage
- Experiment tracking with metrics/parameters
- Feature store for reproducible ML
- Model deployment with A/B testing
- Model drift detection and auto-retraining
- Serving infrastructure with latency SLOs
- Shadow deployment and canary promotion
- Model lineage and reproducibility
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import statistics
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ModelStage(str, Enum):
    """Lifecycle stage for a model version."""
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    SHADOW = "shadow"
    CANARY = "canary"
    CHALLENGER = "challenger"


class ModelFramework(str, Enum):
    """ML framework used to train the model."""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    ONNX = "onnx"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


class DriftType(str, Enum):
    """Type of model or data drift detected."""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"
    PERFORMANCE_DRIFT = "performance_drift"


class FeatureType(str, Enum):
    """Feature data type for the feature store."""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    EMBEDDING = "embedding"
    TEXT = "text"
    IMAGE = "image"
    TIMESTAMP = "timestamp"


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ModelMetrics:
    """Evaluation metrics for a trained model version."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_roc: float = 0.0
    mse: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    throughput_rps: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    evaluated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "auc_roc": self.auc_roc,
            "mse": self.mse,
            "mae": self.mae,
            "rmse": self.rmse,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "throughput_rps": self.throughput_rps,
            "custom_metrics": self.custom_metrics,
            "evaluated_at": self.evaluated_at.isoformat(),
        }

    def is_better_than(self, other: "ModelMetrics", primary_metric: str = "accuracy") -> bool:
        """Compare this model's metrics against another using a primary metric."""
        self_val = getattr(self, primary_metric, self.custom_metrics.get(primary_metric, 0.0))
        other_val = getattr(other, primary_metric, other.custom_metrics.get(primary_metric, 0.0))
        # For error metrics (lower is better)
        lower_is_better = {"mse", "mae", "rmse", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms"}
        if primary_metric in lower_is_better:
            return self_val < other_val
        return self_val > other_val


@dataclass
class ModelArtifact:
    """Binary artifact for a model (weights, config, preprocessing)."""
    artifact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    artifact_type: str = "weights"  # weights, config, preprocessing, tokenizer
    storage_uri: str = ""
    size_bytes: int = 0
    checksum_sha256: str = ""
    framework: ModelFramework = ModelFramework.CUSTOM
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def verify_integrity(self, data: bytes) -> bool:
        """Verify artifact integrity against stored checksum."""
        computed = hashlib.sha256(data).hexdigest()
        return computed == self.checksum_sha256


@dataclass
class ModelVersion:
    """A specific version of an ML model with full lineage tracking."""
    version_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_name: str = ""
    version_number: str = "1.0.0"
    stage: ModelStage = ModelStage.STAGING
    description: str = ""
    framework: ModelFramework = ModelFramework.CUSTOM
    artifacts: List[ModelArtifact] = field(default_factory=list)
    metrics: Optional[ModelMetrics] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_dataset_uri: str = ""
    feature_schema: Dict[str, str] = field(default_factory=dict)  # feature_name -> type
    tags: Dict[str, str] = field(default_factory=dict)
    parent_version_id: Optional[str] = None
    experiment_run_id: Optional[str] = None
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.utcnow)
    promoted_at: Optional[datetime] = None
    archived_at: Optional[datetime] = None
    serving_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "model_name": self.model_name,
            "version_number": self.version_number,
            "stage": self.stage.value,
            "description": self.description,
            "framework": self.framework.value,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "hyperparameters": self.hyperparameters,
            "feature_schema": self.feature_schema,
            "tags": self.tags,
            "parent_version_id": self.parent_version_id,
            "experiment_run_id": self.experiment_run_id,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "promoted_at": self.promoted_at.isoformat() if self.promoted_at else None,
            "archived_at": self.archived_at.isoformat() if self.archived_at else None,
            "serving_config": self.serving_config,
            "artifact_count": len(self.artifacts),
        }


@dataclass
class ModelDeployment:
    """Live deployment record tracking traffic allocation and health."""
    deployment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_name: str = ""
    version_id: str = ""
    version_number: str = ""
    stage: ModelStage = ModelStage.PRODUCTION
    traffic_percentage: float = 100.0
    min_replicas: int = 1
    max_replicas: int = 10
    current_replicas: int = 1
    cpu_limit_millicores: int = 500
    memory_limit_mb: int = 512
    endpoint_url: str = ""
    health_check_url: str = ""
    deployed_at: datetime = field(default_factory=datetime.utcnow)
    last_health_check: Optional[datetime] = None
    is_healthy: bool = True
    request_count: int = 0
    error_count: int = 0
    avg_latency_ms: float = 0.0
    auto_scaling_enabled: bool = True

    @property
    def error_rate(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.error_count / self.request_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "model_name": self.model_name,
            "version_id": self.version_id,
            "version_number": self.version_number,
            "stage": self.stage.value,
            "traffic_percentage": self.traffic_percentage,
            "current_replicas": self.current_replicas,
            "endpoint_url": self.endpoint_url,
            "is_healthy": self.is_healthy,
            "request_count": self.request_count,
            "error_rate": self.error_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "deployed_at": self.deployed_at.isoformat(),
        }


@dataclass
class ExperimentRun:
    """Single experiment run with parameters, metrics, and artifacts."""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    experiment_id: str = ""
    experiment_name: str = ""
    run_name: str = ""
    status: str = "running"  # running, completed, failed, killed
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, List[Tuple[int, float]]] = field(default_factory=dict)  # step -> value
    tags: Dict[str, str] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    source_code_hash: str = ""
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    hardware_info: Dict[str, Any] = field(default_factory=dict)

    def log_metric(self, name: str, value: float, step: int = 0) -> None:
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append((step, value))

    def get_final_metric(self, name: str) -> Optional[float]:
        if name not in self.metrics or not self.metrics[name]:
            return None
        return self.metrics[name][-1][1]

    def to_dict(self) -> Dict[str, Any]:
        final_metrics = {
            k: v[-1][1] for k, v in self.metrics.items() if v
        }
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "status": self.status,
            "parameters": self.parameters,
            "final_metrics": final_metrics,
            "tags": self.tags,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class Feature:
    """A feature definition in the feature store."""
    feature_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    feature_group: str = ""
    feature_type: FeatureType = FeatureType.NUMERICAL
    description: str = ""
    entity_key: str = ""  # e.g., "user_id", "item_id"
    value_type: str = "float"  # float, int, str, list, dict
    online_enabled: bool = True
    offline_enabled: bool = True
    ttl_seconds: int = 86400  # 1 day default
    tags: Dict[str, str] = field(default_factory=dict)
    statistics: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "name": self.name,
            "feature_group": self.feature_group,
            "feature_type": self.feature_type.value,
            "entity_key": self.entity_key,
            "online_enabled": self.online_enabled,
            "ttl_seconds": self.ttl_seconds,
            "statistics": self.statistics,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class DriftReport:
    """Report from drift detection analysis."""
    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_name: str = ""
    version_id: str = ""
    drift_type: DriftType = DriftType.DATA_DRIFT
    drift_score: float = 0.0
    drift_detected: bool = False
    threshold: float = 0.05
    feature_drift_scores: Dict[str, float] = field(default_factory=dict)
    recommendation: str = ""
    detected_at: datetime = field(default_factory=datetime.utcnow)
    samples_analyzed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "model_name": self.model_name,
            "drift_type": self.drift_type.value,
            "drift_score": self.drift_score,
            "drift_detected": self.drift_detected,
            "threshold": self.threshold,
            "feature_drift_scores": self.feature_drift_scores,
            "recommendation": self.recommendation,
            "detected_at": self.detected_at.isoformat(),
            "samples_analyzed": self.samples_analyzed,
        }


# ---------------------------------------------------------------------------
# Feature Store
# ---------------------------------------------------------------------------

class FeatureStore:
    """
    Online + offline feature store for reproducible ML feature serving.
    Supports point-in-time correct lookups, feature groups, and statistics.
    """

    def __init__(self) -> None:
        self._features: Dict[str, Feature] = {}
        self._online_store: Dict[str, Dict[str, Any]] = {}  # entity_key:value -> feature_name -> value
        self._offline_store: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._feature_groups: Dict[str, List[str]] = defaultdict(list)
        self._access_log: deque = deque(maxlen=10000)
        self._lock = asyncio.Lock()

    async def register_feature(self, feature: Feature) -> Feature:
        """Register a new feature definition."""
        async with self._lock:
            self._features[feature.name] = feature
            self._feature_groups[feature.feature_group].append(feature.name)
            logger.info("Registered feature: %s in group %s", feature.name, feature.feature_group)
            return feature

    async def materialize_features(
        self,
        feature_group: str,
        entity_key: str,
        entity_value: str,
        feature_values: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Write feature values to the online store."""
        async with self._lock:
            store_key = f"{entity_key}:{entity_value}"
            if store_key not in self._online_store:
                self._online_store[store_key] = {}

            expiry = datetime.utcnow() + timedelta(seconds=ttl_seconds or 86400)
            for fname, fval in feature_values.items():
                self._online_store[store_key][fname] = {
                    "value": fval,
                    "updated_at": datetime.utcnow().isoformat(),
                    "expires_at": expiry.isoformat(),
                }

            # Offline store with timestamp for point-in-time correctness
            self._offline_store[store_key].append({
                "feature_group": feature_group,
                "values": feature_values,
                "timestamp": datetime.utcnow().isoformat(),
            })

    async def get_online_features(
        self,
        feature_names: List[str],
        entity_key: str,
        entity_value: str,
    ) -> Dict[str, Any]:
        """Retrieve feature values for real-time inference."""
        store_key = f"{entity_key}:{entity_value}"
        result: Dict[str, Any] = {}
        store = self._online_store.get(store_key, {})
        now = datetime.utcnow()

        for fname in feature_names:
            if fname in store:
                entry = store[fname]
                expires_at = datetime.fromisoformat(entry["expires_at"])
                if now <= expires_at:
                    result[fname] = entry["value"]
                else:
                    result[fname] = None  # Expired
            else:
                result[fname] = None  # Not found

        self._access_log.append({
            "entity_key": entity_key,
            "entity_value": entity_value,
            "features": feature_names,
            "hit_count": sum(1 for v in result.values() if v is not None),
            "timestamp": now.isoformat(),
        })
        return result

    async def get_historical_features(
        self,
        feature_names: List[str],
        entity_key: str,
        entity_value: str,
        as_of: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Point-in-time correct feature retrieval from offline store."""
        store_key = f"{entity_key}:{entity_value}"
        history = self._offline_store.get(store_key, [])
        cutoff = as_of or datetime.utcnow()

        filtered = []
        for entry in history:
            ts = datetime.fromisoformat(entry["timestamp"])
            if ts <= cutoff:
                row = {"timestamp": entry["timestamp"]}
                for fname in feature_names:
                    row[fname] = entry["values"].get(fname)
                filtered.append(row)

        return filtered

    async def compute_feature_statistics(self, feature_name: str) -> Dict[str, float]:
        """Compute basic statistics for a feature across all entities."""
        values = []
        for store in self._online_store.values():
            if feature_name in store:
                val = store[feature_name].get("value")
                if isinstance(val, (int, float)):
                    values.append(float(val))

        if not values:
            return {}

        stats = {
            "count": len(values),
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
        if len(values) > 0:
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            stats["p50"] = sorted_vals[n // 2]
            stats["p95"] = sorted_vals[int(n * 0.95)]
            stats["p99"] = sorted_vals[int(n * 0.99)]

        # Update feature statistics
        if feature_name in self._features:
            self._features[feature_name].statistics = stats
            self._features[feature_name].last_updated = datetime.utcnow()

        return stats

    async def list_features(self, feature_group: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all registered features, optionally filtered by group."""
        features = self._features.values()
        if feature_group:
            features = [f for f in features if f.feature_group == feature_group]
        return [f.to_dict() for f in features]

    async def get_feature_groups(self) -> Dict[str, List[str]]:
        return dict(self._feature_groups)


# ---------------------------------------------------------------------------
# Experiment Tracker
# ---------------------------------------------------------------------------

class ExperimentTracker:
    """
    MLflow-compatible experiment tracking system.
    Tracks runs, parameters, metrics, and artifacts with full lineage.
    """

    def __init__(self) -> None:
        self._experiments: Dict[str, Dict[str, Any]] = {}
        self._runs: Dict[str, ExperimentRun] = {}
        self._active_runs: Dict[str, str] = {}  # caller_id -> run_id
        self._lock = asyncio.Lock()

    async def create_experiment(
        self,
        name: str,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Create a new experiment and return its ID."""
        async with self._lock:
            experiment_id = str(uuid.uuid4())
            self._experiments[experiment_id] = {
                "experiment_id": experiment_id,
                "name": name,
                "description": description,
                "tags": tags or {},
                "created_at": datetime.utcnow().isoformat(),
                "run_count": 0,
                "best_run_id": None,
                "primary_metric": "accuracy",
            }
            logger.info("Created experiment: %s (%s)", name, experiment_id)
            return experiment_id

    async def start_run(
        self,
        experiment_id: str,
        run_name: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> ExperimentRun:
        """Start a new experiment run."""
        async with self._lock:
            if experiment_id not in self._experiments:
                raise ValueError(f"Experiment {experiment_id} not found")

            exp = self._experiments[experiment_id]
            run = ExperimentRun(
                experiment_id=experiment_id,
                experiment_name=exp["name"],
                run_name=run_name or f"run_{exp['run_count'] + 1}",
                parameters=parameters or {},
                tags=tags or {},
                status="running",
            )
            self._runs[run.run_id] = run
            exp["run_count"] += 1
            logger.info("Started run %s in experiment %s", run.run_id, experiment_id)
            return run

    async def log_metric(
        self, run_id: str, name: str, value: float, step: int = 0
    ) -> None:
        """Log a metric value for a run."""
        run = self._runs.get(run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")
        run.log_metric(name, value, step)

    async def log_params(self, run_id: str, params: Dict[str, Any]) -> None:
        """Batch log parameters for a run."""
        run = self._runs.get(run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")
        run.parameters.update(params)

    async def end_run(self, run_id: str, status: str = "completed") -> ExperimentRun:
        """End a run and compute duration."""
        async with self._lock:
            run = self._runs.get(run_id)
            if not run:
                raise ValueError(f"Run {run_id} not found")
            run.status = status
            run.end_time = datetime.utcnow()
            run.duration_seconds = (run.end_time - run.start_time).total_seconds()

            # Update experiment best run
            exp = self._experiments.get(run.experiment_id)
            if exp and status == "completed":
                primary = exp.get("primary_metric", "accuracy")
                if exp["best_run_id"] is None:
                    exp["best_run_id"] = run_id
                else:
                    best = self._runs.get(exp["best_run_id"])
                    if best:
                        current_val = run.get_final_metric(primary) or 0.0
                        best_val = best.get_final_metric(primary) or 0.0
                        if current_val > best_val:
                            exp["best_run_id"] = run_id

            logger.info("Ended run %s with status %s (%.1fs)", run_id, status, run.duration_seconds)
            return run

    async def compare_runs(self, run_ids: List[str], metrics: List[str]) -> List[Dict[str, Any]]:
        """Compare multiple runs across specified metrics."""
        results = []
        for run_id in run_ids:
            run = self._runs.get(run_id)
            if not run:
                continue
            row: Dict[str, Any] = {"run_id": run_id, "run_name": run.run_name}
            for metric in metrics:
                row[metric] = run.get_final_metric(metric)
            row["parameters"] = run.parameters
            results.append(row)
        return results

    async def get_best_run(self, experiment_id: str, metric: str = "accuracy") -> Optional[ExperimentRun]:
        """Get the best run in an experiment by a given metric."""
        exp_runs = [r for r in self._runs.values() if r.experiment_id == experiment_id and r.status == "completed"]
        if not exp_runs:
            return None
        lower_is_better = {"mse", "mae", "rmse", "loss"}
        reverse = metric not in lower_is_better
        return max(exp_runs, key=lambda r: (r.get_final_metric(metric) or 0.0) * (1 if reverse else -1))

    async def list_runs(self, experiment_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """List recent runs for an experiment."""
        runs = [r for r in self._runs.values() if r.experiment_id == experiment_id]
        runs.sort(key=lambda r: r.start_time, reverse=True)
        return [r.to_dict() for r in runs[:limit]]


# ---------------------------------------------------------------------------
# Drift Detector
# ---------------------------------------------------------------------------

class DriftDetector:
    """
    Statistical drift detection using Population Stability Index (PSI) and KL divergence.
    Monitors both data drift and prediction drift.
    """

    def __init__(self, drift_threshold: float = 0.2) -> None:
        self._drift_threshold = drift_threshold
        self._reference_distributions: Dict[str, List[float]] = {}
        self._prediction_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

    def register_reference_distribution(self, feature_name: str, values: List[float]) -> None:
        """Register the reference distribution for a feature from training data."""
        self._reference_distributions[feature_name] = values
        logger.info("Registered reference distribution for feature: %s", feature_name)

    def _compute_psi(self, reference: List[float], current: List[float], buckets: int = 10) -> float:
        """Compute Population Stability Index (PSI) between two distributions."""
        if not reference or not current:
            return 0.0

        min_val = min(min(reference), min(current))
        max_val = max(max(reference), max(current))
        if min_val == max_val:
            return 0.0

        bucket_width = (max_val - min_val) / buckets
        ref_hist = [0] * buckets
        cur_hist = [0] * buckets

        for val in reference:
            idx = min(int((val - min_val) / bucket_width), buckets - 1)
            ref_hist[idx] += 1

        for val in current:
            idx = min(int((val - min_val) / bucket_width), buckets - 1)
            cur_hist[idx] += 1

        ref_total = len(reference)
        cur_total = len(current)
        psi = 0.0
        epsilon = 1e-8

        for r, c in zip(ref_hist, cur_hist):
            ref_pct = (r / ref_total) + epsilon
            cur_pct = (c / cur_total) + epsilon
            psi += (cur_pct - ref_pct) * math.log(cur_pct / ref_pct)

        return abs(psi)

    async def detect_data_drift(
        self, model_name: str, feature_values: Dict[str, List[float]]
    ) -> DriftReport:
        """Detect data drift for incoming feature distributions."""
        feature_drift_scores: Dict[str, float] = {}
        max_drift = 0.0

        for fname, values in feature_values.items():
            ref = self._reference_distributions.get(fname, [])
            if ref and values:
                psi = self._compute_psi(ref, values)
                feature_drift_scores[fname] = psi
                max_drift = max(max_drift, psi)

        drift_detected = max_drift > self._drift_threshold
        recommendation = ""
        if drift_detected:
            if max_drift > 0.5:
                recommendation = "Critical drift detected. Immediate retraining recommended."
            elif max_drift > 0.25:
                recommendation = "Significant drift. Schedule retraining within 24 hours."
            else:
                recommendation = "Moderate drift. Monitor closely and retrain within 1 week."

        report = DriftReport(
            model_name=model_name,
            drift_type=DriftType.DATA_DRIFT,
            drift_score=max_drift,
            drift_detected=drift_detected,
            threshold=self._drift_threshold,
            feature_drift_scores=feature_drift_scores,
            recommendation=recommendation,
            samples_analyzed=sum(len(v) for v in feature_values.values()),
        )
        logger.info("Drift detection for %s: score=%.3f, detected=%s", model_name, max_drift, drift_detected)
        return report

    async def detect_prediction_drift(
        self, model_name: str, predictions: List[float], version_id: str
    ) -> DriftReport:
        """Detect drift in model prediction distribution."""
        self._prediction_history[model_name].extend(predictions)
        history = list(self._prediction_history[model_name])

        if len(history) < 100:
            return DriftReport(
                model_name=model_name,
                version_id=version_id,
                drift_type=DriftType.PREDICTION_DRIFT,
                drift_detected=False,
                recommendation="Insufficient data for drift detection (need >= 100 samples)",
            )

        # Compare first half vs second half as reference vs current
        mid = len(history) // 2
        reference = history[:mid]
        current = history[mid:]
        psi = self._compute_psi(reference, current)
        drift_detected = psi > self._drift_threshold

        return DriftReport(
            model_name=model_name,
            version_id=version_id,
            drift_type=DriftType.PREDICTION_DRIFT,
            drift_score=psi,
            drift_detected=drift_detected,
            threshold=self._drift_threshold,
            recommendation="Retrain model due to prediction distribution shift." if drift_detected else "",
            samples_analyzed=len(predictions),
        )


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """
    Central model registry with versioning, promotion workflows, A/B deployments,
    drift detection, and serving infrastructure management.
    """

    def __init__(self) -> None:
        self._models: Dict[str, List[ModelVersion]] = defaultdict(list)
        self._deployments: Dict[str, List[ModelDeployment]] = defaultdict(list)
        self._drift_detector = DriftDetector()
        self._prediction_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self._serving_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._lock = asyncio.Lock()
        self._registered_models: Dict[str, Dict[str, Any]] = {}

    async def register_model(
        self,
        name: str,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
        owner: str = "system",
    ) -> Dict[str, Any]:
        """Register a new model in the registry."""
        async with self._lock:
            if name in self._registered_models:
                raise ValueError(f"Model '{name}' already registered")
            self._registered_models[name] = {
                "name": name,
                "description": description,
                "tags": tags or {},
                "owner": owner,
                "created_at": datetime.utcnow().isoformat(),
                "version_count": 0,
                "latest_version": None,
                "production_version": None,
            }
            logger.info("Registered model: %s", name)
            return self._registered_models[name]

    async def create_version(
        self,
        model_name: str,
        framework: ModelFramework,
        hyperparameters: Dict[str, Any],
        feature_schema: Dict[str, str],
        metrics: Optional[ModelMetrics] = None,
        description: str = "",
        training_dataset_uri: str = "",
        experiment_run_id: Optional[str] = None,
        created_by: str = "system",
        tags: Optional[Dict[str, str]] = None,
    ) -> ModelVersion:
        """Create a new version for a registered model."""
        async with self._lock:
            if model_name not in self._registered_models:
                await self.register_model(model_name)

            existing = self._models[model_name]
            version_num = len(existing) + 1
            version = ModelVersion(
                model_name=model_name,
                version_number=f"{version_num}.0.0",
                stage=ModelStage.STAGING,
                framework=framework,
                hyperparameters=hyperparameters,
                feature_schema=feature_schema,
                metrics=metrics,
                description=description,
                training_dataset_uri=training_dataset_uri,
                experiment_run_id=experiment_run_id,
                created_by=created_by,
                tags=tags or {},
                parent_version_id=existing[-1].version_id if existing else None,
            )
            self._models[model_name].append(version)
            self._registered_models[model_name]["version_count"] = len(self._models[model_name])
            self._registered_models[model_name]["latest_version"] = version.version_number
            logger.info("Created version %s for model %s", version.version_number, model_name)
            return version

    async def promote_version(
        self,
        model_name: str,
        version_id: str,
        target_stage: ModelStage,
        traffic_percentage: float = 100.0,
    ) -> ModelVersion:
        """Promote a model version to a new stage (e.g., staging -> production)."""
        async with self._lock:
            version = self._find_version(model_name, version_id)
            if not version:
                raise ValueError(f"Version {version_id} not found for model {model_name}")

            previous_stage = version.stage

            # Archive current production if promoting to production
            if target_stage == ModelStage.PRODUCTION:
                for v in self._models[model_name]:
                    if v.stage == ModelStage.PRODUCTION and v.version_id != version_id:
                        v.stage = ModelStage.ARCHIVED
                        v.archived_at = datetime.utcnow()
                self._registered_models[model_name]["production_version"] = version.version_number

            version.stage = target_stage
            version.promoted_at = datetime.utcnow()

            # Create deployment record
            deployment = ModelDeployment(
                model_name=model_name,
                version_id=version_id,
                version_number=version.version_number,
                stage=target_stage,
                traffic_percentage=traffic_percentage,
                endpoint_url=f"/models/{model_name}/versions/{version.version_number}/predict",
                deployed_at=datetime.utcnow(),
            )
            self._deployments[model_name].append(deployment)
            logger.info(
                "Promoted model %s version %s: %s -> %s (traffic=%.0f%%)",
                model_name, version.version_number,
                previous_stage.value, target_stage.value, traffic_percentage,
            )
            return version

    async def setup_ab_test(
        self,
        model_name: str,
        champion_version_id: str,
        challenger_version_id: str,
        challenger_traffic_pct: float = 10.0,
    ) -> Dict[str, Any]:
        """Configure A/B test between champion and challenger models."""
        async with self._lock:
            champion = self._find_version(model_name, champion_version_id)
            challenger = self._find_version(model_name, challenger_version_id)

            if not champion or not challenger:
                raise ValueError("Champion or challenger version not found")

            champion_traffic = 100.0 - challenger_traffic_pct

            # Update deployments
            for dep in self._deployments[model_name]:
                if dep.version_id == champion_version_id:
                    dep.traffic_percentage = champion_traffic
                elif dep.version_id == challenger_version_id:
                    dep.traffic_percentage = challenger_traffic_pct

            # Set challenger stage
            challenger.stage = ModelStage.CHALLENGER

            ab_config = {
                "model_name": model_name,
                "champion": {
                    "version_id": champion_version_id,
                    "version_number": champion.version_number,
                    "traffic_pct": champion_traffic,
                },
                "challenger": {
                    "version_id": challenger_version_id,
                    "version_number": challenger.version_number,
                    "traffic_pct": challenger_traffic_pct,
                },
                "started_at": datetime.utcnow().isoformat(),
                "status": "running",
            }
            logger.info(
                "A/B test configured for %s: %s (%.0f%%) vs %s (%.0f%%)",
                model_name, champion.version_number, champion_traffic,
                challenger.version_number, challenger_traffic_pct,
            )
            return ab_config

    async def record_prediction(
        self,
        model_name: str,
        version_id: str,
        prediction: Any,
        ground_truth: Optional[Any] = None,
        latency_ms: float = 0.0,
        features: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a prediction for drift detection and performance monitoring."""
        record = {
            "prediction": prediction,
            "ground_truth": ground_truth,
            "latency_ms": latency_ms,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._prediction_cache[f"{model_name}:{version_id}"].append(record)

        # Update serving metrics
        key = f"{model_name}:{version_id}"
        metrics = self._serving_metrics[key]
        metrics.setdefault("total_predictions", 0)
        metrics.setdefault("latency_sum", 0.0)
        metrics["total_predictions"] += 1
        metrics["latency_sum"] += latency_ms
        metrics["avg_latency_ms"] = metrics["latency_sum"] / metrics["total_predictions"]

        # Update deployment stats
        for dep in self._deployments[model_name]:
            if dep.version_id == version_id:
                dep.request_count += 1
                dep.avg_latency_ms = metrics["avg_latency_ms"]

    async def check_drift(
        self,
        model_name: str,
        version_id: str,
        feature_samples: Dict[str, List[float]],
    ) -> DriftReport:
        """Run drift detection for a deployed model."""
        return await self._drift_detector.detect_data_drift(model_name, feature_samples)

    async def get_model_lineage(self, model_name: str, version_id: str) -> List[Dict[str, Any]]:
        """Trace the full lineage chain for a model version."""
        lineage = []
        current_id = version_id
        max_depth = 20
        depth = 0

        while current_id and depth < max_depth:
            version = self._find_version(model_name, current_id)
            if not version:
                break
            lineage.append(version.to_dict())
            current_id = version.parent_version_id
            depth += 1

        return lineage

    async def get_serving_stats(self, model_name: str) -> Dict[str, Any]:
        """Get real-time serving statistics for all deployed versions."""
        deployments = self._deployments.get(model_name, [])
        stats = {
            "model_name": model_name,
            "total_deployments": len(deployments),
            "versions": [],
        }
        for dep in deployments:
            key = f"{model_name}:{dep.version_id}"
            serving = self._serving_metrics.get(key, {})
            stats["versions"].append({
                **dep.to_dict(),
                "serving_metrics": serving,
            })
        return stats

    async def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models."""
        return list(self._registered_models.values())

    async def list_versions(
        self, model_name: str, stage: Optional[ModelStage] = None
    ) -> List[Dict[str, Any]]:
        """List versions for a model, optionally filtered by stage."""
        versions = self._models.get(model_name, [])
        if stage:
            versions = [v for v in versions if v.stage == stage]
        return [v.to_dict() for v in versions]

    async def get_production_version(self, model_name: str) -> Optional[ModelVersion]:
        """Get the current production version of a model."""
        for v in self._models.get(model_name, []):
            if v.stage == ModelStage.PRODUCTION:
                return v
        return None

    async def archive_version(self, model_name: str, version_id: str) -> ModelVersion:
        """Archive a model version."""
        async with self._lock:
            version = self._find_version(model_name, version_id)
            if not version:
                raise ValueError(f"Version {version_id} not found")
            version.stage = ModelStage.ARCHIVED
            version.archived_at = datetime.utcnow()
            return version

    def _find_version(self, model_name: str, version_id: str) -> Optional[ModelVersion]:
        for v in self._models.get(model_name, []):
            if v.version_id == version_id:
                return v
        return None

    async def get_registry_summary(self) -> Dict[str, Any]:
        """Get a high-level summary of the model registry."""
        total_models = len(self._registered_models)
        total_versions = sum(len(vs) for vs in self._models.values())
        production_count = sum(
            1 for vs in self._models.values()
            for v in vs if v.stage == ModelStage.PRODUCTION
        )
        total_predictions = sum(
            m.get("total_predictions", 0) for m in self._serving_metrics.values()
        )
        return {
            "total_models": total_models,
            "total_versions": total_versions,
            "production_models": production_count,
            "total_predictions_served": total_predictions,
            "registry_health": "healthy",
        }
