"""
Intelligent Data Mesh — Data virtualization, catalog management, lineage
tracking, quality scoring, domain ownership, and federated data access.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


# ─────────────────────────────────── Enums ───────────────────────────────────


class DataDomain(str, Enum):
    USERS = "users"
    ORDERS = "orders"
    PRODUCTS = "products"
    ANALYTICS = "analytics"
    FINANCE = "finance"
    OPERATIONS = "operations"
    INFRASTRUCTURE = "infrastructure"
    ML_FEATURES = "ml_features"
    COMPLIANCE = "compliance"
    CUSTOM = "custom"


class DataAssetType(str, Enum):
    TABLE = "table"
    VIEW = "view"
    STREAM = "stream"
    API_ENDPOINT = "api_endpoint"
    FILE = "file"
    ML_DATASET = "ml_dataset"
    FEATURE_STORE = "feature_store"
    METRIC = "metric"


class DataQualityDimension(str, Enum):
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    UNIQUENESS = "uniqueness"
    VALIDITY = "validity"


class LineageNodeType(str, Enum):
    SOURCE = "source"
    TRANSFORMATION = "transformation"
    SINK = "sink"
    ENRICHMENT = "enrichment"
    AGGREGATION = "aggregation"


class AccessPolicy(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"
    TOP_SECRET = "top_secret"


# ────────────────────────────── Data structures ──────────────────────────────


@dataclass
class SchemaField:
    name: str
    data_type: str
    nullable: bool
    description: str
    tags: List[str] = field(default_factory=list)
    pii: bool = False
    primary_key: bool = False
    foreign_key: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "data_type": self.data_type,
            "nullable": self.nullable,
            "description": self.description,
            "tags": self.tags,
            "pii": self.pii,
            "primary_key": self.primary_key,
            "foreign_key": self.foreign_key,
        }


@dataclass
class DataAsset:
    asset_id: str
    name: str
    description: str
    domain: DataDomain
    asset_type: DataAssetType
    owner_team: str
    schema_fields: List[SchemaField]
    access_policy: AccessPolicy
    tags: List[str]
    quality_score: float
    row_count: Optional[int]
    size_bytes: Optional[int]
    update_frequency: str
    data_source: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "name": self.name,
            "description": self.description,
            "domain": self.domain.value,
            "asset_type": self.asset_type.value,
            "owner_team": self.owner_team,
            "schema_fields": [f.to_dict() for f in self.schema_fields],
            "field_count": len(self.schema_fields),
            "pii_fields": sum(1 for f in self.schema_fields if f.pii),
            "access_policy": self.access_policy.value,
            "tags": self.tags,
            "quality_score": self.quality_score,
            "row_count": self.row_count,
            "size_bytes": self.size_bytes,
            "update_frequency": self.update_frequency,
            "data_source": self.data_source,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class LineageNode:
    node_id: str
    name: str
    node_type: LineageNodeType
    asset_id: Optional[str]
    upstream_ids: List[str]
    downstream_ids: List[str]
    transformation_logic: Optional[str]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "node_type": self.node_type.value,
            "asset_id": self.asset_id,
            "upstream_ids": self.upstream_ids,
            "downstream_ids": self.downstream_ids,
            "transformation_logic": self.transformation_logic,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class QualityReport:
    report_id: str
    asset_id: str
    dimension_scores: Dict[str, float]
    overall_score: float
    issues: List[Dict[str, Any]]
    passed_rules: int
    failed_rules: int
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "asset_id": self.asset_id,
            "dimension_scores": self.dimension_scores,
            "overall_score": self.overall_score,
            "issues": self.issues,
            "passed_rules": self.passed_rules,
            "failed_rules": self.failed_rules,
            "evaluated_at": self.evaluated_at.isoformat(),
        }


@dataclass
class DataContract:
    contract_id: str
    producer_asset_id: str
    consumer_team: str
    sla_freshness_minutes: int
    sla_availability_pct: float
    schema_version: str
    agreed_fields: List[str]
    notifications_enabled: bool
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "producer_asset_id": self.producer_asset_id,
            "consumer_team": self.consumer_team,
            "sla_freshness_minutes": self.sla_freshness_minutes,
            "sla_availability_pct": self.sla_availability_pct,
            "schema_version": self.schema_version,
            "agreed_fields": self.agreed_fields,
            "notifications_enabled": self.notifications_enabled,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


# ─────────────────────────── Data Catalog ───────────────────────────────────


class DataCatalog:
    """
    Central catalog for discovering, documenting, and managing data assets
    across multiple domains with search and recommendation capabilities.
    """

    def __init__(self):
        self._assets: Dict[str, DataAsset] = {}
        self._domain_index: Dict[DataDomain, List[str]] = defaultdict(list)
        self._tag_index: Dict[str, List[str]] = defaultdict(list)
        self._owner_index: Dict[str, List[str]] = defaultdict(list)
        self._search_index: Dict[str, Set[str]] = defaultdict(set)

    def register_asset(self, asset: DataAsset) -> None:
        self._assets[asset.asset_id] = asset
        self._domain_index[asset.domain].append(asset.asset_id)
        self._owner_index[asset.owner_team].append(asset.asset_id)
        for tag in asset.tags:
            self._tag_index[tag].append(asset.asset_id)
        # Index words for search
        for word in (asset.name + " " + asset.description).lower().split():
            if len(word) > 2:
                self._search_index[word].add(asset.asset_id)

    def search(
        self,
        query: str,
        domain: Optional[DataDomain] = None,
        asset_type: Optional[DataAssetType] = None,
        access_policy: Optional[AccessPolicy] = None,
        min_quality: float = 0.0,
        limit: int = 20,
    ) -> List[DataAsset]:
        query_words = query.lower().split()
        # Score assets by query match
        scores: Dict[str, int] = defaultdict(int)
        for word in query_words:
            for asset_id in self._search_index.get(word, set()):
                scores[asset_id] += 1

        # Apply filters
        candidates = []
        for asset_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            asset = self._assets.get(asset_id)
            if asset is None:
                continue
            if domain and asset.domain != domain:
                continue
            if asset_type and asset.asset_type != asset_type:
                continue
            if access_policy and asset.access_policy != access_policy:
                continue
            if asset.quality_score < min_quality:
                continue
            candidates.append(asset)
            if len(candidates) >= limit:
                break
        return candidates

    def get_asset(self, asset_id: str) -> Optional[DataAsset]:
        return self._assets.get(asset_id)

    def get_by_domain(self, domain: DataDomain) -> List[DataAsset]:
        ids = self._domain_index.get(domain, [])
        return [self._assets[aid] for aid in ids if aid in self._assets]

    def get_by_tag(self, tag: str) -> List[DataAsset]:
        ids = self._tag_index.get(tag, [])
        return [self._assets[aid] for aid in ids if aid in self._assets]

    def get_by_owner(self, owner_team: str) -> List[DataAsset]:
        ids = self._owner_index.get(owner_team, [])
        return [self._assets[aid] for aid in ids if aid in self._assets]

    def recommend_related(
        self, asset_id: str, top_k: int = 5
    ) -> List[DataAsset]:
        asset = self._assets.get(asset_id)
        if asset is None:
            return []
        # Score other assets by shared tags and domain
        scores: Dict[str, float] = defaultdict(float)
        for tag in asset.tags:
            for other_id in self._tag_index.get(tag, []):
                if other_id != asset_id:
                    scores[other_id] += 0.5
        for other_id in self._domain_index.get(asset.domain, []):
            if other_id != asset_id:
                scores[other_id] += 0.3
        top_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]
        return [self._assets[aid] for aid in top_ids if aid in self._assets]

    def get_catalog_stats(self) -> Dict[str, Any]:
        assets = list(self._assets.values())
        by_domain: Dict[str, int] = defaultdict(int)
        by_type: Dict[str, int] = defaultdict(int)
        by_policy: Dict[str, int] = defaultdict(int)
        quality_sum = 0.0
        pii_count = 0
        for a in assets:
            by_domain[a.domain.value] += 1
            by_type[a.asset_type.value] += 1
            by_policy[a.access_policy.value] += 1
            quality_sum += a.quality_score
            pii_count += sum(1 for f in a.schema_fields if f.pii)
        return {
            "total_assets": len(assets),
            "by_domain": dict(by_domain),
            "by_type": dict(by_type),
            "by_access_policy": dict(by_policy),
            "avg_quality_score": quality_sum / max(len(assets), 1),
            "total_pii_fields": pii_count,
            "total_tags": len(self._tag_index),
        }


# ─────────────────────────── Lineage Tracker ────────────────────────────────


class DataLineageTracker:
    """
    Tracks data lineage as a directed acyclic graph from source systems
    through transformations to downstream consumers.
    """

    def __init__(self):
        self._nodes: Dict[str, LineageNode] = {}
        self._asset_to_node: Dict[str, str] = {}

    def register_node(self, node: LineageNode) -> None:
        self._nodes[node.node_id] = node
        if node.asset_id:
            self._asset_to_node[node.asset_id] = node.node_id

    def add_edge(self, upstream_id: str, downstream_id: str) -> bool:
        upstream = self._nodes.get(upstream_id)
        downstream = self._nodes.get(downstream_id)
        if upstream is None or downstream is None:
            return False
        if downstream_id not in upstream.downstream_ids:
            upstream.downstream_ids.append(downstream_id)
        if upstream_id not in downstream.upstream_ids:
            downstream.upstream_ids.append(upstream_id)
        return True

    def get_upstream_lineage(
        self, node_id: str, max_depth: int = 5
    ) -> List[Dict[str, Any]]:
        visited: Set[str] = set()
        result: List[Dict[str, Any]] = []
        self._traverse_upstream(node_id, visited, result, 0, max_depth)
        return result

    def get_downstream_lineage(
        self, node_id: str, max_depth: int = 5
    ) -> List[Dict[str, Any]]:
        visited: Set[str] = set()
        result: List[Dict[str, Any]] = []
        self._traverse_downstream(node_id, visited, result, 0, max_depth)
        return result

    def get_impact_analysis(self, node_id: str) -> Dict[str, Any]:
        downstream = self.get_downstream_lineage(node_id, max_depth=10)
        upstream = self.get_upstream_lineage(node_id, max_depth=10)
        return {
            "node_id": node_id,
            "directly_impacted": [
                n["node_id"]
                for n in downstream
                if n.get("depth", 999) == 1
            ],
            "total_downstream": len(downstream),
            "total_upstream": len(upstream),
            "critical_path": self._find_critical_path(node_id),
        }

    def _traverse_upstream(
        self,
        node_id: str,
        visited: Set[str],
        result: List[Dict[str, Any]],
        depth: int,
        max_depth: int,
    ) -> None:
        if depth >= max_depth or node_id in visited:
            return
        visited.add(node_id)
        node = self._nodes.get(node_id)
        if node is None:
            return
        entry = node.to_dict()
        entry["depth"] = depth
        result.append(entry)
        for upstream_id in node.upstream_ids:
            self._traverse_upstream(upstream_id, visited, result, depth + 1, max_depth)

    def _traverse_downstream(
        self,
        node_id: str,
        visited: Set[str],
        result: List[Dict[str, Any]],
        depth: int,
        max_depth: int,
    ) -> None:
        if depth >= max_depth or node_id in visited:
            return
        visited.add(node_id)
        node = self._nodes.get(node_id)
        if node is None:
            return
        entry = node.to_dict()
        entry["depth"] = depth
        result.append(entry)
        for downstream_id in node.downstream_ids:
            self._traverse_downstream(downstream_id, visited, result, depth + 1, max_depth)

    def _find_critical_path(self, node_id: str) -> List[str]:
        # Simplified: find longest path downstream
        node = self._nodes.get(node_id)
        if node is None:
            return [node_id]
        if not node.downstream_ids:
            return [node_id]
        longest: List[str] = [node_id]
        for downstream_id in node.downstream_ids:
            sub_path = self._find_critical_path(downstream_id)
            if len(sub_path) + 1 > len(longest):
                longest = [node_id] + sub_path
        return longest

    def get_lineage_stats(self) -> Dict[str, Any]:
        nodes = list(self._nodes.values())
        by_type: Dict[str, int] = defaultdict(int)
        for n in nodes:
            by_type[n.node_type.value] += 1
        total_edges = sum(len(n.downstream_ids) for n in nodes)
        return {
            "total_nodes": len(nodes),
            "by_type": dict(by_type),
            "total_edges": total_edges,
        }


# ─────────────────────── Data Quality Engine ────────────────────────────────


class DataQualityEngine:
    """
    Evaluates data quality across multiple dimensions with configurable rules,
    automated profiling, and anomaly detection.
    """

    def __init__(self):
        self._rules: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._reports: Dict[str, List[QualityReport]] = defaultdict(list)

    def add_rule(
        self,
        asset_id: str,
        dimension: DataQualityDimension,
        rule_name: str,
        rule_config: Dict[str, Any],
    ) -> str:
        rule_id = str(uuid.uuid4())
        self._rules[asset_id].append(
            {
                "rule_id": rule_id,
                "dimension": dimension.value,
                "rule_name": rule_name,
                "config": rule_config,
                "enabled": True,
            }
        )
        return rule_id

    def evaluate_asset(
        self,
        asset: DataAsset,
        sample_data: Optional[List[Dict[str, Any]]] = None,
    ) -> QualityReport:
        dimension_scores: Dict[str, float] = {}
        issues: List[Dict[str, Any]] = []
        passed = 0
        failed = 0

        # Completeness: check nullable fields vs sample data
        completeness = self._evaluate_completeness(asset, sample_data, issues)
        dimension_scores[DataQualityDimension.COMPLETENESS.value] = completeness
        passed += 1 if completeness >= 0.8 else 0
        failed += 0 if completeness >= 0.8 else 1

        # Accuracy: schema validation
        accuracy = self._evaluate_accuracy(asset, sample_data, issues)
        dimension_scores[DataQualityDimension.ACCURACY.value] = accuracy
        passed += 1 if accuracy >= 0.8 else 0
        failed += 0 if accuracy >= 0.8 else 1

        # Consistency: cross-field rules
        consistency = self._evaluate_consistency(asset, issues)
        dimension_scores[DataQualityDimension.CONSISTENCY.value] = consistency
        passed += 1 if consistency >= 0.8 else 0
        failed += 0 if consistency >= 0.8 else 1

        # Timeliness: based on update_frequency
        timeliness = self._evaluate_timeliness(asset, issues)
        dimension_scores[DataQualityDimension.TIMELINESS.value] = timeliness
        passed += 1 if timeliness >= 0.8 else 0
        failed += 0 if timeliness >= 0.8 else 1

        # Uniqueness: primary key check
        uniqueness = self._evaluate_uniqueness(asset, sample_data, issues)
        dimension_scores[DataQualityDimension.UNIQUENESS.value] = uniqueness
        passed += 1 if uniqueness >= 0.8 else 0
        failed += 0 if uniqueness >= 0.8 else 1

        # Validity: data type enforcement
        validity = self._evaluate_validity(asset, sample_data, issues)
        dimension_scores[DataQualityDimension.VALIDITY.value] = validity
        passed += 1 if validity >= 0.8 else 0
        failed += 0 if validity >= 0.8 else 1

        overall = sum(dimension_scores.values()) / len(dimension_scores)
        report = QualityReport(
            report_id=str(uuid.uuid4()),
            asset_id=asset.asset_id,
            dimension_scores=dimension_scores,
            overall_score=round(overall, 4),
            issues=issues,
            passed_rules=passed,
            failed_rules=failed,
        )
        self._reports[asset.asset_id].append(report)
        return report

    def _evaluate_completeness(
        self,
        asset: DataAsset,
        sample_data: Optional[List[Dict[str, Any]]],
        issues: List[Dict[str, Any]],
    ) -> float:
        non_nullable = [f for f in asset.schema_fields if not f.nullable]
        if not non_nullable or not sample_data:
            return 0.90
        total_checks = len(non_nullable) * len(sample_data)
        missing = 0
        for row in sample_data:
            for f in non_nullable:
                if row.get(f.name) is None:
                    missing += 1
        score = 1.0 - (missing / max(total_checks, 1))
        if missing > 0:
            issues.append(
                {
                    "dimension": "completeness",
                    "severity": "high" if score < 0.8 else "medium",
                    "message": f"{missing} null values in non-nullable fields",
                }
            )
        return round(score, 4)

    def _evaluate_accuracy(
        self,
        asset: DataAsset,
        sample_data: Optional[List[Dict[str, Any]]],
        issues: List[Dict[str, Any]],
    ) -> float:
        if not sample_data or not asset.schema_fields:
            return 0.88
        type_map = {"int": int, "str": str, "float": float, "bool": bool}
        errors = 0
        total = 0
        for row in sample_data:
            for f in asset.schema_fields:
                val = row.get(f.name)
                if val is None:
                    continue
                expected_type = type_map.get(f.data_type.lower())
                if expected_type and not isinstance(val, expected_type):
                    errors += 1
                total += 1
        if errors > 0:
            issues.append(
                {
                    "dimension": "accuracy",
                    "severity": "medium",
                    "message": f"{errors} type mismatches found",
                }
            )
        return round(1.0 - (errors / max(total, 1)), 4)

    def _evaluate_consistency(
        self,
        asset: DataAsset,
        issues: List[Dict[str, Any]],
    ) -> float:
        # Check for duplicate field names
        names = [f.name for f in asset.schema_fields]
        duplicates = len(names) - len(set(names))
        if duplicates > 0:
            issues.append(
                {
                    "dimension": "consistency",
                    "severity": "high",
                    "message": f"{duplicates} duplicate field names",
                }
            )
            return max(0.3, 1.0 - duplicates * 0.2)
        return 0.95

    def _evaluate_timeliness(
        self,
        asset: DataAsset,
        issues: List[Dict[str, Any]],
    ) -> float:
        freshness_map = {
            "real-time": 1.0,
            "streaming": 0.98,
            "hourly": 0.90,
            "daily": 0.75,
            "weekly": 0.60,
            "monthly": 0.45,
            "manual": 0.30,
        }
        freq = asset.update_frequency.lower()
        for key, score in freshness_map.items():
            if key in freq:
                return score
        return 0.70

    def _evaluate_uniqueness(
        self,
        asset: DataAsset,
        sample_data: Optional[List[Dict[str, Any]]],
        issues: List[Dict[str, Any]],
    ) -> float:
        pk_fields = [f.name for f in asset.schema_fields if f.primary_key]
        if not pk_fields or not sample_data:
            return 0.92
        pk_values = [
            tuple(str(row.get(f, "")) for f in pk_fields)
            for row in sample_data
        ]
        unique_pks = len(set(pk_values))
        duplicates = len(pk_values) - unique_pks
        if duplicates > 0:
            issues.append(
                {
                    "dimension": "uniqueness",
                    "severity": "critical",
                    "message": f"{duplicates} duplicate primary key values",
                }
            )
        return round(unique_pks / max(len(pk_values), 1), 4)

    def _evaluate_validity(
        self,
        asset: DataAsset,
        sample_data: Optional[List[Dict[str, Any]]],
        issues: List[Dict[str, Any]],
    ) -> float:
        if not asset.schema_fields:
            return 0.88
        # Check that all fields have non-empty names and valid types
        valid_types = {
            "int", "str", "float", "bool", "datetime", "date",
            "uuid", "json", "array", "bigint", "varchar", "text", "numeric",
        }
        invalid_fields = [
            f.name
            for f in asset.schema_fields
            if not f.name or f.data_type.lower() not in valid_types
        ]
        if invalid_fields:
            issues.append(
                {
                    "dimension": "validity",
                    "severity": "medium",
                    "message": f"{len(invalid_fields)} fields have unrecognized types",
                    "fields": invalid_fields[:5],
                }
            )
            return max(0.5, 1.0 - len(invalid_fields) * 0.1)
        return 0.97

    def get_quality_history(
        self, asset_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._reports[asset_id][-limit:]]

    def get_quality_stats(self) -> Dict[str, Any]:
        all_reports = []
        for reports in self._reports.values():
            all_reports.extend(reports)
        if not all_reports:
            return {"total_evaluations": 0}
        avg_score = sum(r.overall_score for r in all_reports) / len(all_reports)
        return {
            "total_evaluations": len(all_reports),
            "avg_overall_score": round(avg_score, 4),
            "assets_evaluated": len(self._reports),
        }


# ─────────────────────────── Data Contract Manager ──────────────────────────


class DataContractManager:
    """
    Manages data contracts between producers and consumers, enforcing SLAs
    and tracking contract violations.
    """

    def __init__(self):
        self._contracts: Dict[str, DataContract] = {}
        self._violations: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def create_contract(
        self,
        producer_asset_id: str,
        consumer_team: str,
        sla_freshness_minutes: int,
        sla_availability_pct: float,
        schema_version: str,
        agreed_fields: List[str],
    ) -> DataContract:
        contract = DataContract(
            contract_id=str(uuid.uuid4()),
            producer_asset_id=producer_asset_id,
            consumer_team=consumer_team,
            sla_freshness_minutes=sla_freshness_minutes,
            sla_availability_pct=sla_availability_pct,
            schema_version=schema_version,
            agreed_fields=agreed_fields,
            notifications_enabled=True,
        )
        self._contracts[contract.contract_id] = contract
        return contract

    def record_violation(
        self,
        contract_id: str,
        violation_type: str,
        details: str,
        severity: str = "high",
    ) -> None:
        self._violations[contract_id].append(
            {
                "violation_id": str(uuid.uuid4()),
                "violation_type": violation_type,
                "details": details,
                "severity": severity,
                "occurred_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    def get_contract(self, contract_id: str) -> Optional[DataContract]:
        return self._contracts.get(contract_id)

    def get_violations(
        self, contract_id: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        return self._violations[contract_id][-limit:]

    def list_contracts(
        self, producer_asset_id: Optional[str] = None, consumer_team: Optional[str] = None
    ) -> List[DataContract]:
        contracts = list(self._contracts.values())
        if producer_asset_id:
            contracts = [c for c in contracts if c.producer_asset_id == producer_asset_id]
        if consumer_team:
            contracts = [c for c in contracts if c.consumer_team == consumer_team]
        return contracts

    def get_contract_stats(self) -> Dict[str, Any]:
        contracts = list(self._contracts.values())
        total_violations = sum(len(v) for v in self._violations.values())
        return {
            "total_contracts": len(contracts),
            "total_violations": total_violations,
            "contracts_with_violations": sum(
                1 for cid, v in self._violations.items() if v and cid in self._contracts
            ),
        }


# ─────────────────────────── Data Mesh Engine ───────────────────────────────


class DataMeshEngine:
    """
    Master data mesh engine integrating catalog, lineage, quality,
    and contract management into a unified federated data platform.
    """

    def __init__(self):
        self.catalog = DataCatalog()
        self.lineage = DataLineageTracker()
        self.quality = DataQualityEngine()
        self.contracts = DataContractManager()

    def register_data_asset(
        self,
        name: str,
        description: str,
        domain: DataDomain,
        asset_type: DataAssetType,
        owner_team: str,
        schema_fields: List[Dict[str, Any]],
        access_policy: AccessPolicy = AccessPolicy.INTERNAL,
        tags: Optional[List[str]] = None,
        data_source: str = "unknown",
        update_frequency: str = "daily",
        row_count: Optional[int] = None,
        size_bytes: Optional[int] = None,
    ) -> DataAsset:
        asset_id = str(uuid.uuid4())
        fields = [
            SchemaField(
                name=f.get("name", ""),
                data_type=f.get("data_type", "str"),
                nullable=f.get("nullable", True),
                description=f.get("description", ""),
                tags=f.get("tags", []),
                pii=f.get("pii", False),
                primary_key=f.get("primary_key", False),
                foreign_key=f.get("foreign_key"),
            )
            for f in schema_fields
        ]
        asset = DataAsset(
            asset_id=asset_id,
            name=name,
            description=description,
            domain=domain,
            asset_type=asset_type,
            owner_team=owner_team,
            schema_fields=fields,
            access_policy=access_policy,
            tags=tags or [],
            quality_score=0.0,
            row_count=row_count,
            size_bytes=size_bytes,
            update_frequency=update_frequency,
            data_source=data_source,
        )
        self.catalog.register_asset(asset)
        # Register lineage node
        node = LineageNode(
            node_id=str(uuid.uuid4()),
            name=name,
            node_type=LineageNodeType.SOURCE,
            asset_id=asset_id,
            upstream_ids=[],
            downstream_ids=[],
            transformation_logic=None,
        )
        self.lineage.register_node(node)
        return asset

    def evaluate_and_update_quality(
        self,
        asset_id: str,
        sample_data: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[QualityReport]:
        asset = self.catalog.get_asset(asset_id)
        if asset is None:
            return None
        report = self.quality.evaluate_asset(asset, sample_data)
        asset.quality_score = report.overall_score
        asset.updated_at = datetime.now(timezone.utc)
        return report

    def search_assets(
        self,
        query: str,
        domain: Optional[str] = None,
        asset_type: Optional[str] = None,
        min_quality: float = 0.0,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        domain_enum = DataDomain(domain) if domain else None
        type_enum = DataAssetType(asset_type) if asset_type else None
        assets = self.catalog.search(
            query, domain_enum, type_enum, None, min_quality, limit
        )
        return [a.to_dict() for a in assets]

    def get_asset_lineage(
        self, asset_id: str, direction: str = "both"
    ) -> Dict[str, Any]:
        node_id = self.lineage._asset_to_node.get(asset_id)
        if node_id is None:
            return {"error": "Asset not tracked in lineage"}
        result: Dict[str, Any] = {"asset_id": asset_id, "node_id": node_id}
        if direction in ("upstream", "both"):
            result["upstream"] = self.lineage.get_upstream_lineage(node_id)
        if direction in ("downstream", "both"):
            result["downstream"] = self.lineage.get_downstream_lineage(node_id)
        return result

    def get_mesh_overview(self) -> Dict[str, Any]:
        return {
            "catalog": self.catalog.get_catalog_stats(),
            "lineage": self.lineage.get_lineage_stats(),
            "quality": self.quality.get_quality_stats(),
            "contracts": self.contracts.get_contract_stats(),
        }
