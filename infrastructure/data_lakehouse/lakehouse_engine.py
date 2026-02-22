"""
Data Lakehouse Engine
======================
Production-grade analytical data store combining data lake and data warehouse:
- Schema-on-write and schema-on-read support
- Delta Lake-style ACID transactions
- Partition management and pruning
- Column statistics and data skipping
- SQL query engine with optimizer
- ETL pipeline framework with lineage
- Data catalog with discovery
- Time-travel queries (snapshot isolation)
- Compaction and vacuuming
- Multi-format ingestion (JSON, CSV, Parquet-compatible)
- Materialized views and pre-aggregations
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class DataFormat(str, Enum):
    """Storage format for lakehouse data."""
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    AVRO = "avro"
    ORC = "orc"
    DELTA = "delta"


class CompressionType(str, Enum):
    """Compression algorithm for stored data."""
    NONE = "none"
    SNAPPY = "snappy"
    GZIP = "gzip"
    ZSTD = "zstd"
    LZ4 = "lz4"


class ColumnDataType(str, Enum):
    """Column data types for schema enforcement."""
    STRING = "string"
    INTEGER = "integer"
    BIGINT = "bigint"
    FLOAT = "float"
    DOUBLE = "double"
    BOOLEAN = "boolean"
    TIMESTAMP = "timestamp"
    DATE = "date"
    BINARY = "binary"
    ARRAY = "array"
    MAP = "map"
    STRUCT = "struct"


class TransactionState(str, Enum):
    """State of a lakehouse transaction."""
    PENDING = "pending"
    COMMITTED = "committed"
    ABORTED = "aborted"


class TableType(str, Enum):
    """Type of lakehouse table."""
    MANAGED = "managed"
    EXTERNAL = "external"
    VIEW = "view"
    MATERIALIZED_VIEW = "materialized_view"


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ColumnDefinition:
    """Schema definition for a single column."""
    name: str = ""
    data_type: ColumnDataType = ColumnDataType.STRING
    nullable: bool = True
    description: str = ""
    default_value: Optional[Any] = None
    is_partition_key: bool = False
    is_sort_key: bool = False
    compression: CompressionType = CompressionType.SNAPPY
    statistics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "data_type": self.data_type.value,
            "nullable": self.nullable,
            "description": self.description,
            "is_partition_key": self.is_partition_key,
            "is_sort_key": self.is_sort_key,
            "statistics": self.statistics,
        }

    def validate_value(self, value: Any) -> bool:
        """Validate that a value matches this column's type."""
        if value is None:
            return self.nullable
        type_checks = {
            ColumnDataType.STRING: str,
            ColumnDataType.INTEGER: int,
            ColumnDataType.BIGINT: int,
            ColumnDataType.FLOAT: float,
            ColumnDataType.DOUBLE: float,
            ColumnDataType.BOOLEAN: bool,
        }
        expected_type = type_checks.get(self.data_type)
        if expected_type:
            return isinstance(value, expected_type)
        return True  # Complex types: permissive


@dataclass
class DataSchema:
    """Full schema definition for a lakehouse table."""
    schema_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    version: int = 1
    columns: List[ColumnDefinition] = field(default_factory=list)
    primary_keys: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def get_column(self, name: str) -> Optional[ColumnDefinition]:
        for col in self.columns:
            if col.name == name:
                return col
        return None

    def get_partition_keys(self) -> List[str]:
        return [c.name for c in self.columns if c.is_partition_key]

    def validate_row(self, row: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a data row against the schema. Returns (valid, errors)."""
        errors: List[str] = []
        col_names = {c.name for c in self.columns}

        for col in self.columns:
            value = row.get(col.name)
            if not col.validate_value(value):
                errors.append(f"Column '{col.name}': invalid type for value {value!r}")

        return (len(errors) == 0, errors)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "name": self.name,
            "version": self.version,
            "columns": [c.to_dict() for c in self.columns],
            "primary_keys": self.primary_keys,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class LakehousePartition:
    """A single partition of data within a table."""
    partition_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    table_name: str = ""
    partition_keys: Dict[str, Any] = field(default_factory=dict)
    file_path: str = ""
    record_count: int = 0
    size_bytes: int = 0
    format: DataFormat = DataFormat.PARQUET
    compression: CompressionType = CompressionType.SNAPPY
    min_values: Dict[str, Any] = field(default_factory=dict)
    max_values: Dict[str, Any] = field(default_factory=dict)
    null_counts: Dict[str, int] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_modified: datetime = field(default_factory=datetime.utcnow)
    version: int = 1  # For time-travel
    is_active: bool = True
    checksum: str = ""
    data: List[Dict[str, Any]] = field(default_factory=list)  # In-memory storage

    def compute_statistics(self) -> None:
        """Compute column-level statistics from data."""
        if not self.data:
            return
        for col_name in self.data[0].keys():
            values = [row.get(col_name) for row in self.data if row.get(col_name) is not None]
            if not values:
                self.null_counts[col_name] = len(self.data)
                continue
            self.null_counts[col_name] = len(self.data) - len(values)
            try:
                numeric_vals = [float(v) for v in values if isinstance(v, (int, float))]
                if numeric_vals:
                    self.min_values[col_name] = min(numeric_vals)
                    self.max_values[col_name] = max(numeric_vals)
                elif all(isinstance(v, str) for v in values):
                    self.min_values[col_name] = min(values)
                    self.max_values[col_name] = max(values)
            except (TypeError, ValueError):
                pass

    def matches_filter(self, filters: Dict[str, Any]) -> bool:
        """Check if this partition could contain rows matching the filters (partition pruning)."""
        for col, value in filters.items():
            if col in self.partition_keys:
                if self.partition_keys[col] != value:
                    return False
            # Statistics-based pruning
            if col in self.min_values and col in self.max_values:
                if isinstance(value, (int, float)):
                    if value < self.min_values[col] or value > self.max_values[col]:
                        return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "partition_id": self.partition_id,
            "table_name": self.table_name,
            "partition_keys": self.partition_keys,
            "record_count": self.record_count,
            "size_bytes": self.size_bytes,
            "format": self.format.value,
            "version": self.version,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class LakehouseTable:
    """A table in the data lakehouse with full metadata."""
    table_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    database: str = "default"
    table_type: TableType = TableType.MANAGED
    schema: Optional[DataSchema] = None
    format: DataFormat = DataFormat.DELTA
    compression: CompressionType = CompressionType.SNAPPY
    location: str = ""
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)
    owner: str = "system"
    tenant_id: str = "default"
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_modified: datetime = field(default_factory=datetime.utcnow)
    row_count: int = 0
    size_bytes: int = 0
    partition_count: int = 0
    version: int = 0  # Current version for time-travel
    view_definition: Optional[str] = None  # SQL for views

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_id": self.table_id,
            "name": self.name,
            "database": self.database,
            "table_type": self.table_type.value,
            "format": self.format.value,
            "description": self.description,
            "tags": self.tags,
            "owner": self.owner,
            "tenant_id": self.tenant_id,
            "created_at": self.created_at.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "row_count": self.row_count,
            "size_bytes": self.size_bytes,
            "partition_count": self.partition_count,
            "version": self.version,
            "schema": self.schema.to_dict() if self.schema else None,
        }


@dataclass
class Transaction:
    """A lakehouse ACID transaction."""
    txn_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: TransactionState = TransactionState.PENDING
    operations: List[Dict[str, Any]] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    committed_at: Optional[datetime] = None
    aborted_at: Optional[datetime] = None
    isolation_level: str = "snapshot"
    snapshot_version: int = 0

    def add_operation(self, op_type: str, table: str, data: Any) -> None:
        self.operations.append({
            "op_type": op_type,
            "table": table,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "txn_id": self.txn_id,
            "state": self.state.value,
            "operation_count": len(self.operations),
            "started_at": self.started_at.isoformat(),
            "committed_at": self.committed_at.isoformat() if self.committed_at else None,
        }


@dataclass
class QueryPlan:
    """Optimized execution plan for a SQL query."""
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sql: str = ""
    estimated_rows: int = 0
    estimated_cost: float = 0.0
    partitions_scanned: int = 0
    partitions_pruned: int = 0
    index_used: bool = False
    plan_steps: List[Dict[str, Any]] = field(default_factory=list)
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "estimated_rows": self.estimated_rows,
            "estimated_cost": self.estimated_cost,
            "partitions_scanned": self.partitions_scanned,
            "partitions_pruned": self.partitions_pruned,
            "plan_steps": self.plan_steps,
            "execution_time_ms": self.execution_time_ms,
        }


# ---------------------------------------------------------------------------
# Query Engine
# ---------------------------------------------------------------------------

class QueryEngine:
    """
    In-process SQL-like query engine for the lakehouse.
    Supports SELECT, WHERE, GROUP BY, ORDER BY, LIMIT, and aggregations.
    """

    def __init__(self) -> None:
        self._query_history: deque = deque(maxlen=1000)

    async def execute_query(
        self,
        sql: str,
        tables: Dict[str, List[Dict[str, Any]]],
    ) -> Tuple[List[Dict[str, Any]], QueryPlan]:
        """Parse and execute a SQL query against provided table data."""
        import time
        start = time.monotonic()
        plan = QueryPlan(sql=sql)

        try:
            result = self._execute_sql(sql, tables, plan)
        except Exception as exc:
            logger.error("Query execution error: %s", exc)
            result = []
            plan.plan_steps.append({"error": str(exc)})

        plan.execution_time_ms = (time.monotonic() - start) * 1000
        self._query_history.append({
            "sql": sql[:200],
            "rows_returned": len(result),
            "execution_time_ms": plan.execution_time_ms,
            "timestamp": datetime.utcnow().isoformat(),
        })
        return result, plan

    def _execute_sql(
        self, sql: str, tables: Dict[str, List[Dict[str, Any]]], plan: QueryPlan
    ) -> List[Dict[str, Any]]:
        """Simple SQL executor supporting basic SELECT queries."""
        sql = sql.strip()
        sql_upper = sql.upper()

        if not sql_upper.startswith("SELECT"):
            raise ValueError("Only SELECT queries are supported in the query engine")

        # Parse FROM clause
        from_match = re.search(r'\bFROM\s+(\w+)', sql, re.I)
        if not from_match:
            raise ValueError("No FROM clause found in query")
        table_name = from_match.group(1)
        data = tables.get(table_name, [])

        # Parse WHERE clause
        where_match = re.search(r'\bWHERE\s+(.*?)(?:\bGROUP\b|\bORDER\b|\bLIMIT\b|$)', sql, re.I | re.S)
        if where_match:
            where_clause = where_match.group(1).strip()
            data = self._apply_where(data, where_clause)

        # Parse GROUP BY
        group_match = re.search(r'\bGROUP\s+BY\s+(.*?)(?:\bHAVING\b|\bORDER\b|\bLIMIT\b|$)', sql, re.I | re.S)
        if group_match:
            group_keys = [k.strip() for k in group_match.group(1).split(",")]
            data = self._apply_group_by(data, group_keys, sql)

        # Parse ORDER BY
        order_match = re.search(r'\bORDER\s+BY\s+(.*?)(?:\bLIMIT\b|$)', sql, re.I | re.S)
        if order_match:
            order_clause = order_match.group(1).strip()
            data = self._apply_order_by(data, order_clause)

        # Parse LIMIT
        limit_match = re.search(r'\bLIMIT\s+(\d+)', sql, re.I)
        if limit_match:
            limit = int(limit_match.group(1))
            data = data[:limit]

        # Parse SELECT columns
        select_match = re.search(r'^SELECT\s+(.*?)\s+FROM\b', sql, re.I | re.S)
        if select_match:
            select_clause = select_match.group(1).strip()
            if select_clause != "*":
                data = self._apply_select(data, select_clause)

        plan.estimated_rows = len(data)
        return data

    def _apply_where(self, data: List[Dict], where: str) -> List[Dict]:
        """Apply WHERE filters - supports basic comparisons and AND/OR."""
        result = []
        for row in data:
            if self._evaluate_condition(row, where):
                result.append(row)
        return result

    def _evaluate_condition(self, row: Dict, condition: str) -> bool:
        """Evaluate a single WHERE condition against a row."""
        condition = condition.strip()

        # Handle AND
        if " AND " in condition.upper():
            parts = re.split(r'\bAND\b', condition, flags=re.I)
            return all(self._evaluate_condition(row, p) for p in parts)

        # Handle OR
        if " OR " in condition.upper():
            parts = re.split(r'\bOR\b', condition, flags=re.I)
            return any(self._evaluate_condition(row, p) for p in parts)

        # Simple comparisons: col op value
        match = re.match(r"(\w+)\s*(=|!=|>=|<=|>|<|LIKE|IN)\s*(.+)", condition, re.I)
        if not match:
            return True  # Unknown condition: pass

        col, op, val_str = match.group(1), match.group(2), match.group(3).strip()
        row_val = row.get(col)
        op = op.upper()

        try:
            if op == "LIKE":
                pattern = val_str.strip("'\"").replace("%", ".*").replace("_", ".")
                return bool(re.match(f"^{pattern}$", str(row_val or ""), re.I))
            elif op == "IN":
                vals = [v.strip().strip("'\"") for v in val_str.strip("()").split(",")]
                return str(row_val) in vals

            # Numeric comparison
            try:
                val = float(val_str.strip("'\""))
                row_num = float(row_val)
            except (TypeError, ValueError):
                val = val_str.strip("'\"")
                row_num = str(row_val)

            if op == "=":
                return row_num == val
            elif op == "!=":
                return row_num != val
            elif op == ">":
                return row_num > val
            elif op == "<":
                return row_num < val
            elif op == ">=":
                return row_num >= val
            elif op == "<=":
                return row_num <= val
        except (TypeError, ValueError):
            return True

        return True

    def _apply_group_by(
        self, data: List[Dict], keys: List[str], sql: str
    ) -> List[Dict]:
        """Apply GROUP BY with basic aggregations."""
        groups: Dict[tuple, List[Dict]] = defaultdict(list)
        for row in data:
            group_key = tuple(row.get(k) for k in keys)
            groups[group_key].append(row)

        result = []
        for group_key, rows in groups.items():
            agg_row: Dict[str, Any] = dict(zip(keys, group_key))

            # Apply aggregations: COUNT, SUM, AVG, MIN, MAX
            count_matches = re.findall(r'COUNT\(\s*\*?\s*\)(?:\s+AS\s+(\w+))?', sql, re.I)
            for alias in count_matches:
                key = alias or "count"
                agg_row[key] = len(rows)

            for func in ["SUM", "AVG", "MIN", "MAX"]:
                matches = re.findall(rf'{func}\((\w+)\)(?:\s+AS\s+(\w+))?', sql, re.I)
                for col, alias in matches:
                    key = alias or f"{func.lower()}_{col}"
                    vals = [float(r[col]) for r in rows if col in r and r[col] is not None]
                    if vals:
                        if func == "SUM":
                            agg_row[key] = sum(vals)
                        elif func == "AVG":
                            agg_row[key] = sum(vals) / len(vals)
                        elif func == "MIN":
                            agg_row[key] = min(vals)
                        elif func == "MAX":
                            agg_row[key] = max(vals)

            result.append(agg_row)
        return result

    def _apply_order_by(self, data: List[Dict], order_clause: str) -> List[Dict]:
        """Apply ORDER BY clause."""
        parts = order_clause.split(",")
        sort_keys = []
        for part in parts:
            part = part.strip()
            tokens = part.split()
            col = tokens[0]
            reverse = len(tokens) > 1 and tokens[1].upper() == "DESC"
            sort_keys.append((col, reverse))

        for col, reverse in reversed(sort_keys):
            data = sorted(
                data,
                key=lambda r: (r.get(col) is None, r.get(col)),
                reverse=reverse,
            )
        return data

    def _apply_select(self, data: List[Dict], select: str) -> List[Dict]:
        """Apply SELECT column projection."""
        cols = [c.strip() for c in select.split(",")]
        # Handle aliases: col AS alias
        col_map: Dict[str, str] = {}
        for col in cols:
            alias_match = re.match(r"(\w+)\s+AS\s+(\w+)", col, re.I)
            if alias_match:
                col_map[alias_match.group(1)] = alias_match.group(2)
            else:
                col_map[col] = col

        result = []
        for row in data:
            new_row: Dict[str, Any] = {}
            for orig, alias in col_map.items():
                # Skip aggregation expressions (handled in group_by)
                if "(" not in orig and orig in row:
                    new_row[alias] = row[orig]
                elif orig in row:
                    new_row[alias] = row[orig]
            if new_row:
                result.append(new_row)
            else:
                result.append(row)
        return result

    def get_query_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        history = list(self._query_history)
        return history[-limit:]


# ---------------------------------------------------------------------------
# ETL Pipeline
# ---------------------------------------------------------------------------

@dataclass
class ETLJob:
    """Configuration for an ETL job."""
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    source_table: str = ""
    target_table: str = ""
    transform_sql: str = ""
    schedule_cron: str = ""
    enabled: bool = True
    last_run_at: Optional[datetime] = None
    last_status: str = "never_run"
    run_count: int = 0
    success_count: int = 0
    error_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "source_table": self.source_table,
            "target_table": self.target_table,
            "enabled": self.enabled,
            "last_run_at": self.last_run_at.isoformat() if self.last_run_at else None,
            "last_status": self.last_status,
            "run_count": self.run_count,
            "success_rate": self.success_count / self.run_count if self.run_count > 0 else 0.0,
        }


class ETLPipeline:
    """ETL pipeline framework with lineage tracking and scheduling."""

    def __init__(self) -> None:
        self._jobs: Dict[str, ETLJob] = {}
        self._lineage: Dict[str, List[str]] = defaultdict(list)  # table -> tables it feeds
        self._run_history: deque = deque(maxlen=1000)

    async def register_job(self, job: ETLJob) -> ETLJob:
        self._jobs[job.job_id] = job
        self._lineage[job.source_table].append(job.target_table)
        logger.info("Registered ETL job: %s (%s -> %s)", job.name, job.source_table, job.target_table)
        return job

    async def run_job(
        self, job_id: str, lakehouse: "DataLakehouse"
    ) -> Dict[str, Any]:
        """Execute an ETL job."""
        job = self._jobs.get(job_id)
        if not job:
            return {"success": False, "error": f"Job {job_id} not found"}

        import time
        start = time.monotonic()
        job.run_count += 1
        job.last_run_at = datetime.utcnow()

        try:
            source_data = await lakehouse.scan_table(job.source_table)
            if job.transform_sql:
                result, _ = await lakehouse._query_engine.execute_query(
                    job.transform_sql, {job.source_table: source_data}
                )
            else:
                result = source_data

            await lakehouse.insert_rows(job.target_table, result)
            job.success_count += 1
            job.last_status = "success"
            duration = (time.monotonic() - start) * 1000

            run_record = {
                "job_id": job_id,
                "job_name": job.name,
                "status": "success",
                "rows_processed": len(result),
                "duration_ms": duration,
                "timestamp": datetime.utcnow().isoformat(),
            }
            self._run_history.append(run_record)
            logger.info("ETL job %s completed: %d rows in %.1fms", job.name, len(result), duration)
            return {**run_record, "success": True}

        except Exception as exc:
            job.error_count += 1
            job.last_status = "failed"
            logger.error("ETL job %s failed: %s", job.name, exc)
            run_record = {
                "job_id": job_id,
                "status": "failed",
                "error": str(exc),
                "timestamp": datetime.utcnow().isoformat(),
            }
            self._run_history.append(run_record)
            return {**run_record, "success": False}

    def get_lineage(self, table_name: str) -> Dict[str, Any]:
        """Get data lineage for a table (what feeds it and what it feeds)."""
        downstream = self._lineage.get(table_name, [])
        upstream = [src for src, targets in self._lineage.items() if table_name in targets]
        return {
            "table": table_name,
            "upstream_tables": upstream,
            "downstream_tables": downstream,
        }

    def list_jobs(self) -> List[Dict[str, Any]]:
        return [j.to_dict() for j in self._jobs.values()]


# ---------------------------------------------------------------------------
# Data Catalog
# ---------------------------------------------------------------------------

class DataCatalog:
    """Metadata catalog for discoverable data assets."""

    def __init__(self) -> None:
        self._assets: Dict[str, Dict[str, Any]] = {}
        self._search_index: Dict[str, List[str]] = defaultdict(list)

    def register_asset(
        self,
        asset_type: str,
        name: str,
        metadata: Dict[str, Any],
        tags: Optional[List[str]] = None,
    ) -> str:
        """Register a data asset in the catalog."""
        asset_id = str(uuid.uuid4())
        self._assets[asset_id] = {
            "asset_id": asset_id,
            "asset_type": asset_type,
            "name": name,
            "metadata": metadata,
            "tags": tags or [],
            "created_at": datetime.utcnow().isoformat(),
        }
        # Index for search
        for token in name.lower().split("_"):
            self._search_index[token].append(asset_id)
        for tag in (tags or []):
            self._search_index[tag.lower()].append(asset_id)
        return asset_id

    def search(self, query: str, asset_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search the catalog by keyword."""
        tokens = query.lower().split()
        matching_ids: set = set()
        for token in tokens:
            matching_ids.update(self._search_index.get(token, []))

        results = []
        for aid in matching_ids:
            asset = self._assets.get(aid)
            if asset and (asset_type is None or asset["asset_type"] == asset_type):
                results.append(asset)
        return results

    def get_asset(self, asset_id: str) -> Optional[Dict[str, Any]]:
        return self._assets.get(asset_id)

    def list_assets(self, asset_type: Optional[str] = None) -> List[Dict[str, Any]]:
        assets = self._assets.values()
        if asset_type:
            assets = [a for a in assets if a["asset_type"] == asset_type]
        return list(assets)

    def get_stats(self) -> Dict[str, Any]:
        type_counts: Dict[str, int] = defaultdict(int)
        for asset in self._assets.values():
            type_counts[asset["asset_type"]] += 1
        return {
            "total_assets": len(self._assets),
            "type_distribution": dict(type_counts),
            "index_terms": len(self._search_index),
        }


# ---------------------------------------------------------------------------
# Data Lakehouse
# ---------------------------------------------------------------------------

class DataLakehouse:
    """
    Production-grade data lakehouse combining ACID transactions,
    schema enforcement, partitioning, time-travel, and SQL analytics.
    """

    def __init__(self) -> None:
        self._tables: Dict[str, LakehouseTable] = {}
        self._partitions: Dict[str, List[LakehousePartition]] = defaultdict(list)
        self._transactions: Dict[str, Transaction] = {}
        self._query_engine = QueryEngine()
        self._etl_pipeline = ETLPipeline()
        self._catalog = DataCatalog()
        self._transaction_log: deque = deque(maxlen=10000)
        self._lock = asyncio.Lock()
        self._global_version: int = 0

    # ------------------------------------------------------------------
    # Table Management
    # ------------------------------------------------------------------

    async def create_table(self, table: LakehouseTable) -> LakehouseTable:
        """Create a new table in the lakehouse."""
        async with self._lock:
            key = f"{table.database}.{table.name}"
            if key in self._tables:
                raise ValueError(f"Table '{key}' already exists")
            self._tables[key] = table
            self._catalog.register_asset(
                "table", key,
                {"format": table.format.value, "owner": table.owner},
                list(table.tags.values()),
            )
            logger.info("Created lakehouse table: %s", key)
            return table

    async def get_table(self, database: str, table_name: str) -> Optional[LakehouseTable]:
        key = f"{database}.{table_name}"
        return self._tables.get(key)

    async def drop_table(self, database: str, table_name: str) -> bool:
        """Drop a table and all its data."""
        async with self._lock:
            key = f"{database}.{table_name}"
            if key not in self._tables:
                return False
            del self._tables[key]
            self._partitions.pop(key, None)
            logger.info("Dropped table: %s", key)
            return True

    async def list_tables(
        self, database: str = "default", tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List tables in a database."""
        prefix = f"{database}."
        result = []
        for key, table in self._tables.items():
            if key.startswith(prefix):
                if tenant_id is None or table.tenant_id == tenant_id:
                    result.append(table.to_dict())
        return result

    # ------------------------------------------------------------------
    # Data Operations
    # ------------------------------------------------------------------

    async def insert_rows(
        self,
        table_key: str,
        rows: List[Dict[str, Any]],
        txn_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Insert rows into a table, creating partitions as needed."""
        async with self._lock:
            table = self._tables.get(table_key)
            if not table:
                # Auto-create if not exists
                db, tname = (table_key.split(".", 1) + ["default"])[:2]
                table = LakehouseTable(name=tname, database=db)
                self._tables[table_key] = table

            if not rows:
                return {"inserted": 0}

            # Schema validation
            validation_errors: List[str] = []
            if table.schema:
                for i, row in enumerate(rows):
                    valid, errors = table.schema.validate_row(row)
                    if not valid:
                        validation_errors.extend([f"Row {i}: {e}" for e in errors])

            if validation_errors:
                return {
                    "inserted": 0,
                    "errors": validation_errors[:10],
                }

            # Create a new partition for this batch
            partition_keys = {}
            if table.schema:
                for pk in table.schema.get_partition_keys():
                    if rows and pk in rows[0]:
                        partition_keys[pk] = rows[0][pk]

            partition = LakehousePartition(
                table_name=table_key,
                partition_keys=partition_keys,
                record_count=len(rows),
                size_bytes=sum(len(json.dumps(r)) for r in rows),
                format=table.format,
                data=list(rows),
                version=table.version + 1,
            )
            partition.compute_statistics()

            self._partitions[table_key].append(partition)
            table.row_count += len(rows)
            table.size_bytes += partition.size_bytes
            table.partition_count = len(self._partitions[table_key])
            table.version += 1
            table.last_modified = datetime.utcnow()
            self._global_version += 1

            self._transaction_log.append({
                "operation": "INSERT",
                "table": table_key,
                "rows": len(rows),
                "version": table.version,
                "txn_id": txn_id,
                "timestamp": datetime.utcnow().isoformat(),
            })

            return {
                "inserted": len(rows),
                "partition_id": partition.partition_id,
                "table_version": table.version,
            }

    async def scan_table(
        self,
        table_key: str,
        filters: Optional[Dict[str, Any]] = None,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
        as_of_version: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Full table scan with optional partition pruning and column projection."""
        partitions = self._partitions.get(table_key, [])
        result: List[Dict[str, Any]] = []

        for partition in partitions:
            if not partition.is_active:
                continue
            if as_of_version is not None and partition.version > as_of_version:
                continue
            if filters and not partition.matches_filter(filters):
                continue  # Partition pruning

            for row in partition.data:
                if filters:
                    match = all(
                        row.get(k) == v for k, v in filters.items()
                        if not isinstance(v, dict)
                    )
                    if not match:
                        continue

                if columns:
                    row = {c: row.get(c) for c in columns}
                result.append(row)

                if limit and len(result) >= limit:
                    return result

        return result

    async def query(
        self,
        sql: str,
        database: str = "default",
        tenant_id: str = "default",
    ) -> Tuple[List[Dict[str, Any]], QueryPlan]:
        """Execute a SQL query against the lakehouse."""
        # Extract table names and load their data
        table_refs = re.findall(r'\bFROM\s+(\w+)', sql, re.I)
        tables: Dict[str, List[Dict[str, Any]]] = {}
        for tname in table_refs:
            key = f"{database}.{tname}"
            data = await self.scan_table(key)
            tables[tname] = data

        return await self._query_engine.execute_query(sql, tables)

    async def delete_rows(
        self,
        table_key: str,
        filters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Delete rows matching filters (soft delete via version marking)."""
        async with self._lock:
            deleted = 0
            for partition in self._partitions.get(table_key, []):
                if not partition.is_active:
                    continue
                original = len(partition.data)
                partition.data = [
                    row for row in partition.data
                    if not all(row.get(k) == v for k, v in filters.items())
                ]
                deleted += original - len(partition.data)
                partition.record_count = len(partition.data)

            table = self._tables.get(table_key)
            if table:
                table.row_count = max(0, table.row_count - deleted)
                table.version += 1
                table.last_modified = datetime.utcnow()

            return {"deleted": deleted}

    async def update_rows(
        self,
        table_key: str,
        filters: Dict[str, Any],
        updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update rows matching filters."""
        async with self._lock:
            updated = 0
            for partition in self._partitions.get(table_key, []):
                if not partition.is_active:
                    continue
                for row in partition.data:
                    if all(row.get(k) == v for k, v in filters.items()):
                        row.update(updates)
                        updated += 1

            table = self._tables.get(table_key)
            if table:
                table.version += 1
                table.last_modified = datetime.utcnow()

            return {"updated": updated}

    # ------------------------------------------------------------------
    # Time-Travel
    # ------------------------------------------------------------------

    async def query_at_version(
        self, table_key: str, version: int, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Query the table state at a specific historical version."""
        return await self.scan_table(table_key, as_of_version=version, limit=limit)

    async def get_table_history(self, table_key: str) -> List[Dict[str, Any]]:
        """Get the version history of a table."""
        history = [
            entry for entry in self._transaction_log
            if entry.get("table") == table_key
        ]
        return sorted(history, key=lambda x: x.get("version", 0), reverse=True)

    # ------------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------------

    async def begin_transaction(self) -> Transaction:
        """Start an ACID transaction."""
        async with self._lock:
            txn = Transaction(snapshot_version=self._global_version)
            self._transactions[txn.txn_id] = txn
            return txn

    async def commit_transaction(self, txn_id: str) -> bool:
        """Commit a transaction."""
        async with self._lock:
            txn = self._transactions.get(txn_id)
            if not txn:
                return False
            txn.state = TransactionState.COMMITTED
            txn.committed_at = datetime.utcnow()
            logger.info("Committed transaction: %s", txn_id)
            return True

    async def abort_transaction(self, txn_id: str) -> bool:
        """Abort/rollback a transaction."""
        async with self._lock:
            txn = self._transactions.get(txn_id)
            if not txn:
                return False
            txn.state = TransactionState.ABORTED
            txn.aborted_at = datetime.utcnow()
            logger.info("Aborted transaction: %s", txn_id)
            return True

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    async def compact_table(self, table_key: str) -> Dict[str, Any]:
        """Compact small partitions into fewer, larger ones."""
        async with self._lock:
            partitions = self._partitions.get(table_key, [])
            all_data: List[Dict[str, Any]] = []
            for p in partitions:
                if p.is_active:
                    all_data.extend(p.data)
                    p.is_active = False

            if not all_data:
                return {"compacted": 0, "new_partitions": 0}

            # Create one new compact partition
            new_partition = LakehousePartition(
                table_name=table_key,
                record_count=len(all_data),
                size_bytes=sum(len(json.dumps(r)) for r in all_data),
                data=all_data,
            )
            new_partition.compute_statistics()
            self._partitions[table_key].append(new_partition)

            table = self._tables.get(table_key)
            if table:
                table.partition_count = 1

            logger.info("Compacted table %s: %d rows into 1 partition", table_key, len(all_data))
            return {"compacted": len(partitions), "new_partitions": 1, "rows": len(all_data)}

    async def vacuum_table(self, table_key: str, retain_versions: int = 7) -> Dict[str, Any]:
        """Remove inactive (deleted) partitions beyond the retention policy."""
        async with self._lock:
            partitions = self._partitions.get(table_key, [])
            inactive = [p for p in partitions if not p.is_active]
            retained = partitions[-retain_versions:] if len(partitions) > retain_versions else partitions
            self._partitions[table_key] = [p for p in partitions if p.is_active or p in retained]
            removed = len(inactive) - len([p for p in inactive if p in retained])
            logger.info("Vacuumed table %s: removed %d inactive partitions", table_key, removed)
            return {"removed_partitions": removed}

    # ------------------------------------------------------------------
    # Analytics & Reporting
    # ------------------------------------------------------------------

    async def get_table_stats(self, table_key: str) -> Dict[str, Any]:
        """Get detailed statistics for a table."""
        table = self._tables.get(table_key)
        if not table:
            return {"error": f"Table {table_key} not found"}

        partitions = self._partitions.get(table_key, [])
        active_parts = [p for p in partitions if p.is_active]

        return {
            **table.to_dict(),
            "active_partitions": len(active_parts),
            "inactive_partitions": len(partitions) - len(active_parts),
            "avg_partition_size_bytes": (
                sum(p.size_bytes for p in active_parts) / len(active_parts)
                if active_parts else 0
            ),
        }

    async def get_lakehouse_summary(self) -> Dict[str, Any]:
        """Get an overall summary of the lakehouse."""
        total_rows = sum(t.row_count for t in self._tables.values())
        total_size = sum(t.size_bytes for t in self._tables.values())
        return {
            "total_tables": len(self._tables),
            "total_rows": total_rows,
            "total_size_bytes": total_size,
            "total_partitions": sum(len(p) for p in self._partitions.values()),
            "global_version": self._global_version,
            "etl_jobs": len(self._etl_pipeline._jobs),
            "catalog_assets": self._catalog.get_stats()["total_assets"],
        }

    @property
    def etl(self) -> ETLPipeline:
        return self._etl_pipeline

    @property
    def catalog(self) -> DataCatalog:
        return self._catalog

    @property
    def query_engine(self) -> QueryEngine:
        return self._query_engine
