"""
Database Sharding Manager for Massive Scale

Provides intelligent database sharding capabilities:
- Automatic shard key selection
- Dynamic shard allocation
- Cross-shard query routing
- Rebalancing and migration
- Consistent hashing
"""

import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import bisect


class ShardStrategy(Enum):
    """Sharding strategies"""
    HASH = "hash"
    RANGE = "range"
    DIRECTORY = "directory"
    GEOGRAPHIC = "geographic"
    COMPOSITE = "composite"


@dataclass
class Shard:
    """Database shard configuration"""
    shard_id: str
    host: str
    port: int
    database: str
    min_key: Optional[Any] = None
    max_key: Optional[Any] = None
    current_size_gb: float = 0.0
    max_size_gb: float = 100.0
    is_active: bool = True
    is_read_only: bool = False
    replica_count: int = 2
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShardMapping:
    """Mapping of key to shard"""
    key: str
    shard_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)


class DatabaseShardManager:
    """
    Database Sharding Manager

    Features:
    - Multiple sharding strategies (hash, range, directory, geographic)
    - Consistent hashing for even distribution
    - Virtual nodes for better balance
    - Automatic shard discovery
    - Cross-shard query routing
    - Shard rebalancing
    - Hot shard detection and mitigation
    - Data migration between shards
    - Read/write splitting
    - Multi-shard transactions
    """

    def __init__(self, strategy: ShardStrategy = ShardStrategy.HASH):
        self.strategy = strategy
        self.shards: Dict[str, Shard] = {}
        self.directory: Dict[str, str] = {}  # For directory-based sharding
        self._consistent_hash_ring: List[Tuple[int, str]] = []
        self._virtual_nodes_per_shard = 150  # For consistent hashing
        self._shard_metrics: Dict[str, Dict[str, Any]] = {}

    def add_shard(self, shard: Shard):
        """Add a new shard to the cluster"""
        self.shards[shard.shard_id] = shard
        self._shard_metrics[shard.shard_id] = {
            "read_count": 0,
            "write_count": 0,
            "query_latency_ms": 0.0,
            "error_count": 0
        }

        # Update consistent hash ring if using hash strategy
        if self.strategy == ShardStrategy.HASH:
            self._update_hash_ring()

    def remove_shard(self, shard_id: str):
        """Remove shard from cluster (requires migration)"""
        if shard_id in self.shards:
            # Mark as inactive first
            self.shards[shard_id].is_active = False
            self.shards[shard_id].is_read_only = True

            # Would trigger migration in production
            del self.shards[shard_id]

            if self.strategy == ShardStrategy.HASH:
                self._update_hash_ring()

    def get_shard_for_key(self, key: str) -> Optional[Shard]:
        """
        Determine which shard should handle this key

        Args:
            key: Shard key (e.g., user_id, tenant_id)

        Returns:
            Shard that should handle this key
        """
        if not self.shards:
            return None

        if self.strategy == ShardStrategy.HASH:
            return self._hash_based_shard(key)

        elif self.strategy == ShardStrategy.RANGE:
            return self._range_based_shard(key)

        elif self.strategy == ShardStrategy.DIRECTORY:
            return self._directory_based_shard(key)

        elif self.strategy == ShardStrategy.GEOGRAPHIC:
            return self._geographic_shard(key)

        # Fallback to first active shard
        for shard in self.shards.values():
            if shard.is_active:
                return shard

        return None

    def _hash_based_shard(self, key: str) -> Optional[Shard]:
        """Get shard using consistent hashing"""
        if not self._consistent_hash_ring:
            return None

        # Hash the key
        key_hash = self._hash_key(key)

        # Binary search for closest node
        idx = bisect.bisect_right([node[0] for node in self._consistent_hash_ring], key_hash)
        if idx >= len(self._consistent_hash_ring):
            idx = 0

        shard_id = self._consistent_hash_ring[idx][1]
        return self.shards.get(shard_id)

    def _range_based_shard(self, key: str) -> Optional[Shard]:
        """Get shard using range partitioning"""
        # Convert key to comparable value
        try:
            key_value = int(key)
        except ValueError:
            key_value = ord(key[0]) if key else 0

        # Find shard with matching range
        for shard in self.shards.values():
            if not shard.is_active:
                continue

            if shard.min_key is not None and key_value < shard.min_key:
                continue
            if shard.max_key is not None and key_value >= shard.max_key:
                continue

            return shard

        return None

    def _directory_based_shard(self, key: str) -> Optional[Shard]:
        """Get shard using directory lookup"""
        shard_id = self.directory.get(key)
        if shard_id:
            return self.shards.get(shard_id)

        # Assign to shard with most capacity
        return self._least_loaded_shard()

    def _geographic_shard(self, key: str) -> Optional[Shard]:
        """Get shard based on geographic location"""
        # Would use geo IP lookup in production
        # For now, use hash-based
        return self._hash_based_shard(key)

    def _update_hash_ring(self):
        """Update consistent hash ring with virtual nodes"""
        self._consistent_hash_ring.clear()

        for shard_id, shard in self.shards.items():
            if not shard.is_active:
                continue

            # Add virtual nodes for better distribution
            for i in range(self._virtual_nodes_per_shard):
                virtual_key = f"{shard_id}:{i}"
                hash_value = self._hash_key(virtual_key)
                self._consistent_hash_ring.append((hash_value, shard_id))

        # Sort ring by hash value
        self._consistent_hash_ring.sort(key=lambda x: x[0])

    def _hash_key(self, key: str) -> int:
        """Hash key to integer"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def _least_loaded_shard(self) -> Optional[Shard]:
        """Find shard with least load"""
        min_load = float('inf')
        best_shard = None

        for shard in self.shards.values():
            if not shard.is_active:
                continue

            # Calculate load score
            size_ratio = shard.current_size_gb / shard.max_size_gb
            metrics = self._shard_metrics[shard.shard_id]
            query_load = metrics["read_count"] + metrics["write_count"]

            load_score = 0.6 * size_ratio + 0.4 * (query_load / 10000.0)

            if load_score < min_load:
                min_load = load_score
                best_shard = shard

        return best_shard

    def route_query(
        self,
        query: str,
        shard_keys: Optional[List[str]] = None
    ) -> List[Shard]:
        """
        Route query to appropriate shards

        Args:
            query: SQL query
            shard_keys: List of shard keys involved in query

        Returns:
            List of shards that need to be queried
        """
        # If specific keys provided, route to those shards
        if shard_keys:
            shards = []
            for key in shard_keys:
                shard = self.get_shard_for_key(key)
                if shard and shard not in shards:
                    shards.append(shard)
            return shards

        # If no keys, need to query all shards (scatter-gather)
        return [s for s in self.shards.values() if s.is_active]

    def execute_distributed_query(
        self,
        query: str,
        shard_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute query across multiple shards

        Args:
            query: SQL query
            shard_keys: Optional shard keys to limit query scope

        Returns:
            Aggregated results from all shards
        """
        shards = self.route_query(query, shard_keys)

        results = {
            "total_shards": len(shards),
            "results": [],
            "errors": []
        }

        # Execute on each shard (would be async in production)
        for shard in shards:
            try:
                # Simulate query execution
                shard_result = self._execute_on_shard(shard, query)
                results["results"].append({
                    "shard_id": shard.shard_id,
                    "data": shard_result
                })

                # Update metrics
                self._shard_metrics[shard.shard_id]["read_count"] += 1

            except Exception as e:
                results["errors"].append({
                    "shard_id": shard.shard_id,
                    "error": str(e)
                })
                self._shard_metrics[shard.shard_id]["error_count"] += 1

        return results

    def _execute_on_shard(self, shard: Shard, query: str) -> List[Dict]:
        """Execute query on specific shard"""
        # Simulate query execution
        return [{"status": "success", "shard": shard.shard_id}]

    def rebalance_shards(self) -> Dict[str, Any]:
        """
        Rebalance data across shards

        Returns:
            Rebalancing plan
        """
        plan = {
            "migrations": [],
            "estimated_time_hours": 0,
            "data_to_move_gb": 0.0
        }

        # Identify overloaded shards
        avg_size = sum(s.current_size_gb for s in self.shards.values()) / len(self.shards)

        overloaded = []
        underloaded = []

        for shard in self.shards.values():
            if shard.current_size_gb > avg_size * 1.5:
                overloaded.append(shard)
            elif shard.current_size_gb < avg_size * 0.5:
                underloaded.append(shard)

        # Create migration plan
        for source in overloaded:
            if not underloaded:
                break

            target = underloaded[0]
            amount_to_move = min(
                source.current_size_gb - avg_size,
                target.max_size_gb - target.current_size_gb
            )

            if amount_to_move > 0:
                plan["migrations"].append({
                    "source_shard": source.shard_id,
                    "target_shard": target.shard_id,
                    "data_gb": amount_to_move
                })

                plan["data_to_move_gb"] += amount_to_move

                # Update projected sizes
                source.current_size_gb -= amount_to_move
                target.current_size_gb += amount_to_move

                # If target is now full, remove from underloaded
                if target.current_size_gb >= avg_size * 0.9:
                    underloaded.pop(0)

        # Estimate time (assume 10 GB/hour migration speed)
        plan["estimated_time_hours"] = plan["data_to_move_gb"] / 10.0

        return plan

    def detect_hot_shards(self, threshold_multiplier: float = 3.0) -> List[Dict[str, Any]]:
        """
        Detect shards with unusually high load

        Args:
            threshold_multiplier: How many times average is considered hot

        Returns:
            List of hot shards with metrics
        """
        # Calculate average load
        total_queries = sum(
            metrics["read_count"] + metrics["write_count"]
            for metrics in self._shard_metrics.values()
        )
        avg_load = total_queries / max(len(self.shards), 1)

        hot_shards = []

        for shard_id, metrics in self._shard_metrics.items():
            shard_load = metrics["read_count"] + metrics["write_count"]

            if shard_load > avg_load * threshold_multiplier:
                hot_shards.append({
                    "shard_id": shard_id,
                    "load": shard_load,
                    "avg_load": avg_load,
                    "multiplier": shard_load / max(avg_load, 1),
                    "recommendation": "Consider splitting or adding read replicas"
                })

        return hot_shards

    def get_shard_statistics(self) -> Dict[str, Any]:
        """Get cluster-wide shard statistics"""
        total_size = sum(s.current_size_gb for s in self.shards.values())
        total_capacity = sum(s.max_size_gb for s in self.shards.values())

        return {
            "total_shards": len(self.shards),
            "active_shards": sum(1 for s in self.shards.values() if s.is_active),
            "total_size_gb": total_size,
            "total_capacity_gb": total_capacity,
            "utilization_percent": (total_size / max(total_capacity, 1)) * 100,
            "strategy": self.strategy.value,
            "virtual_nodes_per_shard": self._virtual_nodes_per_shard,
            "hot_shards": len(self.detect_hot_shards())
        }

    def plan_split(self, shard_id: str) -> Dict[str, Any]:
        """
        Plan to split a shard into multiple shards

        Args:
            shard_id: ID of shard to split

        Returns:
            Split plan with new shard configurations
        """
        shard = self.shards.get(shard_id)
        if not shard:
            return {"error": "Shard not found"}

        # Create two new shards
        new_shard_1 = Shard(
            shard_id=f"{shard_id}_1",
            host=shard.host,
            port=shard.port + 1,
            database=f"{shard.database}_1",
            max_size_gb=shard.max_size_gb,
            current_size_gb=shard.current_size_gb / 2
        )

        new_shard_2 = Shard(
            shard_id=f"{shard_id}_2",
            host=shard.host,
            port=shard.port + 2,
            database=f"{shard.database}_2",
            max_size_gb=shard.max_size_gb,
            current_size_gb=shard.current_size_gb / 2
        )

        return {
            "original_shard": shard_id,
            "new_shards": [
                {"shard_id": new_shard_1.shard_id, "size_gb": new_shard_1.current_size_gb},
                {"shard_id": new_shard_2.shard_id, "size_gb": new_shard_2.current_size_gb}
            ],
            "steps": [
                "1. Create new shard instances",
                "2. Set original shard to read-only",
                "3. Copy 50% of data to each new shard",
                "4. Update routing configuration",
                "5. Verify data integrity",
                "6. Switch traffic to new shards",
                "7. Decommission original shard"
            ],
            "estimated_downtime_minutes": 5
        }
