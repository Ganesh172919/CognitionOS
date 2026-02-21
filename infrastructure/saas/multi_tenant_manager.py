"""
Enterprise Multi-Tenant Management System
Advanced tenant isolation, resource management, and cross-tenant operations
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from enum import Enum
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio
import json
import hashlib


class TenantStatus(Enum):
    """Tenant status"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    TRIAL = "trial"
    CHURNED = "churned"
    DELETED = "deleted"
    PENDING_ACTIVATION = "pending_activation"


class TenantType(Enum):
    """Type of tenant"""
    INDIVIDUAL = "individual"
    SMALL_BUSINESS = "small_business"
    ENTERPRISE = "enterprise"
    RESELLER = "reseller"
    PARTNER = "partner"


class IsolationLevel(Enum):
    """Data isolation level"""
    SHARED = "shared"  # Shared tables with tenant_id
    SCHEMA = "schema"  # Dedicated schema per tenant
    DATABASE = "database"  # Dedicated database per tenant
    CLUSTER = "cluster"  # Dedicated cluster per tenant


@dataclass
class TenantConfig:
    """Tenant configuration"""
    isolation_level: IsolationLevel
    custom_domain: Optional[str] = None
    white_label_enabled: bool = False
    sso_enabled: bool = False
    sso_provider: Optional[str] = None
    custom_branding: Dict[str, Any] = field(default_factory=dict)
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    resource_quotas: Dict[str, int] = field(default_factory=dict)
    security_settings: Dict[str, Any] = field(default_factory=dict)
    compliance_requirements: List[str] = field(default_factory=list)
    backup_policy: Dict[str, Any] = field(default_factory=dict)
    notification_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TenantResourceUsage:
    """Resource usage tracking"""
    tenant_id: str
    timestamp: datetime
    cpu_usage: Decimal
    memory_usage_mb: Decimal
    storage_usage_gb: Decimal
    network_ingress_gb: Decimal
    network_egress_gb: Decimal
    database_connections: int
    active_users: int
    api_calls: int
    compute_minutes: Decimal
    cost_estimate: Decimal = field(default_factory=lambda: Decimal("0"))


@dataclass
class Tenant:
    """Enterprise tenant entity"""
    tenant_id: str
    name: str
    type: TenantType
    status: TenantStatus
    config: TenantConfig
    owner_user_id: str
    admin_user_ids: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    trial_end_date: Optional[datetime] = None
    subscription_id: Optional[str] = None
    parent_tenant_id: Optional[str] = None  # For hierarchical tenants
    child_tenant_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    suspended_reason: Optional[str] = None
    suspended_at: Optional[datetime] = None

    def is_active(self) -> bool:
        """Check if tenant is active"""
        return self.status == TenantStatus.ACTIVE

    def is_trial(self) -> bool:
        """Check if in trial period"""
        if self.status != TenantStatus.TRIAL:
            return False
        if not self.trial_end_date:
            return False
        return datetime.utcnow() < self.trial_end_date

    def days_until_trial_end(self) -> int:
        """Days until trial ends"""
        if not self.trial_end_date:
            return 0
        delta = self.trial_end_date - datetime.utcnow()
        return max(delta.days, 0)


@dataclass
class TenantUser:
    """User within a tenant"""
    user_id: str
    tenant_id: str
    email: str
    role: str  # admin, member, readonly
    permissions: List[str] = field(default_factory=list)
    is_active: bool = True
    last_login: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossTenantOperation:
    """Cross-tenant operation record"""
    operation_id: str
    source_tenant_id: str
    target_tenant_ids: List[str]
    operation_type: str  # data_share, resource_transfer, etc.
    status: str  # pending, approved, rejected, completed
    requested_by: str
    approved_by: Optional[str] = None
    requested_at: datetime = field(default_factory=datetime.utcnow)
    approved_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TenantIsolationManager:
    """Manages tenant data isolation"""

    def __init__(self):
        self.isolation_strategies = {
            IsolationLevel.SHARED: self._shared_isolation,
            IsolationLevel.SCHEMA: self._schema_isolation,
            IsolationLevel.DATABASE: self._database_isolation,
            IsolationLevel.CLUSTER: self._cluster_isolation,
        }

    async def get_connection_string(
        self,
        tenant: Tenant,
        base_connection: str
    ) -> str:
        """Get appropriate connection string for tenant"""
        level = tenant.config.isolation_level

        if level == IsolationLevel.SHARED:
            return base_connection

        elif level == IsolationLevel.SCHEMA:
            # Use schema-specific connection
            return f"{base_connection}?options=-c search_path={tenant.tenant_id}"

        elif level == IsolationLevel.DATABASE:
            # Replace database name
            return base_connection.replace("/cognition_os", f"/tenant_{tenant.tenant_id}")

        elif level == IsolationLevel.CLUSTER:
            # Use dedicated cluster endpoint
            return tenant.metadata.get("cluster_endpoint", base_connection)

        return base_connection

    async def ensure_isolation(
        self,
        tenant: Tenant
    ) -> Dict[str, Any]:
        """Ensure tenant isolation is properly configured"""
        strategy = self.isolation_strategies.get(tenant.config.isolation_level)
        if not strategy:
            raise ValueError(f"Unknown isolation level: {tenant.config.isolation_level}")

        return await strategy(tenant)

    async def _shared_isolation(self, tenant: Tenant) -> Dict[str, Any]:
        """Set up shared table isolation"""
        # Ensure tenant_id column exists and is indexed
        return {
            "isolation_level": "shared",
            "tenant_id": tenant.tenant_id,
            "row_level_security": True
        }

    async def _schema_isolation(self, tenant: Tenant) -> Dict[str, Any]:
        """Set up schema-level isolation"""
        schema_name = f"tenant_{tenant.tenant_id}"

        # Create schema if not exists
        # In real implementation, would execute SQL
        return {
            "isolation_level": "schema",
            "schema_name": schema_name,
            "created": True
        }

    async def _database_isolation(self, tenant: Tenant) -> Dict[str, Any]:
        """Set up database-level isolation"""
        db_name = f"tenant_{tenant.tenant_id}"

        # Create database if not exists
        # In real implementation, would execute SQL
        return {
            "isolation_level": "database",
            "database_name": db_name,
            "created": True
        }

    async def _cluster_isolation(self, tenant: Tenant) -> Dict[str, Any]:
        """Set up cluster-level isolation"""
        # Provision dedicated cluster
        # In real implementation, would call cloud provider API
        return {
            "isolation_level": "cluster",
            "cluster_endpoint": tenant.metadata.get("cluster_endpoint"),
            "provisioned": True
        }


class TenantResourceManager:
    """Manages tenant resource allocation and quotas"""

    def __init__(self):
        self.resource_usage: Dict[str, List[TenantResourceUsage]] = {}
        self.quota_violations: Dict[str, List[Dict[str, Any]]] = {}

    async def track_resource_usage(
        self,
        tenant_id: str,
        cpu_usage: Decimal,
        memory_usage_mb: Decimal,
        storage_usage_gb: Decimal,
        network_ingress_gb: Decimal,
        network_egress_gb: Decimal,
        database_connections: int,
        active_users: int,
        api_calls: int,
        compute_minutes: Decimal
    ) -> TenantResourceUsage:
        """Track resource usage for tenant"""
        # Calculate cost estimate
        cost = self._calculate_cost(
            cpu_usage=cpu_usage,
            memory_usage_mb=memory_usage_mb,
            storage_usage_gb=storage_usage_gb,
            network_ingress_gb=network_ingress_gb,
            network_egress_gb=network_egress_gb,
            compute_minutes=compute_minutes
        )

        usage = TenantResourceUsage(
            tenant_id=tenant_id,
            timestamp=datetime.utcnow(),
            cpu_usage=cpu_usage,
            memory_usage_mb=memory_usage_mb,
            storage_usage_gb=storage_usage_gb,
            network_ingress_gb=network_ingress_gb,
            network_egress_gb=network_egress_gb,
            database_connections=database_connections,
            active_users=active_users,
            api_calls=api_calls,
            compute_minutes=compute_minutes,
            cost_estimate=cost
        )

        if tenant_id not in self.resource_usage:
            self.resource_usage[tenant_id] = []

        self.resource_usage[tenant_id].append(usage)

        return usage

    def _calculate_cost(
        self,
        cpu_usage: Decimal,
        memory_usage_mb: Decimal,
        storage_usage_gb: Decimal,
        network_ingress_gb: Decimal,
        network_egress_gb: Decimal,
        compute_minutes: Decimal
    ) -> Decimal:
        """Calculate resource cost"""
        # Simplified cost calculation
        cost = Decimal("0")
        cost += cpu_usage * Decimal("0.05")  # $0.05 per CPU hour
        cost += (memory_usage_mb / 1024) * Decimal("0.01")  # $0.01 per GB-hour
        cost += storage_usage_gb * Decimal("0.10")  # $0.10 per GB
        cost += network_egress_gb * Decimal("0.09")  # $0.09 per GB
        cost += (compute_minutes / 60) * Decimal("0.50")  # $0.50 per compute hour

        return cost

    async def check_quota_violations(
        self,
        tenant: Tenant
    ) -> List[Dict[str, Any]]:
        """Check if tenant has exceeded quotas"""
        violations = []
        quotas = tenant.config.resource_quotas

        # Get recent usage
        recent_usage = [
            u for u in self.resource_usage.get(tenant.tenant_id, [])
            if u.timestamp >= datetime.utcnow() - timedelta(hours=1)
        ]

        if not recent_usage:
            return violations

        latest_usage = recent_usage[-1]

        # Check each quota
        if "max_storage_gb" in quotas:
            if latest_usage.storage_usage_gb > quotas["max_storage_gb"]:
                violations.append({
                    "type": "storage",
                    "current": float(latest_usage.storage_usage_gb),
                    "limit": quotas["max_storage_gb"],
                    "severity": "high"
                })

        if "max_api_calls_per_hour" in quotas:
            total_api_calls = sum(u.api_calls for u in recent_usage)
            if total_api_calls > quotas["max_api_calls_per_hour"]:
                violations.append({
                    "type": "api_calls",
                    "current": total_api_calls,
                    "limit": quotas["max_api_calls_per_hour"],
                    "severity": "medium"
                })

        if "max_active_users" in quotas:
            if latest_usage.active_users > quotas["max_active_users"]:
                violations.append({
                    "type": "active_users",
                    "current": latest_usage.active_users,
                    "limit": quotas["max_active_users"],
                    "severity": "low"
                })

        # Store violations
        if violations:
            if tenant.tenant_id not in self.quota_violations:
                self.quota_violations[tenant.tenant_id] = []
            self.quota_violations[tenant.tenant_id].extend(violations)

        return violations

    async def get_resource_report(
        self,
        tenant_id: str,
        period_hours: int = 24
    ) -> Dict[str, Any]:
        """Get resource usage report"""
        cutoff = datetime.utcnow() - timedelta(hours=period_hours)
        usage_records = [
            u for u in self.resource_usage.get(tenant_id, [])
            if u.timestamp >= cutoff
        ]

        if not usage_records:
            return {"error": "No usage data available"}

        total_cost = sum(u.cost_estimate for u in usage_records)
        avg_cpu = sum(u.cpu_usage for u in usage_records) / len(usage_records)
        avg_memory = sum(u.memory_usage_mb for u in usage_records) / len(usage_records)
        max_storage = max(u.storage_usage_gb for u in usage_records)
        total_api_calls = sum(u.api_calls for u in usage_records)

        return {
            "tenant_id": tenant_id,
            "period_hours": period_hours,
            "total_cost": float(total_cost),
            "avg_cpu_usage": float(avg_cpu),
            "avg_memory_mb": float(avg_memory),
            "max_storage_gb": float(max_storage),
            "total_api_calls": total_api_calls,
            "data_points": len(usage_records)
        }


class MultiTenantManager:
    """Main multi-tenant management system"""

    def __init__(self):
        self.tenants: Dict[str, Tenant] = {}
        self.tenant_users: Dict[str, List[TenantUser]] = {}
        self.isolation_manager = TenantIsolationManager()
        self.resource_manager = TenantResourceManager()
        self.cross_tenant_operations: Dict[str, CrossTenantOperation] = {}

    async def create_tenant(
        self,
        name: str,
        type: TenantType,
        owner_user_id: str,
        isolation_level: IsolationLevel = IsolationLevel.SHARED,
        trial_days: int = 14,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tenant:
        """Create new tenant"""
        import uuid

        tenant_id = str(uuid.uuid4())

        # Determine status
        status = TenantStatus.TRIAL if trial_days > 0 else TenantStatus.PENDING_ACTIVATION

        # Set trial end date
        trial_end_date = None
        if trial_days > 0:
            trial_end_date = datetime.utcnow() + timedelta(days=trial_days)

        # Create default config
        config = TenantConfig(
            isolation_level=isolation_level,
            resource_quotas=self._get_default_quotas(type),
            feature_flags=self._get_default_features(type)
        )

        tenant = Tenant(
            tenant_id=tenant_id,
            name=name,
            type=type,
            status=status,
            config=config,
            owner_user_id=owner_user_id,
            admin_user_ids=[owner_user_id],
            trial_end_date=trial_end_date,
            metadata=metadata or {}
        )

        # Ensure isolation
        await self.isolation_manager.ensure_isolation(tenant)

        self.tenants[tenant_id] = tenant

        return tenant

    def _get_default_quotas(self, tenant_type: TenantType) -> Dict[str, int]:
        """Get default quotas based on tenant type"""
        quotas = {
            TenantType.INDIVIDUAL: {
                "max_storage_gb": 10,
                "max_api_calls_per_hour": 1000,
                "max_active_users": 3,
                "max_compute_hours": 10
            },
            TenantType.SMALL_BUSINESS: {
                "max_storage_gb": 100,
                "max_api_calls_per_hour": 10000,
                "max_active_users": 25,
                "max_compute_hours": 100
            },
            TenantType.ENTERPRISE: {
                "max_storage_gb": 1000,
                "max_api_calls_per_hour": 100000,
                "max_active_users": 500,
                "max_compute_hours": 1000
            },
            TenantType.RESELLER: {
                "max_storage_gb": 5000,
                "max_api_calls_per_hour": 500000,
                "max_active_users": 2000,
                "max_compute_hours": 5000
            },
            TenantType.PARTNER: {
                "max_storage_gb": 10000,
                "max_api_calls_per_hour": 1000000,
                "max_active_users": 5000,
                "max_compute_hours": 10000
            }
        }

        return quotas.get(tenant_type, quotas[TenantType.INDIVIDUAL])

    def _get_default_features(self, tenant_type: TenantType) -> Dict[str, bool]:
        """Get default feature flags based on tenant type"""
        features = {
            "advanced_analytics": tenant_type in [TenantType.ENTERPRISE, TenantType.PARTNER],
            "custom_branding": tenant_type in [TenantType.ENTERPRISE, TenantType.PARTNER],
            "sso": tenant_type in [TenantType.ENTERPRISE, TenantType.PARTNER],
            "api_access": True,
            "webhook_support": tenant_type != TenantType.INDIVIDUAL,
            "audit_logs": tenant_type in [TenantType.ENTERPRISE, TenantType.PARTNER],
            "priority_support": tenant_type in [TenantType.ENTERPRISE, TenantType.PARTNER]
        }

        return features

    async def suspend_tenant(
        self,
        tenant_id: str,
        reason: str
    ) -> Tenant:
        """Suspend tenant"""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            raise ValueError("Tenant not found")

        tenant.status = TenantStatus.SUSPENDED
        tenant.suspended_reason = reason
        tenant.suspended_at = datetime.utcnow()
        tenant.updated_at = datetime.utcnow()

        return tenant

    async def activate_tenant(
        self,
        tenant_id: str,
        subscription_id: Optional[str] = None
    ) -> Tenant:
        """Activate tenant"""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            raise ValueError("Tenant not found")

        tenant.status = TenantStatus.ACTIVE
        tenant.subscription_id = subscription_id
        tenant.suspended_reason = None
        tenant.suspended_at = None
        tenant.updated_at = datetime.utcnow()

        return tenant

    async def add_user_to_tenant(
        self,
        tenant_id: str,
        user_id: str,
        email: str,
        role: str,
        permissions: Optional[List[str]] = None
    ) -> TenantUser:
        """Add user to tenant"""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            raise ValueError("Tenant not found")

        user = TenantUser(
            user_id=user_id,
            tenant_id=tenant_id,
            email=email,
            role=role,
            permissions=permissions or []
        )

        if tenant_id not in self.tenant_users:
            self.tenant_users[tenant_id] = []

        self.tenant_users[tenant_id].append(user)

        if role == "admin" and user_id not in tenant.admin_user_ids:
            tenant.admin_user_ids.append(user_id)
            tenant.updated_at = datetime.utcnow()

        return user

    async def create_cross_tenant_operation(
        self,
        source_tenant_id: str,
        target_tenant_ids: List[str],
        operation_type: str,
        requested_by: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CrossTenantOperation:
        """Create cross-tenant operation request"""
        import uuid

        # Validate tenants exist
        if source_tenant_id not in self.tenants:
            raise ValueError("Source tenant not found")

        for target_id in target_tenant_ids:
            if target_id not in self.tenants:
                raise ValueError(f"Target tenant {target_id} not found")

        operation = CrossTenantOperation(
            operation_id=str(uuid.uuid4()),
            source_tenant_id=source_tenant_id,
            target_tenant_ids=target_tenant_ids,
            operation_type=operation_type,
            status="pending",
            requested_by=requested_by,
            metadata=metadata or {}
        )

        self.cross_tenant_operations[operation.operation_id] = operation

        return operation

    async def approve_cross_tenant_operation(
        self,
        operation_id: str,
        approved_by: str
    ) -> CrossTenantOperation:
        """Approve cross-tenant operation"""
        operation = self.cross_tenant_operations.get(operation_id)
        if not operation:
            raise ValueError("Operation not found")

        operation.status = "approved"
        operation.approved_by = approved_by
        operation.approved_at = datetime.utcnow()

        # Execute operation
        await self._execute_cross_tenant_operation(operation)

        return operation

    async def _execute_cross_tenant_operation(
        self,
        operation: CrossTenantOperation
    ):
        """Execute approved cross-tenant operation"""
        # Simulate execution
        await asyncio.sleep(0.1)

        operation.status = "completed"
        operation.completed_at = datetime.utcnow()

    async def get_tenant_hierarchy(
        self,
        tenant_id: str
    ) -> Dict[str, Any]:
        """Get tenant hierarchy (parent and children)"""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            raise ValueError("Tenant not found")

        result = {
            "tenant_id": tenant_id,
            "name": tenant.name,
            "type": tenant.type.value,
            "parent": None,
            "children": []
        }

        # Get parent
        if tenant.parent_tenant_id:
            parent = self.tenants.get(tenant.parent_tenant_id)
            if parent:
                result["parent"] = {
                    "tenant_id": parent.tenant_id,
                    "name": parent.name,
                    "type": parent.type.value
                }

        # Get children
        for child_id in tenant.child_tenant_ids:
            child = self.tenants.get(child_id)
            if child:
                result["children"].append({
                    "tenant_id": child.tenant_id,
                    "name": child.name,
                    "type": child.type.value,
                    "status": child.status.value
                })

        return result

    async def get_tenant_analytics(
        self,
        tenant_id: str
    ) -> Dict[str, Any]:
        """Get comprehensive tenant analytics"""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            raise ValueError("Tenant not found")

        # Get resource usage
        resource_report = await self.resource_manager.get_resource_report(tenant_id)

        # Get quota violations
        violations = await self.resource_manager.check_quota_violations(tenant)

        # Get user stats
        users = self.tenant_users.get(tenant_id, [])
        active_users = [u for u in users if u.is_active]

        return {
            "tenant_id": tenant_id,
            "name": tenant.name,
            "type": tenant.type.value,
            "status": tenant.status.value,
            "is_trial": tenant.is_trial(),
            "days_until_trial_end": tenant.days_until_trial_end() if tenant.is_trial() else None,
            "user_count": len(users),
            "active_user_count": len(active_users),
            "resource_usage": resource_report,
            "quota_violations": violations,
            "features_enabled": [k for k, v in tenant.config.feature_flags.items() if v]
        }
