"""
REST API Routes — CognitionOS

FastAPI-compatible route definitions for all platform endpoints.
This module wires together all platform services into a unified API layer.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RouteDefinition:
    method: str
    path: str
    handler_name: str
    description: str = ""
    auth_required: bool = True
    rate_limit: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    version: str = "v1"


# ── All Platform API Routes ──

PLATFORM_ROUTES: List[RouteDefinition] = [
    # ── Health & Status ──
    RouteDefinition("GET", "/health", "health_check",
                     description="Platform health check", auth_required=False,
                     tags=["system"]),
    RouteDefinition("GET", "/status", "platform_status",
                     description="Full platform status", tags=["system"]),
    RouteDefinition("GET", "/dashboard", "get_dashboard",
                     description="Platform dashboard data", tags=["system"]),

    # ── Agents ──
    RouteDefinition("POST", "/agents/sessions", "create_agent_session",
                     description="Create new agent session", tags=["agents"]),
    RouteDefinition("POST", "/agents/sessions/{session_id}/messages", "send_message",
                     description="Send message to agent", tags=["agents"]),
    RouteDefinition("GET", "/agents/sessions/{session_id}", "get_session",
                     description="Get session details", tags=["agents"]),
    RouteDefinition("DELETE", "/agents/sessions/{session_id}", "end_session",
                     description="End agent session", tags=["agents"]),
    RouteDefinition("GET", "/agents/stats", "agent_stats",
                     description="Agent usage statistics", tags=["agents"]),

    # ── Code Generation ──
    RouteDefinition("POST", "/codegen/generate", "generate_code",
                     description="Generate code from requirements", tags=["codegen"],
                     rate_limit=20),
    RouteDefinition("POST", "/codegen/validate", "validate_code",
                     description="Validate generated code", tags=["codegen"]),
    RouteDefinition("POST", "/codegen/refactor", "refactor_code",
                     description="AI-powered code refactoring", tags=["codegen"]),

    # ── Workflows ──
    RouteDefinition("POST", "/workflows", "create_workflow",
                     description="Create workflow definition", tags=["workflows"]),
    RouteDefinition("GET", "/workflows", "list_workflows",
                     description="List all workflows", tags=["workflows"]),
    RouteDefinition("POST", "/workflows/{workflow_id}/execute", "execute_workflow",
                     description="Execute a workflow", tags=["workflows"]),
    RouteDefinition("GET", "/workflows/{workflow_id}/executions", "list_executions",
                     description="List workflow executions", tags=["workflows"]),
    RouteDefinition("POST", "/workflows/{execution_id}/pause", "pause_workflow",
                     description="Pause workflow execution", tags=["workflows"]),
    RouteDefinition("POST", "/workflows/{execution_id}/resume", "resume_workflow",
                     description="Resume paused workflow", tags=["workflows"]),

    # ── Tenants ──
    RouteDefinition("POST", "/tenants", "create_tenant",
                     description="Create a new tenant", tags=["tenants"]),
    RouteDefinition("GET", "/tenants", "list_tenants",
                     description="List all tenants", tags=["tenants"]),
    RouteDefinition("GET", "/tenants/{tenant_id}", "get_tenant",
                     description="Get tenant details", tags=["tenants"]),
    RouteDefinition("PUT", "/tenants/{tenant_id}/tier", "upgrade_tenant",
                     description="Upgrade tenant tier", tags=["tenants"]),
    RouteDefinition("POST", "/tenants/{tenant_id}/suspend", "suspend_tenant",
                     description="Suspend a tenant", tags=["tenants"]),
    RouteDefinition("GET", "/tenants/{tenant_id}/usage", "tenant_usage",
                     description="Get tenant usage details", tags=["tenants"]),

    # ── Billing & Costs ──
    RouteDefinition("GET", "/billing/costs", "get_costs",
                     description="Get cost breakdown", tags=["billing"]),
    RouteDefinition("GET", "/billing/forecast", "cost_forecast",
                     description="Get cost forecast", tags=["billing"]),
    RouteDefinition("POST", "/billing/budgets", "set_budget",
                     description="Set billing budget", tags=["billing"]),
    RouteDefinition("GET", "/billing/alerts", "billing_alerts",
                     description="Get billing alerts", tags=["billing"]),
    RouteDefinition("GET", "/billing/optimizations", "cost_optimizations",
                     description="Get cost optimization suggestions", tags=["billing"]),

    # ── Feature Flags ──
    RouteDefinition("POST", "/flags", "create_flag",
                     description="Create feature flag", tags=["flags"]),
    RouteDefinition("GET", "/flags", "list_flags",
                     description="List all flags", tags=["flags"]),
    RouteDefinition("GET", "/flags/{flag_key}/evaluate", "evaluate_flag",
                     description="Evaluate a feature flag", tags=["flags"]),
    RouteDefinition("PUT", "/flags/{flag_key}", "update_flag",
                     description="Update a feature flag", tags=["flags"]),
    RouteDefinition("GET", "/flags/stats", "flag_stats",
                     description="Flag evaluation stats", tags=["flags"]),

    # ── Webhooks ──
    RouteDefinition("POST", "/webhooks/subscriptions", "create_webhook",
                     description="Create webhook subscription", tags=["webhooks"]),
    RouteDefinition("GET", "/webhooks/subscriptions", "list_webhooks",
                     description="List webhook subscriptions", tags=["webhooks"]),
    RouteDefinition("DELETE", "/webhooks/subscriptions/{sub_id}", "delete_webhook",
                     description="Delete webhook subscription", tags=["webhooks"]),
    RouteDefinition("POST", "/webhooks/{sub_id}/test", "test_webhook",
                     description="Test webhook delivery", tags=["webhooks"]),
    RouteDefinition("GET", "/webhooks/deliveries", "webhook_deliveries",
                     description="List webhook deliveries", tags=["webhooks"]),

    # ── Notifications ──
    RouteDefinition("POST", "/notifications/send", "send_notification",
                     description="Send a notification", tags=["notifications"]),
    RouteDefinition("GET", "/notifications/inbox", "get_inbox",
                     description="Get notification inbox", tags=["notifications"]),
    RouteDefinition("POST", "/notifications/{id}/read", "mark_read",
                     description="Mark notification as read", tags=["notifications"]),
    RouteDefinition("PUT", "/notifications/preferences", "update_prefs",
                     description="Update notification preferences", tags=["notifications"]),

    # ── Analytics ──
    RouteDefinition("POST", "/analytics/track", "track_event",
                     description="Track an analytics event", tags=["analytics"]),
    RouteDefinition("GET", "/analytics/funnel", "analyze_funnel",
                     description="Analyze conversion funnel", tags=["analytics"]),
    RouteDefinition("GET", "/analytics/usage", "usage_stats",
                     description="Get usage statistics", tags=["analytics"]),
    RouteDefinition("GET", "/analytics/revenue", "revenue_metrics",
                     description="Get revenue metrics", tags=["analytics"]),
    RouteDefinition("GET", "/analytics/engagement", "engagement_metrics",
                     description="Get user engagement metrics", tags=["analytics"]),

    # ── Recommendations ──
    RouteDefinition("GET", "/recommendations", "get_recommendations",
                     description="Get personalized recommendations", tags=["intelligence"]),
    RouteDefinition("POST", "/recommendations/{id}/feedback", "rec_feedback",
                     description="Submit recommendation feedback", tags=["intelligence"]),

    # ── Plugins ──
    RouteDefinition("GET", "/plugins", "list_plugins",
                     description="List available plugins", tags=["plugins"]),
    RouteDefinition("POST", "/plugins/install", "install_plugin",
                     description="Install a plugin", tags=["plugins"]),
    RouteDefinition("DELETE", "/plugins/{plugin_id}", "uninstall_plugin",
                     description="Uninstall a plugin", tags=["plugins"]),

    # ── Audit ──
    RouteDefinition("GET", "/audit/log", "audit_log",
                     description="Get audit trail", tags=["audit"]),
    RouteDefinition("GET", "/audit/export", "audit_export",
                     description="Export audit log", tags=["audit"]),
    RouteDefinition("GET", "/audit/integrity", "audit_integrity",
                     description="Verify audit trail integrity", tags=["audit"]),

    # ── Growth ──
    RouteDefinition("GET", "/growth/metrics", "growth_metrics",
                     description="Get growth metrics", tags=["growth"]),
    RouteDefinition("GET", "/growth/at-risk", "at_risk_users",
                     description="Get at-risk users", tags=["growth"]),
    RouteDefinition("GET", "/growth/experiments", "list_experiments",
                     description="List growth experiments", tags=["growth"]),

    # ── Admin ──
    RouteDefinition("GET", "/admin/config", "get_config",
                     description="Get platform configuration", tags=["admin"]),
    RouteDefinition("PUT", "/admin/config", "update_config",
                     description="Update platform configuration", tags=["admin"]),
    RouteDefinition("GET", "/admin/migrations", "migration_status",
                     description="Get migration status", tags=["admin"]),
    RouteDefinition("POST", "/admin/migrations/run", "run_migrations",
                     description="Run pending migrations", tags=["admin"]),
    RouteDefinition("GET", "/admin/cache/stats", "cache_stats",
                     description="Get cache statistics", tags=["admin"]),
    RouteDefinition("POST", "/admin/cache/invalidate", "cache_invalidate",
                     description="Invalidate cache entries", tags=["admin"]),
    RouteDefinition("GET", "/admin/queues", "queue_stats",
                     description="Get message queue statistics", tags=["admin"]),
]


def get_route_map() -> Dict[str, List[Dict[str, Any]]]:
    """Group routes by tag for documentation."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for route in PLATFORM_ROUTES:
        for tag in route.tags:
            if tag not in groups:
                groups[tag] = []
            groups[tag].append({
                "method": route.method,
                "path": f"/api/{route.version}{route.path}",
                "description": route.description,
                "auth_required": route.auth_required,
                "rate_limit": route.rate_limit,
            })
    return groups


def get_openapi_paths() -> Dict[str, Any]:
    """Generate OpenAPI-compatible path definitions."""
    paths: Dict[str, Any] = {}
    for route in PLATFORM_ROUTES:
        full_path = f"/api/{route.version}{route.path}"
        method = route.method.lower()
        if full_path not in paths:
            paths[full_path] = {}
        paths[full_path][method] = {
            "summary": route.description,
            "operationId": route.handler_name,
            "tags": route.tags,
            "security": [{"bearerAuth": []}] if route.auth_required else [],
        }
    return paths


def print_routes():
    """Print all registered routes (useful for debugging)."""
    for route in PLATFORM_ROUTES:
        auth = "🔐" if route.auth_required else "🔓"
        limit = f"[{route.rate_limit}/min]" if route.rate_limit else ""
        print(f"  {auth} {route.method:6s} /api/{route.version}{route.path} "
              f"— {route.description} {limit}")
