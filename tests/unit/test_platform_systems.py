"""
Unit tests for the new platform systems:
- Multi-Tenant RBAC Engine
- Configuration Management System
- Agent Memory Consolidator
- Multi-Agent Coordination Bus
- Workflow DSL Compiler
- Onboarding Engine
"""

from __future__ import annotations

import asyncio
import pytest
import time

# ──────────────────────────────────────────────────────────────────────────────
# RBAC Engine Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestPermission:
    def test_permission_string_representation(self):
        from infrastructure.auth.rbac import Permission
        p = Permission("workflows", "execute")
        assert str(p) == "workflows:execute:*"

    def test_permission_from_string(self):
        from infrastructure.auth.rbac import Permission
        p = Permission.from_string("workflows:execute:my-workflow")
        assert p.resource == "workflows"
        assert p.action == "execute"
        assert p.qualifier == "my-workflow"

    def test_permission_from_string_two_parts(self):
        from infrastructure.auth.rbac import Permission
        p = Permission.from_string("agents:read")
        assert p.resource == "agents"
        assert p.action == "read"
        assert p.qualifier == "*"

    def test_permission_wildcard_match_action(self):
        from infrastructure.auth.rbac import Permission
        granted = Permission("workflows", "*")
        requested = Permission("workflows", "execute")
        assert granted.matches(requested)

    def test_permission_wildcard_match_resource(self):
        from infrastructure.auth.rbac import Permission
        granted = Permission("*", "read")
        requested = Permission("workflows", "read")
        assert granted.matches(requested)

    def test_permission_no_match(self):
        from infrastructure.auth.rbac import Permission
        granted = Permission("workflows", "read")
        requested = Permission("agents", "read")
        assert not granted.matches(requested)

    def test_permission_exact_match(self):
        from infrastructure.auth.rbac import Permission
        p = Permission("api_keys", "create")
        assert p.matches(p)

    def test_permission_invalid_string(self):
        from infrastructure.auth.rbac import Permission
        with pytest.raises(ValueError):
            Permission.from_string("invalid")


class TestRBACEngine:
    def setup_method(self):
        from infrastructure.auth.rbac import RBACEngine
        self.engine = RBACEngine()

    def test_default_roles_exist(self):
        roles = self.engine.list_roles()
        role_ids = {r.role_id for r in roles}
        assert "super_admin" in role_ids
        assert "tenant_admin" in role_ids
        assert "developer" in role_ids
        assert "viewer" in role_ids

    def test_assign_and_check_permission(self):
        from infrastructure.auth.rbac import Permission
        self.engine.assign_role("user-1", "developer", "tenant-a")
        result = self.engine.check("user-1", Permission("workflows", "read"), "tenant-a")
        assert result.allowed

    def test_tenant_isolation(self):
        from infrastructure.auth.rbac import Permission
        self.engine.assign_role("user-1", "developer", "tenant-a")
        result = self.engine.check("user-1", Permission("workflows", "read"), "tenant-b")
        assert not result.allowed

    def test_super_admin_wildcard(self):
        from infrastructure.auth.rbac import Permission
        self.engine.assign_role("admin-1", "super_admin", "tenant-a")
        result = self.engine.check("admin-1", Permission("anything", "everything"), "tenant-a")
        assert result.allowed

    def test_permission_denied_no_role(self):
        from infrastructure.auth.rbac import Permission
        result = self.engine.check("unknown-user", Permission("workflows", "read"), "tenant-a")
        assert not result.allowed
        assert result.denial_reason is not None

    def test_require_raises_on_denied(self):
        from infrastructure.auth.rbac import Permission, PermissionDeniedError
        with pytest.raises(PermissionDeniedError):
            self.engine.require("no-user", Permission("secrets", "delete"), "tenant-a")

    def test_role_inheritance(self):
        from infrastructure.auth.rbac import Permission
        # Developer inherits from viewer
        self.engine.assign_role("dev-1", "developer", "tenant-x")
        result = self.engine.check("dev-1", Permission("analytics", "read"), "tenant-x")
        assert result.allowed  # viewer has analytics:read, developer inherits it

    def test_revoke_role(self):
        from infrastructure.auth.rbac import Permission
        self.engine.assign_role("user-2", "developer", "tenant-a")
        self.engine.revoke_role("user-2", "developer", "tenant-a")
        result = self.engine.check("user-2", Permission("workflows", "read"), "tenant-a")
        assert not result.allowed

    def test_bulk_check(self):
        from infrastructure.auth.rbac import Permission
        self.engine.assign_role("user-3", "developer", "tenant-a")
        results = self.engine.bulk_check(
            "user-3",
            [Permission("workflows", "read"), Permission("billing", "manage")],
            "tenant-a",
        )
        assert results["workflows:read:*"] is True
        assert results["billing:manage:*"] is False  # Only tenant_admin has this

    def test_get_effective_permissions(self):
        from infrastructure.auth.rbac import Permission
        self.engine.assign_role("user-4", "viewer", "tenant-a")
        perms = self.engine.get_effective_permissions("user-4", "tenant-a")
        perm_strings = [str(p) for p in perms]
        assert "workflows:read:*" in perm_strings

    def test_expired_role_assignment(self):
        from infrastructure.auth.rbac import Permission
        # Assign with expiry in the past
        self.engine.assign_role(
            "user-5", "developer", "tenant-a",
            expires_at=time.time() - 1.0  # already expired
        )
        result = self.engine.check("user-5", Permission("workflows", "read"), "tenant-a")
        assert not result.allowed

    def test_abac_policy_deny(self):
        from infrastructure.auth.rbac import (
            Permission, Policy, PolicyCondition, Effect, PolicyOperator
        )
        from uuid import uuid4
        # Create a deny policy for a specific IP
        policy = Policy(
            policy_id=str(uuid4()),
            name="Block IP 1.2.3.4",
            tenant_id="tenant-a",
            effect=Effect.DENY,
            permissions=[Permission("*", "*")],
            conditions=[PolicyCondition("request.ip", "eq", "1.2.3.4")],
            operator=PolicyOperator.ALL,
        )
        self.engine.add_policy(policy)
        self.engine.assign_role("user-6", "super_admin", "tenant-a")

        # Should be denied due to policy
        result = self.engine.check(
            "user-6", Permission("workflows", "read"), "tenant-a",
            context={"request": {"ip": "1.2.3.4"}}
        )
        assert not result.allowed

        # Different IP should be allowed
        result2 = self.engine.check(
            "user-6", Permission("workflows", "read"), "tenant-a",
            context={"request": {"ip": "5.6.7.8"}}
        )
        assert result2.allowed

    def test_authorization_result_dict(self):
        from infrastructure.auth.rbac import Permission
        self.engine.assign_role("user-7", "viewer", "tenant-a")
        result = self.engine.check("user-7", Permission("workflows", "read"), "tenant-a")
        d = result.to_dict()
        assert "allowed" in d
        assert "principal_id" in d
        assert "evaluation_time_ms" in d


class TestPolicyCondition:
    def test_eq_condition(self):
        from infrastructure.auth.rbac import PolicyCondition
        cond = PolicyCondition("user.role", "eq", "admin")
        assert cond.evaluate({"user": {"role": "admin"}})
        assert not cond.evaluate({"user": {"role": "viewer"}})

    def test_in_condition(self):
        from infrastructure.auth.rbac import PolicyCondition
        cond = PolicyCondition("user.department", "in", ["engineering", "product"])
        assert cond.evaluate({"user": {"department": "engineering"}})
        assert not cond.evaluate({"user": {"department": "sales"}})

    def test_gt_condition(self):
        from infrastructure.auth.rbac import PolicyCondition
        cond = PolicyCondition("request.payload_size", "gt", 1000)
        assert cond.evaluate({"request": {"payload_size": 2000}})
        assert not cond.evaluate({"request": {"payload_size": 500}})

    def test_missing_attribute(self):
        from infrastructure.auth.rbac import PolicyCondition
        cond = PolicyCondition("nonexistent.attr", "eq", "value")
        assert not cond.evaluate({})


# ──────────────────────────────────────────────────────────────────────────────
# Config Manager Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestConfigManager:
    def setup_method(self):
        from infrastructure.config.config_manager import ConfigManager
        self.cfg = ConfigManager(environment="test")

    def test_set_and_get(self):
        self.cfg.set("test.key", "hello")
        assert self.cfg.get("test.key") == "hello"

    def test_default_fallback(self):
        assert self.cfg.get("does.not.exist", "default") == "default"

    def test_get_int(self):
        self.cfg.set("app.port", "8080")
        assert self.cfg.get_int("app.port") == 8080

    def test_get_float(self):
        self.cfg.set("rate.limit", "0.75")
        assert abs(self.cfg.get_float("rate.limit") - 0.75) < 0.001

    def test_get_bool_true(self):
        for truthy in ("true", "1", "yes", "on"):
            self.cfg.set("flag", truthy)
            assert self.cfg.get_bool("flag") is True

    def test_get_bool_false(self):
        self.cfg.set("flag", "false")
        assert self.cfg.get_bool("flag") is False

    def test_get_list_from_csv(self):
        self.cfg.set("allowed.ips", "1.2.3.4, 5.6.7.8, 9.0.1.2")
        result = self.cfg.get_list("allowed.ips")
        assert result == ["1.2.3.4", "5.6.7.8", "9.0.1.2"]

    def test_get_list_from_list(self):
        self.cfg.set("items", ["a", "b", "c"])
        result = self.cfg.get_list("items")
        assert result == ["a", "b", "c"]

    def test_load_defaults(self):
        self.cfg.load_defaults({"database": {"pool_size": 10, "timeout": 30}})
        assert self.cfg.get_int("database.pool_size") == 10
        assert self.cfg.get_int("database.timeout") == 30

    def test_load_defaults_does_not_overwrite(self):
        self.cfg.set("existing.key", "runtime_value")
        self.cfg.load_defaults({"existing": {"key": "default_value"}})
        assert self.cfg.get("existing.key") == "runtime_value"

    def test_secret_masking(self):
        from infrastructure.config.config_manager import ConfigType
        self.cfg.set("api.secret", "super-secret-key-123", secret=True, config_type=ConfigType.SECRET)
        entry = self.cfg._entries["api.secret"]
        masked = entry.masked_value
        assert "super" not in masked
        assert "***" in masked

    def test_namespace(self):
        self.cfg.set("database.host", "localhost")
        self.cfg.set("database.port", "5432")
        ns = self.cfg.namespace("database")
        assert ns.get("host") == "localhost"
        assert ns.get_int("port") == 5432

    def test_change_listener(self):
        events = []
        self.cfg.on_change("*", lambda e: events.append(e))
        self.cfg.set("tracked.key", "value1")
        self.cfg.set("tracked.key", "value2")
        assert len(events) == 2
        assert events[0].old_value is None
        assert events[1].old_value == "value1"

    def test_specific_listener(self):
        events = []
        self.cfg.on_change("specific.key", lambda e: events.append(e))
        self.cfg.set("specific.key", "triggered")
        self.cfg.set("other.key", "not-triggered")
        assert len(events) == 1

    def test_immutable_key_cannot_be_overridden(self):
        from infrastructure.config.config_manager import ConfigImmutableError, ConfigType
        self.cfg.set("immutable.key", "original", immutable=True)
        with pytest.raises(ConfigImmutableError):
            self.cfg.set("immutable.key", "changed")

    def test_tenant_override(self):
        self.cfg.set("feature.limit", "100")
        self.cfg.set_tenant_override("tenant-premium", "feature.limit", "10000")
        assert self.cfg.get_for_tenant("tenant-premium", "feature.limit") == "10000"
        assert self.cfg.get_for_tenant("tenant-free", "feature.limit") == "100"

    def test_change_history(self):
        self.cfg.set("history.key", "v1")
        self.cfg.set("history.key", "v2")
        self.cfg.set("history.key", "v3")
        history = self.cfg.get_change_history("history.key")
        assert len(history) == 3

    def test_checksum(self):
        self.cfg.set("key.a", "value1")
        c1 = self.cfg.checksum()
        self.cfg.set("key.b", "value2")
        c2 = self.cfg.checksum()
        assert c1 != c2

    def test_snapshot(self):
        self.cfg.set("snap.key", "snap-value")
        snapshot = self.cfg.snapshot()
        assert "environment" in snapshot
        assert "total_keys" in snapshot
        assert "config" in snapshot


# ──────────────────────────────────────────────────────────────────────────────
# Memory Consolidator Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestMemoryConsolidator:
    def setup_method(self):
        from infrastructure.agent.vector_memory import VectorMemoryStore, MemoryTier, MemoryType
        from infrastructure.agent.memory_consolidator import MemoryConsolidator
        self.store = VectorMemoryStore()
        self.consolidator = MemoryConsolidator(self.store)
        self.MemoryTier = MemoryTier
        self.MemoryType = MemoryType

    def test_consolidation_returns_stats(self):
        self.store.store(
            "Test fact one", self.MemoryType.FACT, self.MemoryTier.WORKING,
            importance=0.8, agent_id="agent-1"
        )
        stats = self.consolidator.consolidate("agent-1")
        assert stats.agent_id == "agent-1"
        assert stats.entries_examined >= 0
        assert stats.duration_ms >= 0

    def test_consolidation_history_tracked(self):
        self.consolidator.consolidate("agent-2")
        self.consolidator.consolidate("agent-2")
        history = self.consolidator.get_consolidation_history("agent-2")
        assert len(history) == 2

    def test_consolidation_history_filtered_by_agent(self):
        self.consolidator.consolidate("agent-x")
        self.consolidator.consolidate("agent-y")
        history_x = self.consolidator.get_consolidation_history("agent-x")
        assert all(h["agent_id"] == "agent-x" for h in history_x)

    def test_cluster_entries_single_item(self):
        self.store.store(
            "Single fact", self.MemoryType.FACT, self.MemoryTier.EPISODIC,
            importance=0.5, agent_id="agent-3"
        )
        entries = self.store.get_recent(10, tier=self.MemoryTier.EPISODIC, agent_id="agent-3")
        clusters = self.consolidator.cluster_entries(entries)
        assert len(clusters) == len(entries)

    def test_cluster_similar_entries(self):
        for i in range(3):
            self.store.store(
                f"Machine learning neural network deep learning {i}",
                self.MemoryType.FACT,
                self.MemoryTier.EPISODIC,
                importance=0.6,
                agent_id="agent-4",
            )
        entries = self.store.get_recent(10, tier=self.MemoryTier.EPISODIC, agent_id="agent-4")
        clusters = self.consolidator.cluster_entries(entries, threshold=0.5)
        # Some entries should cluster together
        assert len(clusters) <= len(entries)

    def test_empty_store_consolidation(self):
        stats = self.consolidator.consolidate("agent-empty")
        assert stats.entries_examined == 0

    def test_stats_dict_format(self):
        stats = self.consolidator.consolidate("agent-fmt")
        d = stats.to_dict()
        required_keys = ["run_id", "agent_id", "entries_examined", "duration_ms"]
        for key in required_keys:
            assert key in d


# ──────────────────────────────────────────────────────────────────────────────
# Agent Coordination Bus Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestAgentCoordinationBus:
    def setup_method(self):
        from infrastructure.agent.coordination_bus import AgentCoordinationBus
        self.bus = AgentCoordinationBus()

    def test_register_agent(self):
        reg = self.bus.register_agent("agent-1", capabilities={"code_gen"})
        assert reg.agent_id == "agent-1"
        assert "code_gen" in reg.capabilities

    def test_deregister_agent(self):
        self.bus.register_agent("agent-2")
        result = self.bus.deregister_agent("agent-2")
        assert result is True
        assert "agent-2" not in {a["agent_id"] for a in self.bus.list_agents()}

    def test_deregister_nonexistent_returns_false(self):
        assert self.bus.deregister_agent("ghost-agent") is False

    def test_heartbeat(self):
        self.bus.register_agent("agent-3")
        result = self.bus.heartbeat("agent-3")
        assert result is True

    def test_heartbeat_unknown_agent(self):
        assert self.bus.heartbeat("unknown") is False

    def test_find_capable_agents(self):
        self.bus.register_agent("agent-a", capabilities={"code_gen", "search"})
        self.bus.register_agent("agent-b", capabilities={"search"})
        capable = self.bus.find_capable_agents({"code_gen"})
        ids = {a.agent_id for a in capable}
        assert "agent-a" in ids
        assert "agent-b" not in ids

    def test_find_capable_agents_multi_cap(self):
        self.bus.register_agent("agent-c", capabilities={"code_gen", "search", "analysis"})
        capable = self.bus.find_capable_agents({"code_gen", "analysis"})
        assert any(a.agent_id == "agent-c" for a in capable)

    @pytest.mark.asyncio
    async def test_send_and_receive_message(self):
        from infrastructure.agent.coordination_bus import MessageType, MessagePriority
        self.bus.register_agent("sender")
        self.bus.register_agent("receiver")
        msg = self.bus.create_message(
            MessageType.BROADCAST,
            sender_id="sender",
            payload={"text": "hello"},
            recipient_id="receiver",
        )
        delivered = await self.bus.send(msg)
        assert delivered is True
        received = await self.bus.receive("receiver")
        assert received is not None
        assert received.payload["text"] == "hello"

    @pytest.mark.asyncio
    async def test_send_to_unknown_recipient_goes_to_dlq(self):
        from infrastructure.agent.coordination_bus import MessageType
        self.bus.register_agent("sender2")
        msg = self.bus.create_message(
            MessageType.TASK_DELEGATION,
            sender_id="sender2",
            payload={},
            recipient_id="ghost",
        )
        delivered = await self.bus.send(msg)
        assert delivered is False
        assert self.bus.stats()["dead_letter_count"] > 0

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        from infrastructure.agent.coordination_bus import MessageType, MessagePriority
        self.bus.register_agent("prio-sender")
        self.bus.register_agent("prio-receiver")

        low_msg = self.bus.create_message(
            MessageType.BROADCAST, "prio-sender", {"order": "low"},
            recipient_id="prio-receiver", priority=MessagePriority.LOW
        )
        high_msg = self.bus.create_message(
            MessageType.BROADCAST, "prio-sender", {"order": "high"},
            recipient_id="prio-receiver", priority=MessagePriority.CRITICAL
        )
        await self.bus.send(low_msg)
        await self.bus.send(high_msg)

        first = await self.bus.receive("prio-receiver")
        assert first is not None
        assert first.payload["order"] == "high"

    def test_acquire_lock(self):
        lock = self.bus.acquire_lock("resource-1", "agent-lock")
        assert lock is not None
        assert lock.holder_id == "agent-lock"

    def test_acquire_lock_conflict(self):
        self.bus.acquire_lock("resource-2", "agent-x")
        lock = self.bus.acquire_lock("resource-2", "agent-y")
        assert lock is None  # Already locked

    def test_release_lock(self):
        self.bus.acquire_lock("resource-3", "agent-r")
        released = self.bus.release_lock("resource-3", "agent-r")
        assert released is True
        assert not self.bus.is_locked("resource-3")

    def test_release_lock_wrong_holder(self):
        self.bus.acquire_lock("resource-4", "agent-owner")
        released = self.bus.release_lock("resource-4", "agent-thief")
        assert released is False

    def test_expired_lock_not_locked(self):
        import time
        lock = self.bus.acquire_lock("resource-5", "agent-exp", ttl_seconds=0.01)
        assert lock is not None
        time.sleep(0.05)
        assert not self.bus.is_locked("resource-5")

    def test_consensus_proposal(self):
        self.bus.register_agent("voter-1")
        self.bus.register_agent("voter-2")
        proposal = self.bus.propose_consensus(
            "deploy_to_prod",
            {"version": "1.2.3"},
            proposer_id="orchestrator",
            required_agents=["voter-1", "voter-2"],
            quorum_fraction=1.0,
        )
        assert proposal.proposal_id is not None
        r1 = self.bus.cast_vote(proposal.proposal_id, "voter-1", True)
        assert r1 is None  # not decided yet
        r2 = self.bus.cast_vote(proposal.proposal_id, "voter-2", True)
        assert r2 is True  # now approved

    def test_consensus_rejected(self):
        self.bus.register_agent("voter-a")
        self.bus.register_agent("voter-b")
        proposal = self.bus.propose_consensus(
            "risky_operation",
            {},
            proposer_id="orchestrator",
            required_agents=["voter-a", "voter-b"],
            quorum_fraction=0.6,
        )
        self.bus.cast_vote(proposal.proposal_id, "voter-a", False)
        result = self.bus.cast_vote(proposal.proposal_id, "voter-b", False)
        assert result is False

    def test_bus_stats(self):
        stats = self.bus.stats()
        assert "registered_agents" in stats
        assert "messages_sent" in stats
        assert "dead_letter_count" in stats


# ──────────────────────────────────────────────────────────────────────────────
# Workflow DSL Compiler Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestWorkflowDSLCompiler:
    def setup_method(self):
        from infrastructure.workflow.dsl_compiler import WorkflowDSLCompiler
        self.compiler = WorkflowDSLCompiler()

    def _simple_workflow(self):
        return {
            "name": "Simple Workflow",
            "version": "1.0",
            "steps": [
                {"id": "step1", "type": "tool_call", "tool": "my_tool"},
                {"id": "step2", "type": "tool_call", "tool": "other_tool", "depends_on": ["step1"]},
            ],
        }

    def test_compile_simple_workflow(self):
        result = self.compiler.compile_dict(self._simple_workflow())
        assert result.success
        assert result.workflow is not None
        assert result.workflow.name == "Simple Workflow"

    def test_execution_order_respects_deps(self):
        result = self.compiler.compile_dict(self._simple_workflow())
        order = result.workflow.execution_order
        assert order.index("step1") < order.index("step2")

    def test_error_on_missing_steps(self):
        result = self.compiler.compile_dict({"name": "Empty"})
        assert not result.success
        assert any("step" in e.field.lower() for e in result.errors)

    def test_error_on_missing_step_id(self):
        result = self.compiler.compile_dict({
            "name": "Bad",
            "steps": [{"type": "tool_call", "tool": "t"}],
        })
        assert not result.success

    def test_error_on_unknown_step_type(self):
        result = self.compiler.compile_dict({
            "name": "Bad",
            "steps": [{"id": "s1", "type": "nonexistent_type"}],
        })
        assert not result.success

    def test_error_on_unknown_dependency(self):
        result = self.compiler.compile_dict({
            "name": "Bad",
            "steps": [
                {"id": "s1", "type": "tool_call", "tool": "t", "depends_on": ["ghost"]},
            ],
        })
        assert not result.success

    def test_cycle_detection(self):
        result = self.compiler.compile_dict({
            "name": "Cycle",
            "steps": [
                {"id": "a", "type": "tool_call", "tool": "t", "depends_on": ["b"]},
                {"id": "b", "type": "tool_call", "tool": "t", "depends_on": ["a"]},
            ],
        })
        assert not result.success
        assert any("cycle" in e.message.lower() for e in result.errors)

    def test_parallel_group_detection(self):
        result = self.compiler.compile_dict({
            "name": "Parallel",
            "steps": [
                {"id": "a", "type": "tool_call", "tool": "t"},
                {"id": "b", "type": "tool_call", "tool": "t"},
                {"id": "c", "type": "tool_call", "tool": "t", "depends_on": ["a", "b"]},
            ],
        })
        assert result.success
        # a and b can run in parallel
        assert any(len(g) > 1 for g in result.workflow.parallel_groups)

    def test_retry_policy_parsed(self):
        result = self.compiler.compile_dict({
            "name": "Retry",
            "steps": [
                {
                    "id": "s1",
                    "type": "tool_call",
                    "tool": "flaky_tool",
                    "retry": {"max_attempts": 5, "backoff": "exponential"},
                },
            ],
        })
        assert result.success
        step = result.workflow.step_map["s1"]
        assert step.retry is not None
        assert step.retry.max_attempts == 5

    def test_variables_extracted(self):
        result = self.compiler.compile_dict({
            "name": "Variables",
            "variables": {"bucket": "my-bucket", "limit": 100},
            "steps": [{"id": "s1", "type": "tool_call", "tool": "reader"}],
        })
        assert result.success
        assert result.workflow.variables["bucket"] == "my-bucket"

    def test_on_failure_parsed(self):
        result = self.compiler.compile_dict({
            "name": "Failure",
            "steps": [
                {"id": "s1", "type": "tool_call", "tool": "t", "on_failure": "skip"},
            ],
        })
        assert result.success
        from infrastructure.workflow.dsl_compiler import OnFailure
        assert result.workflow.step_map["s1"].on_failure == OnFailure.SKIP

    def test_trigger_parsed(self):
        result = self.compiler.compile_dict({
            "name": "Triggered",
            "steps": [{"id": "s1", "type": "tool_call", "tool": "t"}],
            "triggers": [
                {"type": "schedule", "cron": "0 * * * *"},
                {"type": "webhook", "path": "/my/hook"},
            ],
        })
        assert result.success
        assert len(result.workflow.triggers) == 2

    def test_compile_json(self):
        import json
        definition = self._simple_workflow()
        json_str = json.dumps(definition)
        result = self.compiler.compile_json(json_str)
        assert result.success

    def test_compile_invalid_json(self):
        result = self.compiler.compile_json("{invalid json}")
        assert not result.success
        assert any("JSON" in e.message for e in result.errors)

    def test_compilation_result_to_dict(self):
        result = self.compiler.compile_dict(self._simple_workflow())
        d = result.to_dict()
        assert d["success"] is True
        assert "workflow" in d
        assert "errors" in d

    def test_retry_policy_delay_exponential(self):
        from infrastructure.workflow.dsl_compiler import RetryPolicy, BackoffStrategy
        policy = RetryPolicy(
            max_attempts=5,
            backoff=BackoffStrategy.EXPONENTIAL,
            initial_delay_seconds=1.0,
            max_delay_seconds=60.0,
        )
        assert policy.compute_delay(0) == 1.0
        assert policy.compute_delay(1) == 2.0
        assert policy.compute_delay(2) == 4.0
        assert policy.compute_delay(10) == 60.0  # capped at max

    def test_retry_policy_delay_fixed(self):
        from infrastructure.workflow.dsl_compiler import RetryPolicy, BackoffStrategy
        policy = RetryPolicy(backoff=BackoffStrategy.FIXED, initial_delay_seconds=5.0)
        assert policy.compute_delay(0) == 5.0
        assert policy.compute_delay(5) == 5.0


# ──────────────────────────────────────────────────────────────────────────────
# Onboarding Engine Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestOnboardingEngine:
    def setup_method(self):
        from infrastructure.onboarding.onboarding_engine import (
            OnboardingEngine,
            OnboardingPersona,
            TriggerType,
        )
        self.engine = OnboardingEngine()
        self.OnboardingPersona = OnboardingPersona
        self.TriggerType = TriggerType

    def test_start_onboarding(self):
        state = self.engine.start_onboarding("user-1", "tenant-a", self.OnboardingPersona.DEVELOPER)
        assert state.user_id == "user-1"
        assert state.persona == self.OnboardingPersona.DEVELOPER

    def test_start_onboarding_idempotent(self):
        state1 = self.engine.start_onboarding("user-2", "tenant-a")
        state2 = self.engine.start_onboarding("user-2", "tenant-a")
        assert state1 is state2

    def test_complete_milestone(self):
        self.engine.start_onboarding("user-3", "tenant-a")
        result = self.engine.complete_milestone("user-3", "complete_profile")
        assert result.get("completed") is True
        assert result.get("points_earned", 0) > 0

    def test_complete_milestone_twice_is_idempotent(self):
        self.engine.start_onboarding("user-4", "tenant-a")
        self.engine.complete_milestone("user-4", "complete_profile")
        result = self.engine.complete_milestone("user-4", "complete_profile")
        assert result.get("already_completed") is True

    def test_milestone_prerequisites_enforced(self):
        self.engine.start_onboarding("user-5", "tenant-a")
        # create_api_key requires complete_profile
        result = self.engine.complete_milestone("user-5", "create_api_key")
        assert "error" in result

    def test_sequential_milestones(self):
        self.engine.start_onboarding("user-6", "tenant-a")
        r1 = self.engine.complete_milestone("user-6", "complete_profile")
        assert r1.get("completed") is True
        r2 = self.engine.complete_milestone("user-6", "create_api_key")
        assert r2.get("completed") is True

    def test_skip_optional_milestone(self):
        self.engine.start_onboarding("user-7", "tenant-a")
        result = self.engine.skip_milestone("user-7", "install_plugin")
        assert result is True

    def test_cannot_skip_required_milestone(self):
        self.engine.start_onboarding("user-8", "tenant-a")
        result = self.engine.skip_milestone("user-8", "complete_profile")
        assert result is False  # not optional

    def test_trigger_event_auto_completes(self):
        self.engine.start_onboarding("user-9", "tenant-a")
        self.engine.complete_milestone("user-9", "complete_profile")
        triggered = self.engine.trigger_event(
            "user-9", self.TriggerType.FEATURE_USE, "api_keys"
        )
        assert any(m.milestone_id == "create_api_key" for m in triggered)

    def test_get_checklist(self):
        self.engine.start_onboarding("user-10", "tenant-a")
        checklist = self.engine.get_checklist("user-10")
        assert checklist is not None
        assert checklist.total_count > 0
        assert isinstance(checklist.items, list)

    def test_checklist_shows_locked_items(self):
        self.engine.start_onboarding("user-11", "tenant-a")
        checklist = self.engine.get_checklist("user-11")
        statuses = {item["status"] for item in checklist.items}
        assert "locked" in statuses or "available" in statuses

    def test_checklist_progress_increases(self):
        self.engine.start_onboarding("user-12", "tenant-a")
        c1 = self.engine.get_checklist("user-12")
        self.engine.complete_milestone("user-12", "complete_profile")
        c2 = self.engine.get_checklist("user-12")
        assert c2.completed_count >= c1.completed_count

    def test_spotlights_require_milestone(self):
        self.engine.start_onboarding("user-13", "tenant-a")
        # No milestones completed yet - spotlights requiring milestones shouldn't show
        spotlights = self.engine.get_spotlights_for_user("user-13")
        # All defaults require milestone completion
        assert len(spotlights) == 0

    def test_spotlights_after_milestone(self):
        self.engine.start_onboarding("user-14", "tenant-a")
        self.engine.complete_milestone("user-14", "complete_profile")
        self.engine.complete_milestone("user-14", "create_api_key")
        spotlights = self.engine.get_spotlights_for_user("user-14")
        assert len(spotlights) > 0

    def test_spotlight_show_count_limit(self):
        self.engine.start_onboarding("user-15", "tenant-a")
        self.engine.complete_milestone("user-15", "complete_profile")
        self.engine.complete_milestone("user-15", "create_api_key")
        # Record that we showed the spotlight max times
        spotlight_id = "workflow_builder_intro"
        for _ in range(10):
            self.engine.record_spotlight_shown("user-15", spotlight_id)
        spotlights = self.engine.get_spotlights_for_user("user-15")
        assert not any(s.spotlight_id == spotlight_id for s in spotlights)

    def test_funnel_analytics(self):
        self.engine.start_onboarding("user-16", "tenant-a")
        self.engine.complete_milestone("user-16", "complete_profile")
        analytics = self.engine.get_funnel_analytics()
        assert analytics["total_users"] >= 1
        assert "activation_rate_pct" in analytics

    def test_unknown_user_checklist_returns_none(self):
        checklist = self.engine.get_checklist("ghost-user")
        assert checklist is None

    def test_points_accumulate(self):
        self.engine.start_onboarding("user-17", "tenant-a")
        self.engine.complete_milestone("user-17", "complete_profile")
        self.engine.complete_milestone("user-17", "create_api_key")
        state = self.engine.get_state("user-17")
        assert state.total_points >= 30  # 10 + 20
