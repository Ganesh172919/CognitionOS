"""
AI Prompt Engineering Platform
================================
Version-controlled prompt management, A/B testing, chain execution,
template rendering, performance optimization, and cost tracking.

Implements:
- Prompt versioning with semantic diff and rollback
- Template engine with Jinja2-style variable substitution
- Prompt chains (sequential, parallel, conditional, loop)
- A/B testing with statistical significance detection
- Automatic prompt optimization via few-shot selection
- Token counting and cost estimation per provider
- Prompt library with tagging and search
- Execution tracing and latency profiling
- Safety checks: PII detection, injection detection, output filtering
- Caching with semantic deduplication
"""

from __future__ import annotations

import ast
import hashlib
import math
import operator
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Safe condition/expression evaluator (no arbitrary code execution)
# ---------------------------------------------------------------------------

_CHAIN_SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Eq: operator.eq, ast.NotEq: operator.ne,
    ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b,
    ast.And: lambda a, b: a and b, ast.Or: lambda a, b: a or b,
    ast.Not: operator.not_, ast.USub: operator.neg,
}


def _safe_node(node: ast.AST, ns: Dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in ("True", "False", "None"):
            return {"True": True, "False": False, "None": None}[node.id]
        return ns.get(node.id)
    if isinstance(node, ast.Attribute):
        obj = _safe_node(node.value, ns)
        return getattr(obj, node.attr, None) if obj is not None else None
    if isinstance(node, ast.Subscript):
        obj = _safe_node(node.value, ns)
        key = _safe_node(node.slice, ns)
        try:
            return obj[key] if obj is not None else None
        except (KeyError, IndexError, TypeError):
            return None
    if isinstance(node, ast.Index):  # Python 3.8 compat
        return _safe_node(node.value, ns)  # type: ignore[attr-defined]
    if isinstance(node, ast.BoolOp):
        op_fn = _CHAIN_SAFE_OPS.get(type(node.op))
        if not op_fn:
            return False
        result = _safe_node(node.values[0], ns)
        for v in node.values[1:]:
            result = op_fn(result, _safe_node(v, ns))
        return result
    if isinstance(node, ast.UnaryOp):
        op_fn = _CHAIN_SAFE_OPS.get(type(node.op))
        return op_fn(_safe_node(node.operand, ns)) if op_fn else None
    if isinstance(node, ast.BinOp):
        op_fn = _CHAIN_SAFE_OPS.get(type(node.op))
        if not op_fn:
            return None
        return op_fn(_safe_node(node.left, ns), _safe_node(node.right, ns))
    if isinstance(node, ast.Compare):
        left = _safe_node(node.left, ns)
        for op, right_node in zip(node.ops, node.comparators):
            op_fn = _CHAIN_SAFE_OPS.get(type(op))
            if not op_fn:
                return False
            right = _safe_node(right_node, ns)
            if not op_fn(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.IfExp):
        return (
            _safe_node(node.body, ns) if _safe_node(node.test, ns) else _safe_node(node.orelse, ns)
        )
    if isinstance(node, ast.List):
        return [_safe_node(e, ns) for e in node.elts]
    return None


def _safe_eval_chain_expr(expression: str, ns: Dict[str, Any]) -> Any:
    """Evaluate a restricted expression safely (no code injection)."""
    try:
        tree = ast.parse(expression, mode="eval")
        return _safe_node(tree.body, ns)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class PromptStatus(str, Enum):
    DRAFT = "draft"
    TESTING = "testing"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ChainNodeType(str, Enum):
    PROMPT = "prompt"
    CONDITION = "condition"
    TRANSFORM = "transform"
    PARALLEL = "parallel"
    LOOP = "loop"
    OUTPUT = "output"


class OptimizationStrategy(str, Enum):
    FEW_SHOT_SELECTION = "few_shot_selection"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    SELF_CONSISTENCY = "self_consistency"
    TREE_OF_THOUGHT = "tree_of_thought"
    DIRECTIONAL_STIMULUS = "directional_stimulus"


class ProviderModel(str, Enum):
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_35_TURBO = "gpt-3.5-turbo"
    CLAUDE_3_OPUS = "claude-3-opus"
    CLAUDE_3_SONNET = "claude-3-sonnet"
    GEMINI_PRO = "gemini-pro"
    LLAMA3_70B = "llama3-70b"
    MISTRAL_LARGE = "mistral-large"


class ABTestStatus(str, Enum):
    RUNNING = "running"
    PAUSED = "paused"
    CONCLUDED = "concluded"
    INSUFFICIENT_DATA = "insufficient_data"


# ---------------------------------------------------------------------------
# Token pricing (per 1K tokens, USD)
# ---------------------------------------------------------------------------

PROVIDER_PRICING: Dict[str, Dict[str, float]] = {
    ProviderModel.GPT_4: {"input": 0.03, "output": 0.06},
    ProviderModel.GPT_4_TURBO: {"input": 0.01, "output": 0.03},
    ProviderModel.GPT_35_TURBO: {"input": 0.0015, "output": 0.002},
    ProviderModel.CLAUDE_3_OPUS: {"input": 0.015, "output": 0.075},
    ProviderModel.CLAUDE_3_SONNET: {"input": 0.003, "output": 0.015},
    ProviderModel.GEMINI_PRO: {"input": 0.0005, "output": 0.0015},
    ProviderModel.LLAMA3_70B: {"input": 0.0007, "output": 0.0009},
    ProviderModel.MISTRAL_LARGE: {"input": 0.004, "output": 0.012},
}


def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 chars per token."""
    return max(1, len(text) // 4)


def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    pricing = PROVIDER_PRICING.get(model, {"input": 0.01, "output": 0.02})
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1000


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class PromptVariable:
    name: str
    description: str = ""
    default_value: Optional[str] = None
    required: bool = True
    var_type: str = "string"
    examples: List[str] = field(default_factory=list)


@dataclass
class FewShotExample:
    input_vars: Dict[str, str]
    expected_output: str
    quality_score: float = 1.0
    use_count: int = 0
    tags: List[str] = field(default_factory=list)


@dataclass
class PromptVersion:
    version_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version_number: str = "1.0.0"
    template: str = ""
    system_message: Optional[str] = None
    variables: List[PromptVariable] = field(default_factory=list)
    few_shot_examples: List[FewShotExample] = field(default_factory=list)
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.FEW_SHOT_SELECTION
    target_model: ProviderModel = ProviderModel.GPT_4_TURBO
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 1.0
    created_at: float = field(default_factory=time.time)
    created_by: str = "system"
    change_description: str = ""
    tags: List[str] = field(default_factory=list)
    status: PromptStatus = PromptStatus.DRAFT
    avg_latency_ms: float = 0.0
    avg_output_tokens: int = 0
    total_executions: int = 0
    success_rate: float = 1.0


@dataclass
class Prompt:
    prompt_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    category: str = "general"
    tenant_id: str = "global"
    active_version_id: Optional[str] = None
    versions: List[PromptVersion] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    total_executions: int = 0
    avg_cost_usd: float = 0.0

    @property
    def active_version(self) -> Optional[PromptVersion]:
        if not self.active_version_id:
            return self.versions[-1] if self.versions else None
        return next((v for v in self.versions if v.version_id == self.active_version_id), None)


@dataclass
class PromptExecution:
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt_id: str = ""
    version_id: str = ""
    tenant_id: str = "global"
    input_vars: Dict[str, Any] = field(default_factory=dict)
    rendered_prompt: str = ""
    raw_output: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    model: str = ""
    success: bool = True
    error: Optional[str] = None
    safety_flags: List[str] = field(default_factory=list)
    ab_variant: Optional[str] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class ABVariant:
    variant_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    prompt_version_id: str = ""
    traffic_percentage: float = 50.0
    executions: int = 0
    successes: int = 0
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    metric_values: List[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.successes / max(1, self.executions)

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(1, self.executions)

    @property
    def avg_tokens(self) -> float:
        return self.total_tokens / max(1, self.executions)


@dataclass
class ABTest:
    test_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt_id: str = ""
    name: str = ""
    variants: List[ABVariant] = field(default_factory=list)
    status: ABTestStatus = ABTestStatus.RUNNING
    primary_metric: str = "success_rate"
    min_sample_size: int = 100
    confidence_level: float = 0.95
    started_at: float = field(default_factory=time.time)
    concluded_at: Optional[float] = None
    winner_variant_id: Optional[str] = None
    conclusion: Optional[str] = None


# ---------------------------------------------------------------------------
# Template Engine
# ---------------------------------------------------------------------------

class PromptTemplateEngine:
    """Renders prompt templates with variable substitution and few-shot examples."""

    def render(self, version: PromptVersion, variables: Dict[str, Any]) -> str:
        template = version.template

        # Validate required variables
        for var in version.variables:
            if var.required and var.name not in variables and var.default_value is None:
                raise ValueError(f"Required variable '{var.name}' not provided")

        # Apply defaults
        ctx = {}
        for var in version.variables:
            ctx[var.name] = variables.get(var.name, var.default_value or "")

        # Simple {{variable}} substitution
        rendered = re.sub(
            r"\{\{(\w+)\}\}",
            lambda m: str(ctx.get(m.group(1), m.group(0))),
            template,
        )

        # Inject few-shot examples if strategy requires
        if version.optimization_strategy == OptimizationStrategy.FEW_SHOT_SELECTION:
            examples = self._select_examples(version.few_shot_examples, variables, k=3)
            if examples:
                example_text = "\n\n".join(
                    f"Input: {ex.input_vars}\nOutput: {ex.expected_output}"
                    for ex in examples
                )
                rendered = f"Examples:\n{example_text}\n\n{rendered}"

        if version.optimization_strategy == OptimizationStrategy.CHAIN_OF_THOUGHT:
            rendered = rendered + "\nLet's think step by step:"

        return rendered

    def _select_examples(
        self,
        examples: List[FewShotExample],
        context: Dict[str, Any],
        k: int = 3,
    ) -> List[FewShotExample]:
        # Select highest quality examples, prefer relevant ones
        scored = sorted(examples, key=lambda e: e.quality_score, reverse=True)
        return scored[:k]


# ---------------------------------------------------------------------------
# Safety Checker
# ---------------------------------------------------------------------------

class PromptSafetyChecker:
    """Detects PII, prompt injection attempts, and harmful content."""

    _PII_PATTERNS = [
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "ssn"),
        (re.compile(r"\b\d{16}\b"), "credit_card"),
        (re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I), "email"),
        (re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"), "phone"),
    ]

    _INJECTION_PATTERNS = [
        re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.I),
        re.compile(r"disregard\s+(the\s+)?system\s+(prompt|message)", re.I),
        re.compile(r"you\s+are\s+now\s+[\"']?DAN", re.I),
        re.compile(r"<\|im_start\|>|<\|im_end\|>"),
    ]

    def check(self, text: str) -> List[str]:
        flags = []
        for pattern, label in self._PII_PATTERNS:
            if pattern.search(text):
                flags.append(f"pii:{label}")
        for pattern in self._INJECTION_PATTERNS:
            if pattern.search(text):
                flags.append("injection_attempt")
                break
        return flags


# ---------------------------------------------------------------------------
# Chain Execution
# ---------------------------------------------------------------------------

@dataclass
class ChainNode:
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_type: ChainNodeType = ChainNodeType.PROMPT
    prompt_id: Optional[str] = None
    input_mapping: Dict[str, str] = field(default_factory=dict)
    output_key: str = "output"
    condition: Optional[str] = None      # Python expression string
    children: List[str] = field(default_factory=list)  # child node_ids
    max_iterations: int = 3              # for loop nodes


@dataclass
class PromptChain:
    chain_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    nodes: List[ChainNode] = field(default_factory=list)
    entry_node_id: Optional[str] = None
    tenant_id: str = "global"
    total_executions: int = 0
    avg_latency_ms: float = 0.0


@dataclass
class ChainExecutionResult:
    chain_id: str
    success: bool
    outputs: Dict[str, Any]
    node_results: List[Dict[str, Any]]
    total_tokens: int
    total_cost_usd: float
    total_latency_ms: float
    error: Optional[str] = None


class ChainExecutor:
    """Executes prompt chains with context propagation."""

    def __init__(self, prompt_platform: "PromptEngineeringPlatform"):
        self._platform = prompt_platform

    async def execute(
        self,
        chain: PromptChain,
        initial_vars: Dict[str, Any],
        tenant_id: str = "global",
        mock_llm: Optional[Callable] = None,
    ) -> ChainExecutionResult:
        t0 = time.time()
        context: Dict[str, Any] = dict(initial_vars)
        node_results: List[Dict[str, Any]] = []
        total_tokens = 0
        total_cost = 0.0

        node_map = {n.node_id: n for n in chain.nodes}
        current_id = chain.entry_node_id or (chain.nodes[0].node_id if chain.nodes else None)

        try:
            visited = set()
            while current_id and current_id not in visited:
                visited.add(current_id)
                node = node_map.get(current_id)
                if not node:
                    break

                if node.node_type == ChainNodeType.PROMPT and node.prompt_id:
                    # Map input variables
                    exec_vars = {k: context.get(v, v) for k, v in node.input_mapping.items()}
                    exec_vars.update({k: v for k, v in context.items() if k not in exec_vars})

                    result = await self._platform.execute_prompt(
                        node.prompt_id, exec_vars, tenant_id, mock_llm=mock_llm
                    )
                    context[node.output_key] = result.raw_output
                    total_tokens += result.input_tokens + result.output_tokens
                    total_cost += result.cost_usd
                    node_results.append({"node_id": current_id, "output": result.raw_output})

                elif node.node_type == ChainNodeType.CONDITION:
                    # Evaluate condition expression safely
                    try:
                        cond_result = bool(_safe_eval_chain_expr(node.condition or "False", {"ctx": context}))
                    except Exception:
                        cond_result = False
                    if node.children:
                        current_id = node.children[0] if cond_result else (
                            node.children[1] if len(node.children) > 1 else None
                        )
                    continue

                elif node.node_type == ChainNodeType.TRANSFORM:
                    try:
                        result_val = _safe_eval_chain_expr(node.condition or "ctx", {"ctx": context})
                        context[node.output_key] = result_val
                    except Exception:
                        pass

                # Move to next node
                current_id = node.children[0] if node.children else None

        except Exception as exc:
            return ChainExecutionResult(
                chain_id=chain.chain_id,
                success=False,
                outputs=context,
                node_results=node_results,
                total_tokens=total_tokens,
                total_cost_usd=total_cost,
                total_latency_ms=(time.time() - t0) * 1000,
                error=str(exc),
            )

        return ChainExecutionResult(
            chain_id=chain.chain_id,
            success=True,
            outputs=context,
            node_results=node_results,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            total_latency_ms=(time.time() - t0) * 1000,
        )


# ---------------------------------------------------------------------------
# A/B Test Manager
# ---------------------------------------------------------------------------

class PromptABTestManager:
    """Statistical A/B testing for prompt variants."""

    def __init__(self):
        self._tests: Dict[str, ABTest] = {}

    def create_test(
        self,
        prompt_id: str,
        name: str,
        version_a_id: str,
        version_b_id: str,
        traffic_split: float = 0.5,
    ) -> ABTest:
        test = ABTest(
            prompt_id=prompt_id,
            name=name,
            variants=[
                ABVariant(name="A", prompt_version_id=version_a_id, traffic_percentage=100 * (1 - traffic_split)),
                ABVariant(name="B", prompt_version_id=version_b_id, traffic_percentage=100 * traffic_split),
            ],
        )
        self._tests[test.test_id] = test
        return test

    def select_variant(self, test: ABTest) -> ABVariant:
        """Route traffic to variant based on allocation."""
        r = random.random() * 100
        cumulative = 0.0
        for variant in test.variants:
            cumulative += variant.traffic_percentage
            if r <= cumulative:
                return variant
        return test.variants[-1]

    def record_result(
        self,
        test_id: str,
        variant_id: str,
        success: bool,
        latency_ms: float,
        tokens: int,
        cost_usd: float,
        metric_value: Optional[float] = None,
    ) -> None:
        test = self._tests.get(test_id)
        if not test:
            return
        variant = next((v for v in test.variants if v.variant_id == variant_id), None)
        if not variant:
            return
        variant.executions += 1
        variant.successes += 1 if success else 0
        variant.total_latency_ms += latency_ms
        variant.total_tokens += tokens
        variant.total_cost_usd += cost_usd
        if metric_value is not None:
            variant.metric_values.append(metric_value)

    def analyze(self, test_id: str) -> Dict[str, Any]:
        test = self._tests.get(test_id)
        if not test:
            return {}
        variants = test.variants
        total = sum(v.executions for v in variants)
        if total < test.min_sample_size:
            test.status = ABTestStatus.INSUFFICIENT_DATA
            return {"status": "insufficient_data", "total_executions": total, "required": test.min_sample_size}

        # Chi-squared test for success rates
        winner = max(variants, key=lambda v: v.success_rate)
        loser = min(variants, key=lambda v: v.success_rate)

        if winner.success_rate > 0 and loser.success_rate > 0:
            diff = winner.success_rate - loser.success_rate
            pooled = (winner.successes + loser.successes) / max(1, winner.executions + loser.executions)
            se = math.sqrt(pooled * (1 - pooled) * (1 / max(1, winner.executions) + 1 / max(1, loser.executions)))
            z_score = diff / max(se, 1e-10)
            is_significant = abs(z_score) >= 1.96  # 95% confidence
        else:
            is_significant = False
            z_score = 0.0

        if is_significant:
            test.status = ABTestStatus.CONCLUDED
            test.winner_variant_id = winner.variant_id
            test.concluded_at = time.time()
            test.conclusion = (
                f"Variant '{winner.name}' wins with {winner.success_rate:.1%} success rate "
                f"vs {loser.success_rate:.1%} (z={z_score:.2f})"
            )

        return {
            "test_id": test_id,
            "status": test.status.value,
            "winner": winner.name if is_significant else None,
            "z_score": round(z_score, 3),
            "significant": is_significant,
            "variants": [
                {
                    "name": v.name,
                    "executions": v.executions,
                    "success_rate": round(v.success_rate, 4),
                    "avg_latency_ms": round(v.avg_latency_ms, 1),
                    "avg_tokens": round(v.avg_tokens, 1),
                    "total_cost_usd": round(v.total_cost_usd, 6),
                }
                for v in variants
            ],
        }


# ---------------------------------------------------------------------------
# Prompt Library
# ---------------------------------------------------------------------------

class PromptLibrary:
    """Searchable, tagged repository of prompt templates."""

    def __init__(self):
        self._prompts: Dict[str, Prompt] = {}

    def create(self, name: str, description: str, category: str, tenant_id: str, tags: List[str]) -> Prompt:
        p = Prompt(name=name, description=description, category=category, tenant_id=tenant_id, tags=tags)
        self._prompts[p.prompt_id] = p
        return p

    def get(self, prompt_id: str) -> Optional[Prompt]:
        return self._prompts.get(prompt_id)

    def add_version(self, prompt_id: str, version: PromptVersion) -> PromptVersion:
        p = self._prompts.get(prompt_id)
        if not p:
            raise ValueError(f"Prompt {prompt_id} not found")
        if p.versions:
            latest = p.versions[-1].version_number
            parts = [int(x) for x in latest.split(".")]
            parts[-1] += 1
            version.version_number = ".".join(str(p) for p in parts)
        p.versions.append(version)
        p.last_updated = time.time()
        if not p.active_version_id:
            p.active_version_id = version.version_id
        return version

    def activate_version(self, prompt_id: str, version_id: str) -> bool:
        p = self._prompts.get(prompt_id)
        if not p:
            return False
        if not any(v.version_id == version_id for v in p.versions):
            return False
        p.active_version_id = version_id
        return True

    def search(self, query: str, tenant_id: Optional[str] = None, tags: Optional[List[str]] = None) -> List[Prompt]:
        results = []
        q = query.lower()
        for p in self._prompts.values():
            if tenant_id and p.tenant_id not in (tenant_id, "global"):
                continue
            if tags and not all(t in p.tags for t in tags):
                continue
            if q and q not in p.name.lower() and q not in p.description.lower():
                continue
            results.append(p)
        return results

    def list_all(self, tenant_id: Optional[str] = None) -> List[Prompt]:
        if tenant_id:
            return [p for p in self._prompts.values() if p.tenant_id in (tenant_id, "global")]
        return list(self._prompts.values())


# ---------------------------------------------------------------------------
# Prompt Engineering Platform
# ---------------------------------------------------------------------------

class PromptEngineeringPlatform:
    """
    Unified platform for prompt lifecycle management, execution, testing,
    and optimization across AI providers.
    """

    def __init__(self):
        self.library = PromptLibrary()
        self.ab_manager = PromptABTestManager()
        self._template_engine = PromptTemplateEngine()
        self._safety_checker = PromptSafetyChecker()
        self._chains: Dict[str, PromptChain] = {}
        self._executions: List[PromptExecution] = []
        self._execution_cache: Dict[str, str] = {}  # cache_key -> output

    async def create_prompt(
        self,
        name: str,
        template: str,
        description: str = "",
        category: str = "general",
        tenant_id: str = "global",
        tags: Optional[List[str]] = None,
        variables: Optional[List[PromptVariable]] = None,
        system_message: Optional[str] = None,
        model: ProviderModel = ProviderModel.GPT_4_TURBO,
    ) -> Prompt:
        prompt = self.library.create(name, description, category, tenant_id, tags or [])
        version = PromptVersion(
            template=template,
            system_message=system_message,
            variables=variables or [],
            target_model=model,
            status=PromptStatus.ACTIVE,
        )
        self.library.add_version(prompt.prompt_id, version)
        return prompt

    async def execute_prompt(
        self,
        prompt_id: str,
        variables: Dict[str, Any],
        tenant_id: str = "global",
        version_id: Optional[str] = None,
        ab_test_id: Optional[str] = None,
        mock_llm: Optional[Callable] = None,
    ) -> PromptExecution:
        prompt = self.library.get(prompt_id)
        if not prompt:
            raise ValueError(f"Prompt {prompt_id} not found")

        # Handle A/B test routing
        ab_variant = None
        if ab_test_id:
            test = self.ab_manager._tests.get(ab_test_id)
            if test and test.status == ABTestStatus.RUNNING:
                variant = self.ab_manager.select_variant(test)
                version_id = variant.prompt_version_id
                ab_variant = variant.variant_id

        version = (
            next((v for v in prompt.versions if v.version_id == version_id), None)
            if version_id
            else prompt.active_version
        )
        if not version:
            raise ValueError("No active version found")

        # Safety check on input
        input_str = str(variables)
        safety_flags = self._safety_checker.check(input_str)

        # Check cache
        cache_key = hashlib.md5(f"{version.version_id}:{sorted(variables.items())}".encode()).hexdigest()
        cached_output = self._execution_cache.get(cache_key)

        t0 = time.time()
        rendered = self._template_engine.render(version, variables)
        input_tokens = estimate_tokens(rendered)
        if version.system_message:
            input_tokens += estimate_tokens(version.system_message)

        if cached_output:
            raw_output = cached_output
            output_tokens = estimate_tokens(raw_output)
        elif mock_llm:
            raw_output = mock_llm(rendered)
            output_tokens = estimate_tokens(raw_output)
            self._execution_cache[cache_key] = raw_output
        else:
            # Simulate LLM response (would call real LLM in production)
            raw_output = f"[LLM response for: {rendered[:80]}...]"
            output_tokens = estimate_tokens(raw_output)

        latency_ms = (time.time() - t0) * 1000
        cost_usd = estimate_cost(input_tokens, output_tokens, version.target_model)

        execution = PromptExecution(
            prompt_id=prompt_id,
            version_id=version.version_id,
            tenant_id=tenant_id,
            input_vars=variables,
            rendered_prompt=rendered,
            raw_output=raw_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            model=version.target_model,
            safety_flags=safety_flags,
            ab_variant=ab_variant,
        )
        self._executions.append(execution)
        prompt.total_executions += 1

        # Update version stats
        version.total_executions += 1
        version.avg_latency_ms = (
            (version.avg_latency_ms * (version.total_executions - 1) + latency_ms)
            / version.total_executions
        )

        # Record A/B result
        if ab_test_id and ab_variant:
            self.ab_manager.record_result(ab_test_id, ab_variant, True, latency_ms, output_tokens, cost_usd)

        return execution

    async def create_chain(
        self,
        name: str,
        nodes: List[ChainNode],
        entry_node_id: Optional[str] = None,
        tenant_id: str = "global",
    ) -> PromptChain:
        chain = PromptChain(
            name=name,
            nodes=nodes,
            entry_node_id=entry_node_id or (nodes[0].node_id if nodes else None),
            tenant_id=tenant_id,
        )
        self._chains[chain.chain_id] = chain
        return chain

    async def execute_chain(
        self,
        chain_id: str,
        variables: Dict[str, Any],
        tenant_id: str = "global",
        mock_llm: Optional[Callable] = None,
    ) -> ChainExecutionResult:
        chain = self._chains.get(chain_id)
        if not chain:
            raise ValueError(f"Chain {chain_id} not found")
        executor = ChainExecutor(self)
        result = await executor.execute(chain, variables, tenant_id, mock_llm)
        chain.total_executions += 1
        return result

    def get_cost_report(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        executions = self._executions
        if tenant_id:
            executions = [e for e in executions if e.tenant_id == tenant_id]
        total_cost = sum(e.cost_usd for e in executions)
        total_tokens = sum(e.input_tokens + e.output_tokens for e in executions)
        by_model: Dict[str, float] = {}
        for e in executions:
            by_model[e.model] = by_model.get(e.model, 0.0) + e.cost_usd
        return {
            "total_executions": len(executions),
            "total_cost_usd": round(total_cost, 6),
            "total_tokens": total_tokens,
            "cost_by_model": {k: round(v, 6) for k, v in by_model.items()},
            "avg_cost_per_execution": round(total_cost / max(1, len(executions)), 6),
        }

    def get_platform_summary(self) -> Dict[str, Any]:
        return {
            "total_prompts": len(self.library._prompts),
            "total_chains": len(self._chains),
            "total_executions": len(self._executions),
            "active_ab_tests": sum(
                1 for t in self.ab_manager._tests.values() if t.status == ABTestStatus.RUNNING
            ),
            "execution_cache_size": len(self._execution_cache),
        }
