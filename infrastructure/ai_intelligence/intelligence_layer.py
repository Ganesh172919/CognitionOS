"""
AI Intelligence Layer — CognitionOS

Enhances the single-agent runtime with:
- Prompt optimization and caching
- Context window management
- Token budget tracking
- Response quality scoring
- Hallucination confidence analysis
- Model routing (cost vs quality)
- Conversation summarization
- Chain-of-thought scaffolding
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    ECONOMY = "economy"     # fast, cheap (flash, haiku)
    STANDARD = "standard"   # balanced (gpt-4o-mini, sonnet)
    PREMIUM = "premium"     # best quality (gpt-4o, opus)


class TaskComplexity(str, Enum):
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


@dataclass
class ModelConfig:
    model_id: str
    provider: str
    tier: ModelTier
    max_context: int
    cost_per_1k_input: float
    cost_per_1k_output: float
    avg_latency_ms: float = 500.0
    quality_score: float = 0.8

    @property
    def cost_score(self) -> float:
        return 1.0 - min(1.0, (self.cost_per_1k_input + self.cost_per_1k_output) / 0.1)


@dataclass
class TokenBudget:
    max_input_tokens: int = 8000
    max_output_tokens: int = 4000
    max_total_tokens: int = 12000
    used_input: int = 0
    used_output: int = 0
    cost_usd: float = 0.0
    max_cost_usd: float = 10.0

    @property
    def remaining_input(self) -> int:
        return max(0, self.max_input_tokens - self.used_input)

    @property
    def remaining_output(self) -> int:
        return max(0, self.max_output_tokens - self.used_output)

    @property
    def budget_used_pct(self) -> float:
        return (self.cost_usd / self.max_cost_usd * 100) if self.max_cost_usd > 0 else 0

    def can_afford(self, estimated_tokens: int, cost_per_1k: float) -> bool:
        est_cost = (estimated_tokens / 1000) * cost_per_1k
        return (self.cost_usd + est_cost) <= self.max_cost_usd

    def record_usage(self, input_tokens: int, output_tokens: int, cost: float) -> None:
        self.used_input += input_tokens
        self.used_output += output_tokens
        self.cost_usd += cost


@dataclass
class QualityScore:
    relevance: float = 0.0  # 0-1
    coherence: float = 0.0
    completeness: float = 0.0
    accuracy_confidence: float = 0.0
    code_validity: float = 0.0
    overall: float = 0.0

    def compute_overall(self) -> float:
        weights = {"relevance": 0.25, "coherence": 0.2, "completeness": 0.25,
                    "accuracy_confidence": 0.15, "code_validity": 0.15}
        self.overall = sum(getattr(self, k) * w for k, w in weights.items())
        return self.overall


@dataclass
class PromptTemplate:
    name: str
    system_prompt: str
    user_template: str
    variables: List[str] = field(default_factory=list)
    model_tier: ModelTier = ModelTier.STANDARD
    max_tokens: int = 2000
    temperature: float = 0.3


# Default model catalog
MODEL_CATALOG: Dict[str, ModelConfig] = {
    "gpt-4o-mini": ModelConfig("gpt-4o-mini", "openai", ModelTier.ECONOMY, 128000, 0.00015, 0.0006, 300, 0.75),
    "gpt-4o": ModelConfig("gpt-4o", "openai", ModelTier.PREMIUM, 128000, 0.005, 0.015, 800, 0.95),
    "claude-3-haiku": ModelConfig("claude-3-haiku", "anthropic", ModelTier.ECONOMY, 200000, 0.00025, 0.00125, 250, 0.72),
    "claude-3-sonnet": ModelConfig("claude-3-sonnet", "anthropic", ModelTier.STANDARD, 200000, 0.003, 0.015, 600, 0.88),
    "claude-3-opus": ModelConfig("claude-3-opus", "anthropic", ModelTier.PREMIUM, 200000, 0.015, 0.075, 1200, 0.96),
    "gemini-pro": ModelConfig("gemini-pro", "google", ModelTier.STANDARD, 1000000, 0.00125, 0.005, 400, 0.85),
    "gemini-flash": ModelConfig("gemini-flash", "google", ModelTier.ECONOMY, 1000000, 0.000075, 0.0003, 200, 0.70),
}


class AIIntelligenceLayer:
    """Orchestrates AI model selection, prompt optimization, and quality control."""

    def __init__(self, *, models: Dict[str, ModelConfig] | None = None) -> None:
        self._models = models or dict(MODEL_CATALOG)
        self._prompt_cache: Dict[str, str] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._usage_history: List[Dict[str, Any]] = []
        self._quality_history: Dict[str, List[float]] = defaultdict(list)
        self._templates: Dict[str, PromptTemplate] = {}

    # ---- model routing ----
    def select_model(self, *, complexity: TaskComplexity = TaskComplexity.MODERATE,
                     budget: TokenBudget | None = None,
                     prefer_quality: bool = False,
                     prefer_speed: bool = False,
                     required_context: int = 0) -> ModelConfig:
        candidates = list(self._models.values())

        # Filter by context window
        if required_context > 0:
            candidates = [m for m in candidates if m.max_context >= required_context]

        if not candidates:
            candidates = list(self._models.values())

        # Score each model
        scored: List[Tuple[float, ModelConfig]] = []
        for m in candidates:
            score = 0.0
            if complexity in (TaskComplexity.TRIVIAL, TaskComplexity.SIMPLE):
                score += m.cost_score * 0.5 + (1 - m.avg_latency_ms / 2000) * 0.3
            elif complexity == TaskComplexity.MODERATE:
                score += m.quality_score * 0.4 + m.cost_score * 0.3
            else:
                score += m.quality_score * 0.6 + (1 - m.avg_latency_ms / 2000) * 0.1

            if prefer_quality:
                score = m.quality_score * 0.8 + score * 0.2
            if prefer_speed:
                speed_score = 1 - min(1, m.avg_latency_ms / 1500)
                score = speed_score * 0.7 + score * 0.3

            if budget and not budget.can_afford(1000, m.cost_per_1k_output):
                score *= 0.1

            scored.append((score, m))

        scored.sort(key=lambda x: -x[0])
        return scored[0][1]

    def classify_complexity(self, requirement: str) -> TaskComplexity:
        length = len(requirement.split())
        keywords_complex = {"architecture", "system", "integrate", "optimize", "refactor",
                            "multi", "distributed", "scale", "enterprise", "migration"}
        keywords_simple = {"fix", "add", "change", "update", "rename", "remove", "delete"}

        complex_count = sum(1 for w in requirement.lower().split() if w in keywords_complex)
        simple_count = sum(1 for w in requirement.lower().split() if w in keywords_simple)

        if length < 10 and simple_count > 0:
            return TaskComplexity.TRIVIAL
        if length < 30 and complex_count == 0:
            return TaskComplexity.SIMPLE
        if complex_count >= 3 or length > 100:
            return TaskComplexity.EXPERT
        if complex_count >= 1:
            return TaskComplexity.COMPLEX
        return TaskComplexity.MODERATE

    # ---- prompt optimization ----
    def optimize_prompt(self, system: str, user: str, *,
                        max_tokens: int = 8000) -> Tuple[str, str]:
        # Remove redundant whitespace
        system = re.sub(r'\s+', ' ', system).strip()
        user = re.sub(r'\n{3,}', '\n\n', user).strip()

        est_tokens = (len(system) + len(user)) // 4
        if est_tokens > max_tokens:
            ratio = max_tokens / est_tokens
            user = user[:int(len(user) * ratio)]
        return system, user

    def get_cached_response(self, prompt_hash: str) -> str | None:
        result = self._prompt_cache.get(prompt_hash)
        if result:
            self._cache_hits += 1
        else:
            self._cache_misses += 1
        return result

    def cache_response(self, prompt_hash: str, response: str) -> None:
        self._prompt_cache[prompt_hash] = response
        if len(self._prompt_cache) > 5000:
            keys = list(self._prompt_cache.keys())
            for k in keys[:1000]:
                del self._prompt_cache[k]

    def hash_prompt(self, system: str, user: str) -> str:
        raw = json.dumps({"s": system, "u": user}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # ---- quality scoring ----
    def score_response(self, response: str, *, requirement: str = "",
                       expected_format: str = "") -> QualityScore:
        score = QualityScore()

        # Relevance
        if requirement:
            req_words = set(requirement.lower().split())
            resp_words = set(response.lower().split())
            overlap = len(req_words & resp_words)
            score.relevance = min(1.0, overlap / max(1, len(req_words)) * 2)

        # Coherence
        sentences = response.split('.')
        score.coherence = min(1.0, len(sentences) / max(1, len(response) / 100))

        # Completeness
        score.completeness = min(1.0, len(response) / 500) if response else 0

        # Code validity (if code expected)
        if expected_format == "python" or "```python" in response:
            code_blocks = re.findall(r'```(?:python)?\n(.*?)```', response, re.DOTALL)
            if code_blocks:
                valid = 0
                for block in code_blocks:
                    try:
                        compile(block, "<string>", "exec")
                        valid += 1
                    except SyntaxError:
                        pass
                score.code_validity = valid / len(code_blocks)
            else:
                score.code_validity = 0.5

        # Accuracy confidence (heuristic)
        hedging = sum(1 for w in ["maybe", "possibly", "might", "i think", "not sure", "unclear"]
                      if w in response.lower())
        score.accuracy_confidence = max(0, 1.0 - hedging * 0.15)

        score.compute_overall()
        return score

    # ---- chain-of-thought scaffolding ----
    def build_cot_prompt(self, requirement: str, *, steps: int = 5) -> str:
        return (
            f"You are an expert software engineer. Solve this step-by-step.\n\n"
            f"Requirement: {requirement}\n\n"
            f"Think through this in {steps} structured steps:\n"
            + "".join(f"Step {i+1}: [describe your reasoning]\n" for i in range(steps))
            + "\nFinal Answer: [your complete solution]"
        )

    # ---- context management ----
    def truncate_context(self, messages: List[Dict[str, str]], *,
                         max_tokens: int = 8000) -> List[Dict[str, str]]:
        total = sum(len(m.get("content", "")) // 4 for m in messages)
        if total <= max_tokens:
            return messages

        # Keep system + first user + last N messages
        system = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        if len(non_system) <= 2:
            return messages

        budget = max_tokens - sum(len(m["content"]) // 4 for m in system)
        result = list(system)
        kept = []
        for m in reversed(non_system):
            mtokens = len(m.get("content", "")) // 4
            if budget >= mtokens:
                kept.insert(0, m)
                budget -= mtokens
            else:
                break
        return result + kept

    def summarize_conversation(self, messages: List[Dict[str, str]]) -> str:
        parts = []
        for m in messages[-10:]:
            role = m.get("role", "unknown")
            content = m.get("content", "")[:200]
            parts.append(f"[{role}]: {content}")
        return "\n".join(parts)

    # ---- template management ----
    def register_template(self, template: PromptTemplate) -> None:
        self._templates[template.name] = template

    def render_template(self, name: str, **variables: str) -> Tuple[str, str]:
        t = self._templates.get(name)
        if not t:
            raise ValueError(f"Template not found: {name}")
        user = t.user_template
        for var in t.variables:
            user = user.replace(f"{{{{{var}}}}}", variables.get(var, f"[{var}]"))
        return t.system_prompt, user

    # ---- usage tracking ----
    def record_usage(self, model: str, input_tokens: int, output_tokens: int,
                     cost: float, *, task_type: str = "general") -> None:
        entry = {
            "model": model, "input_tokens": input_tokens,
            "output_tokens": output_tokens, "cost_usd": cost,
            "task_type": task_type,
            "timestamp": datetime.now(timezone.utc).isoformat()}
        self._usage_history.append(entry)
        if len(self._usage_history) > 10000:
            self._usage_history = self._usage_history[-10000:]

    def get_usage_stats(self) -> Dict[str, Any]:
        if not self._usage_history:
            return {"total_cost": 0, "total_tokens": 0, "request_count": 0}
        total_cost = sum(e["cost_usd"] for e in self._usage_history)
        total_input = sum(e["input_tokens"] for e in self._usage_history)
        total_output = sum(e["output_tokens"] for e in self._usage_history)
        by_model: Dict[str, float] = defaultdict(float)
        for e in self._usage_history:
            by_model[e["model"]] += e["cost_usd"]
        return {
            "total_cost_usd": round(total_cost, 4),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "request_count": len(self._usage_history),
            "cost_by_model": dict(by_model),
            "cache_hit_rate": self._cache_hits / max(1, self._cache_hits + self._cache_misses)}

    # ---- metrics ----
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "cache_size": len(self._prompt_cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "templates": len(self._templates),
            "models": len(self._models),
            **self.get_usage_stats()}


_layer: AIIntelligenceLayer | None = None

def get_ai_intelligence() -> AIIntelligenceLayer:
    global _layer
    if not _layer:
        _layer = AIIntelligenceLayer()
    return _layer
