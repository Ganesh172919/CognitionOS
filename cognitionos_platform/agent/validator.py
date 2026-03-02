"""
Agent Validation Pipeline.

Performs syntax validation and optional repair loop using an LLM.
"""

from __future__ import annotations

import ast
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from infrastructure.llm.provider import LLMRequest, LLMRouter

from cognitionos_platform.agent.policy import evaluate_policies
from cognitionos_platform.sandbox.runner_client import ToolRunnerClient


logger = logging.getLogger(__name__)


def _validate_python_locally(files: List[Dict[str, Any]]) -> Dict[str, Any]:
    results: Dict[str, Any] = {"files": [], "ok": True}
    for f in files:
        path = str(f.get("path", "unknown"))
        content = f.get("content", "")
        if not isinstance(content, str):
            results["files"].append({"path": path, "ok": False, "error": "content is not a string"})
            results["ok"] = False
            continue
        try:
            ast.parse(content, filename=path)
            results["files"].append({"path": path, "ok": True})
        except SyntaxError as exc:
            results["files"].append(
                {
                    "path": path,
                    "ok": False,
                    "error": f"SyntaxError: {exc.msg} (line {exc.lineno}, col {exc.offset})",
                }
            )
            results["ok"] = False
    return results


async def _validate_python_in_sandbox(
    files: List[Dict[str, Any]],
    *,
    sandbox: ToolRunnerClient,
    user_id: str,
    agent_id: str,
) -> Dict[str, Any]:
    sources = {str(f.get("path", "unknown")): str(f.get("content", "")) for f in files}
    script = (
        "import json\n"
        "sources = " + json.dumps(sources) + "\n"
        "out = {\"files\": [], \"ok\": True}\n"
        "for path, src in sources.items():\n"
        "    try:\n"
        "        compile(src, path, 'exec')\n"
        "        out[\"files\"].append({\"path\": path, \"ok\": True})\n"
        "    except SyntaxError as e:\n"
        "        out[\"files\"].append({\"path\": path, \"ok\": False, \"error\": f\"SyntaxError: {e.msg} (line {e.lineno}, col {e.offset})\"})\n"
        "        out[\"ok\"] = False\n"
        "print(json.dumps(out))\n"
    )
    result = await sandbox.execute_tool(
        tool_name="execute_python",
        parameters={"code": script},
        permissions=["code_execution"],
        user_id=user_id,
        agent_id=agent_id,
        timeout_seconds=30,
    )
    stdout = (result.get("stdout") or "").strip()
    try:
        return json.loads(stdout) if stdout else {"ok": False, "files": [], "error": "no stdout"}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Sandbox validation returned non-JSON stdout", exc_info=True)
        return {"ok": False, "files": [], "error": f"invalid sandbox stdout: {exc}", "stdout": stdout}


async def validate(
    files: List[Dict[str, Any]],
    *,
    sandbox: Optional[ToolRunnerClient] = None,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    policy_mode: str = "warn",
) -> Dict[str, Any]:
    syntax_result: Dict[str, Any]
    if sandbox and user_id and agent_id:
        try:
            syntax_result = await _validate_python_in_sandbox(files, sandbox=sandbox, user_id=user_id, agent_id=agent_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sandbox validation failed; falling back to local", exc_info=True)
            syntax_result = _validate_python_locally(files) | {"sandbox_error": str(exc)}
    else:
        syntax_result = _validate_python_locally(files)

    policy = evaluate_policies(files, mode=policy_mode)
    combined_ok = bool(syntax_result.get("ok")) and bool(policy.get("ok"))

    result = dict(syntax_result)
    result["policy"] = policy
    result["policy_violations"] = list(policy.get("violations") or [])
    result["ok"] = combined_ok
    return result


async def repair_with_llm(
    *,
    requirement: str,
    plan: Dict[str, Any],
    previous: Dict[str, Any],
    validation: Dict[str, Any],
    llm: LLMRouter,
    model: str,
    max_tokens: int = 1600,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    system = (
        "You are a senior engineer fixing code generation errors. Output ONLY valid JSON. "
        "Schema: {\"version\":\"v1\",\"strategy\":\"llm_repair\",\"files\":[{\"path\":str,"
        "\"language\":\"python\",\"content\":str}],\"notes\":str}."
    )
    req = LLMRequest(
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "requirement": requirement,
                        "plan": plan,
                        "previous": previous,
                        "validation": validation,
                    }
                ),
            },
        ],
        model=model,
        temperature=0.2,
        max_tokens=max_tokens,
        metadata={"purpose": "agent_repair"},
    )
    resp = await llm.generate(req)
    usage = {"tokens": resp.usage, "cost_usd": resp.cost_usd, "model": resp.model, "provider": resp.provider.value}
    try:
        data = json.loads(resp.content)
        files = data.get("files", [])
        if not isinstance(files, list) or not files:
            raise ValueError("No files produced")
        for f in files:
            if not isinstance(f, dict) or "path" not in f or "content" not in f:
                raise ValueError("Invalid file entry")
        return data, usage
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM repair output invalid JSON; returning previous", exc_info=True)
        previous = dict(previous)
        previous["llm_repair_raw"] = resp.content
        previous["llm_repair_error"] = str(exc)
        return previous, usage
