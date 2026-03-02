"""
Policy checks for generated artifacts.

This module is intentionally lightweight and dependency-free. It provides:
- Path safety validation (no absolute paths / traversal)
- Python AST safety checks (dangerous builtins / disallowed imports)

The policy output is designed to be persisted as part of validation reports.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional, Set, Tuple

_WINDOWS_ABS_RE = re.compile(r"^[a-zA-Z]:[\\\\/].*")
_WINDOWS_UNC_RE = re.compile(r"^\\\\\\\\[^\\\\]+\\\\[^\\\\]+.*")


def _normalize_path(path: str) -> str:
    return path.replace("\\\\", "/").replace("\\", "/")


def _is_absolute_path(path: str) -> bool:
    p = _normalize_path(path)
    return p.startswith("/") or bool(_WINDOWS_ABS_RE.match(p)) or bool(_WINDOWS_UNC_RE.match(path))


def _has_traversal(path: str) -> bool:
    parts = [seg for seg in _normalize_path(path).split("/") if seg not in ("", ".")]
    return any(seg == ".." for seg in parts)


def validate_artifact_path(path: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Validate that an artifact path is safe to persist/apply.

    Returns: (ok, violation_dict_or_none)
    """
    if not isinstance(path, str) or not path.strip():
        return False, {
            "kind": "invalid_path",
            "severity": "error",
            "message": "File path must be a non-empty string",
            "path": str(path),
        }

    if "\x00" in path:
        return False, {
            "kind": "invalid_path",
            "severity": "error",
            "message": "File path contains NUL byte",
            "path": path,
        }

    if path.startswith("~"):
        return False, {
            "kind": "absolute_path",
            "severity": "error",
            "message": "Tilde paths are not allowed",
            "path": path,
        }

    if _is_absolute_path(path):
        return False, {
            "kind": "absolute_path",
            "severity": "error",
            "message": "Absolute paths are not allowed",
            "path": path,
        }

    if _has_traversal(path):
        return False, {
            "kind": "path_traversal",
            "severity": "error",
            "message": "Path traversal ('..') is not allowed",
            "path": path,
        }

    return True, None


def _import_root(module: str) -> str:
    return (module or "").split(".", 1)[0]


def _scan_python_ast(
    tree: ast.AST,
    *,
    disallowed_imports: Set[str],
    dangerous_builtins: Set[str],
    severity: str,
) -> List[Dict[str, Any]]:
    violations: List[Dict[str, Any]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = _import_root(alias.name)
                if root in disallowed_imports:
                    violations.append(
                        {
                            "kind": "disallowed_import",
                            "severity": severity,
                            "message": f"Import of '{root}' is disallowed by policy",
                            "detail": {"module": alias.name},
                            "line": getattr(node, "lineno", None),
                            "col": getattr(node, "col_offset", None),
                        }
                    )

        if isinstance(node, ast.ImportFrom):
            root = _import_root(node.module or "")
            if root in disallowed_imports:
                violations.append(
                    {
                        "kind": "disallowed_import",
                        "severity": severity,
                        "message": f"Import of '{root}' is disallowed by policy",
                        "detail": {"module": node.module, "names": [n.name for n in node.names]},
                        "line": getattr(node, "lineno", None),
                        "col": getattr(node, "col_offset", None),
                    }
                )

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in dangerous_builtins:
                violations.append(
                    {
                        "kind": "dangerous_builtin",
                        "severity": severity,
                        "message": f"Call to '{node.func.id}' is disallowed by policy",
                        "detail": {"builtin": node.func.id},
                        "line": getattr(node, "lineno", None),
                        "col": getattr(node, "col_offset", None),
                    }
                )

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            # Detect common dangerous patterns even when module is aliased/imported differently.
            attr = node.func.attr
            if attr in {"system", "popen", "Popen"}:
                violations.append(
                    {
                        "kind": "dangerous_call",
                        "severity": severity,
                        "message": f"Call to '{attr}' is disallowed by policy",
                        "detail": {"attribute": attr},
                        "line": getattr(node, "lineno", None),
                        "col": getattr(node, "col_offset", None),
                    }
                )

    return violations


def evaluate_policies(
    files: List[Dict[str, Any]],
    *,
    mode: str = "warn",
    disallowed_imports: Optional[Set[str]] = None,
    dangerous_builtins: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate safety policies for generated files.

    mode:
      - "warn": violations are reported but do not block (except hard path violations)
      - "block": violations block validation success
    """
    normalized_mode = (mode or "warn").strip().lower()
    if normalized_mode not in {"warn", "block"}:
        normalized_mode = "warn"

    disallowed = set(disallowed_imports or {"subprocess", "socket", "ctypes"})
    dangerous = set(dangerous_builtins or {"eval", "exec"})

    violations: List[Dict[str, Any]] = []
    hard_block = False
    severity = "warning" if normalized_mode == "warn" else "error"

    for f in files or []:
        path = str(f.get("path") or "")
        ok, v = validate_artifact_path(path)
        if not ok and v:
            violations.append(v)
            hard_block = True
            continue

        # Only scan Python sources (or python-labeled artifacts).
        language = str(f.get("language") or "").lower()
        if language and language != "python":
            continue
        if not language and not path.endswith(".py"):
            continue

        content = f.get("content", "")
        if not isinstance(content, str):
            violations.append(
                {
                    "kind": "invalid_content",
                    "severity": "error",
                    "message": "File content must be a string for policy scanning",
                    "path": path,
                }
            )
            hard_block = True
            continue

        try:
            tree = ast.parse(content, filename=path or "<generated>")
        except SyntaxError:
            # Syntax validation will report this; policy scanning skips.
            continue

        for v in _scan_python_ast(tree, disallowed_imports=disallowed, dangerous_builtins=dangerous, severity=severity):
            v["path"] = path
            violations.append(v)

    # Hard path/content violations always block.
    if hard_block:
        ok = False
    else:
        ok = normalized_mode == "warn" or not any(v.get("severity") == "error" for v in violations)

    return {
        "mode": normalized_mode,
        "ok": bool(ok),
        "hard_block": bool(hard_block),
        "violations": violations,
        "disallowed_imports": sorted(disallowed),
        "dangerous_builtins": sorted(dangerous),
    }

