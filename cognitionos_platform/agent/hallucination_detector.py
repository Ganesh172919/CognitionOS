"""
Hallucination Detector - Heuristic checks on generated code.

Reduces hallucinations by validating imports, paths, and API usage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class HallucinationFlag:
    """Flag for potential hallucination."""

    rule: str
    message: str
    path: str
    line: Optional[int] = None
    snippet: Optional[str] = None


# Dangerous imports that suggest hallucination or unsafe code
DANGEROUS_IMPORTS = {
    "os.system",
    "subprocess.Popen",
    "subprocess.call",
    "subprocess.run",
    "eval",
    "exec",
    "compile",
    "__import__",
}

# Suspicious path patterns
SUSPICIOUS_PATH_PATTERNS = [
    r"/etc/",
    r"/usr/",
    r"/var/",
    r"\.\./",
    r"~\w",
    r"C:\\",
    r"[A-Z]:\\",
]

# Whitelisted HTTP hosts (empty = none allowed in generated code by default)
ALLOWED_HTTP_HOSTS: Set[str] = set()


class HallucinationDetector:
    """
    Heuristic detector for potential LLM hallucinations in generated code.
    """

    def __init__(
        self,
        dangerous_imports: Optional[Set[str]] = None,
        allowed_hosts: Optional[Set[str]] = None,
    ):
        self.dangerous_imports = dangerous_imports or DANGEROUS_IMPORTS
        self.allowed_hosts = allowed_hosts or ALLOWED_HTTP_HOSTS

    def detect(
        self,
        files: List[Dict[str, Any]],
    ) -> List[HallucinationFlag]:
        """
        Run heuristic checks on generated files.
        Returns list of potential hallucination flags.
        """
        flags: List[HallucinationFlag] = []
        for f in files:
            path = str(f.get("path", "unknown"))
            content = f.get("content", "")
            if not isinstance(content, str):
                flags.append(
                    HallucinationFlag("invalid_content", "content must be string", path)
                )
                continue
            flags.extend(self._check_file(path, content))
        return flags

    def _check_file(self, path: str, content: str) -> List[HallucinationFlag]:
        flags = []
        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            for imp in self.dangerous_imports:
                if imp in line_stripped and not line_stripped.startswith("#"):
                    flags.append(
                        HallucinationFlag(
                            "dangerous_import",
                            f"Suspicious import: {imp}",
                            path,
                            line=i,
                            snippet=line_stripped[:100],
                        )
                    )
            for pat in SUSPICIOUS_PATH_PATTERNS:
                if re.search(pat, line_stripped):
                    flags.append(
                        HallucinationFlag(
                            "suspicious_path",
                            f"Path may escape sandbox: {pat}",
                            path,
                            line=i,
                            snippet=line_stripped[:100],
                        )
                    )
            if "requests.get(" in line_stripped or "httpx.get(" in line_stripped:
                if self.allowed_hosts and "localhost" not in line_stripped:
                    flags.append(
                        HallucinationFlag(
                            "external_http",
                            "HTTP call without whitelisted host",
                            path,
                            line=i,
                            snippet=line_stripped[:100],
                        )
                    )
        return flags
