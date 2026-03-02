"""Platform runtime settings (self-host first)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from core.config import CognitionOSConfig, get_config


@dataclass(frozen=True)
class PlatformRuntimeSettings:
    """Resolved runtime settings for the platform assembly."""

    core: CognitionOSConfig

    enable_v4: bool
    tool_runner_url: str
    license_token: Optional[str]


def get_platform_runtime_settings() -> PlatformRuntimeSettings:
    cfg = get_config()
    return PlatformRuntimeSettings(
        core=cfg,
        enable_v4=os.getenv("ENABLE_PLATFORM_V4", "true").lower() == "true",
        tool_runner_url=os.getenv("TOOL_RUNNER_URL", "http://tool-runner:8006").rstrip("/"),
        license_token=os.getenv("COGNITIONOS_LICENSE_TOKEN"),
    )
