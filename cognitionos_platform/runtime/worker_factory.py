"""
Platform worker factory (Celery).

For self-host simplicity, this reuses the existing Celery configuration and
registers any additional platform task modules.
"""

from __future__ import annotations

from importlib import import_module
from typing import List

from celery import Celery


def create_worker() -> Celery:
    from infrastructure.tasks.celery_config import celery_app

    # Celery's autodiscovery is package-oriented; we keep this defensive to avoid
    # breaking minimal deployments.
    candidate_modules: List[str] = [
        "infrastructure.tasks.agent_tasks",
    ]
    for mod in candidate_modules:
        try:
            import_module(mod)
        except Exception:  # noqa: BLE001
            # Non-fatal: worker can still start and run other tasks.
            continue
    return celery_app
