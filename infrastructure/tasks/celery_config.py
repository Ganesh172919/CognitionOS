"""
Celery Configuration for CognitionOS V3

Provides async task queue for long-running workflow executions.
"""

import sys
import os

# Add core modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from celery import Celery
from core.config import get_config


# Get configuration
config = get_config()

# Create Celery app
celery_app = Celery(
    "cognitionos",
    broker=config.celery.broker_url,
    backend=config.celery.result_backend,
)

# Configure Celery
celery_app.conf.update(
    task_serializer=config.celery.task_serializer,
    result_serializer=config.celery.result_serializer,
    accept_content=config.celery.accept_content,
    timezone=config.celery.timezone,
    enable_utc=config.celery.enable_utc,
    task_track_started=config.celery.task_track_started,
    task_time_limit=config.celery.task_time_limit,
    task_soft_time_limit=config.celery.task_soft_time_limit,
    # Task routing
    task_routes={
        'infrastructure.tasks.workflow_tasks.*': {'queue': 'workflows'},
        'infrastructure.tasks.agent_tasks.*': {'queue': 'agents'},
    },
    # Result expiration
    result_expires=3600,
    # Worker settings
    worker_prefetch_multiplier=4,
    worker_max_tasks_per_child=1000,
)

# Auto-discover tasks
celery_app.autodiscover_tasks([
    'infrastructure.tasks.workflow_tasks',
    'infrastructure.tasks.agent_tasks',
])


if __name__ == "__main__":
    celery_app.start()
