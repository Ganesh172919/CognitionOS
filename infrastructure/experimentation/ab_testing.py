"""A/B Testing Framework"""
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum

class ExperimentStatus(str, Enum):
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"

@dataclass
class Experiment:
    experiment_id: str
    name: str
    status: ExperimentStatus

class ABTestingFramework:
    """A/B testing framework"""
    def __init__(self, session):
        self.session = session
    
    async def create_experiment(self, name: str, description: str) -> Experiment:
        return Experiment(
            experiment_id="exp_1",
            name=name,
            status=ExperimentStatus.DRAFT
        )
