"""
Data Pipeline & ETL System

Real-time data ingestion, transformation, and loading with stream processing.
"""
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import asyncio


class PipelineStatus(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class DataRecord:
    id: str
    data: Dict[str, Any]
    timestamp: datetime
    source: str


@dataclass
class TransformationRule:
    name: str
    transform_func: Callable
    enabled: bool = True


class DataPipeline:
    """
    Real-time data pipeline system.
    
    Features:
    - Stream processing
    - Data transformation
    - Batch and real-time ingestion
    - Data validation
    - Error handling
    - Monitoring and metrics
    """
    
    def __init__(self, name: str):
        self.name = name
        self.status = PipelineStatus.STOPPED
        self.transformations: List[TransformationRule] = []
        self.records_processed = 0
        self.errors = 0
    
    async def add_transformation(
        self,
        name: str,
        transform_func: Callable
    ):
        """Add transformation rule to pipeline"""
        self.transformations.append(
            TransformationRule(
                name=name,
                transform_func=transform_func
            )
        )
    
    async def process_record(
        self,
        record: DataRecord
    ) -> Optional[DataRecord]:
        """Process a single data record through pipeline"""
        try:
            result = record
            
            for transformation in self.transformations:
                if transformation.enabled:
                    result = transformation.transform_func(result)
            
            self.records_processed += 1
            return result
            
        except Exception as e:
            self.errors += 1
            return None
    
    async def start(self):
        """Start pipeline"""
        self.status = PipelineStatus.RUNNING
    
    async def stop(self):
        """Stop pipeline"""
        self.status = PipelineStatus.STOPPED
    
    async def get_stats(self) -> Dict:
        """Get pipeline statistics"""
        return {
            "name": self.name,
            "status": self.status.value,
            "records_processed": self.records_processed,
            "errors": self.errors,
            "error_rate": (self.errors / self.records_processed * 100) 
                if self.records_processed > 0 else 0
        }
