"""
Plugin SQLAlchemy Models

Database models for plugin entities.
"""

from datetime import datetime
from sqlalchemy import (
    Column, String, DateTime, Enum as SQLEnum, Integer, Boolean, JSON, Float, ForeignKey, Text
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY
import uuid

from infrastructure.persistence.base import Base
from core.domain.plugin.entities import PluginStatus, PluginRuntime, ExecutionStatus


class PluginModel(Base):
    """SQLAlchemy model for Plugin entity"""
    
    __tablename__ = "plugins"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Manifest fields
    name = Column(String(255), nullable=False, index=True)
    version_major = Column(Integer, nullable=False)
    version_minor = Column(Integer, nullable=False)
    version_patch = Column(Integer, nullable=False)
    version_prerelease = Column(String(50), nullable=True)
    author = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    entry_point = Column(String(500), nullable=False)
    runtime = Column(
        SQLEnum(PluginRuntime, name="plugin_runtime_enum", create_type=False),
        nullable=False,
        index=True
    )
    
    # Manifest stored as JSON for complex fields
    permissions = Column(JSON, nullable=False, default=[])
    dependencies = Column(JSON, nullable=False, default={})
    sandbox_config = Column(JSON, nullable=False, default={})
    homepage_url = Column(String(500), nullable=True)
    documentation_url = Column(String(500), nullable=True)
    source_url = Column(String(500), nullable=True)
    tags = Column(ARRAY(String), nullable=False, default=[])
    
    # Plugin metadata
    status = Column(
        SQLEnum(PluginStatus, name="plugin_status_enum", create_type=False),
        nullable=False,
        index=True
    )
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=True, index=True)
    
    # Trust score
    trust_score = Column(Integer, nullable=True)
    trust_level = Column(String(50), nullable=True)
    trust_factors = Column(JSON, nullable=True)
    trust_last_calculated = Column(DateTime(timezone=True), nullable=True)
    
    # Statistics
    install_count = Column(Integer, nullable=False, default=0)
    execution_count = Column(Integer, nullable=False, default=0)
    success_count = Column(Integer, nullable=False, default=0)
    failure_count = Column(Integer, nullable=False, default=0)
    average_execution_time_ms = Column(Float, nullable=False, default=0.0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    published_at = Column(DateTime(timezone=True), nullable=True)
    deprecated_at = Column(DateTime(timezone=True), nullable=True)
    
    plugin_metadata = Column("metadata", JSON, nullable=False, default={})
    
    def __repr__(self):
        return f"<PluginModel(id={self.id}, name={self.name}, version={self.version_major}.{self.version_minor}.{self.version_patch}, status={self.status})>"


class PluginExecutionModel(Base):
    """SQLAlchemy model for PluginExecution entity"""
    
    __tablename__ = "plugin_executions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    plugin_id = Column(UUID(as_uuid=True), ForeignKey("plugins.id", ondelete="CASCADE"), nullable=False, index=True)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    
    status = Column(
        SQLEnum(ExecutionStatus, name="execution_status_enum", create_type=False),
        nullable=False,
        index=True
    )
    
    # Execution data
    input_data = Column(JSON, nullable=False)
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    
    # Timing
    start_time = Column(DateTime(timezone=True), nullable=False, index=True)
    end_time = Column(DateTime(timezone=True), nullable=True)
    
    # Context and logs
    execution_context = Column(JSON, nullable=False, default={})
    resource_usage = Column(JSON, nullable=False, default={})
    logs = Column(ARRAY(Text), nullable=False, default=[])
    
    def __repr__(self):
        return f"<PluginExecutionModel(id={self.id}, plugin_id={self.plugin_id}, status={self.status})>"


class PluginInstallationModel(Base):
    """SQLAlchemy model for PluginInstallation entity"""
    
    __tablename__ = "plugin_installations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    plugin_id = Column(UUID(as_uuid=True), ForeignKey("plugins.id", ondelete="CASCADE"), nullable=False, index=True)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    
    installed_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)
    configuration = Column(JSON, nullable=False, default={})
    enabled = Column(Boolean, nullable=False, default=True, index=True)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    usage_count = Column(Integer, nullable=False, default=0)
    
    def __repr__(self):
        return f"<PluginInstallationModel(id={self.id}, plugin_id={self.plugin_id}, tenant_id={self.tenant_id}, enabled={self.enabled})>"
