"""
Advanced Systems API Routes

Routes for WAF, code generation, workflows, APM, and testing systems.
"""
from typing import Optional, List, Dict

from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.src.dependencies.injection import get_db_session
from infrastructure.security.waf_protection import (
    WAFProtection,
    AttackType
)
from infrastructure.codegen.code_generator import (
    CodeGenerator,
    CodeSpec,
    CodeLanguage
)
from infrastructure.workflow_builder.builder import (
    WorkflowBuilder,
    WorkflowNodeType
)
from infrastructure.apm.apm_system import APMSystem
from infrastructure.testing.test_generator import TestGenerator


router = APIRouter(prefix="/api/v3/advanced", tags=["advanced"])

# Initialize systems
waf = WAFProtection()
code_gen = CodeGenerator()
workflow_builder = WorkflowBuilder()
apm = APMSystem()
test_gen = TestGenerator()


@router.get("/waf/statistics")
async def get_waf_statistics(
    time_window: int = 3600
):
    """Get WAF threat statistics"""
    stats = await waf.get_threat_statistics(time_window)
    return stats


@router.get("/waf/threats")
async def get_recent_threats(
    limit: int = 100,
    attack_type: Optional[AttackType] = None
):
    """Get recent security threats"""
    threats = await waf.get_recent_threats(limit, attack_type)
    return {
        "total": len(threats),
        "threats": [
            {
                "threat_id": t.threat_id,
                "attack_type": t.attack_type.value,
                "threat_level": t.threat_level.value,
                "source_ip": t.source_ip,
                "target_path": t.target_path,
                "detected_at": t.detected_at.isoformat(),
                "blocked": t.blocked,
                "reason": t.reason
            }
            for t in threats
        ]
    }


@router.post("/waf/block-ip")
async def block_ip(
    ip: str,
    duration: Optional[int] = None
):
    """Block an IP address"""
    await waf.block_ip(ip, duration)
    return {
        "success": True,
        "ip": ip,
        "duration": duration
    }


@router.post("/codegen/generate")
async def generate_code(
    spec: Dict = Body(...)
):
    """Generate code from specification"""
    code_spec = CodeSpec(
        name=spec["name"],
        description=spec["description"],
        language=CodeLanguage(spec["language"]),
        functionality=spec["functionality"],
        inputs=spec.get("inputs", []),
        outputs=spec.get("outputs", [])
    )
    
    generated = await code_gen.generate_code(code_spec)
    
    return {
        "language": generated.language.value,
        "code": generated.code,
        "documentation": generated.documentation,
        "tests": generated.tests,
        "quality_score": generated.quality_score,
        "generated_at": generated.generated_at.isoformat()
    }


@router.get("/workflows/templates")
async def get_workflow_templates(
    category: Optional[str] = None
):
    """Get workflow templates"""
    templates = await workflow_builder.get_templates(category)
    
    return {
        "total": len(templates),
        "templates": [
            {
                "id": t.id,
                "name": t.name,
                "description": t.description,
                "category": t.category,
                "version": t.version,
                "node_count": len(t.nodes),
                "edge_count": len(t.edges)
            }
            for t in templates
        ]
    }


@router.post("/workflows/create")
async def create_workflow(
    name: str,
    description: str,
    template_id: Optional[str] = None
):
    """Create new workflow"""
    workflow = await workflow_builder.create_workflow(
        name, description, template_id
    )
    
    return {
        "id": workflow.id,
        "name": workflow.name,
        "description": workflow.description,
        "category": workflow.category,
        "version": workflow.version,
        "created_at": workflow.created_at.isoformat()
    }


@router.post("/workflows/{workflow_id}/validate")
async def validate_workflow(workflow_id: str):
    """Validate workflow structure"""
    validation = await workflow_builder.validate_workflow(workflow_id)
    return validation


@router.get("/apm/summary")
async def get_apm_summary(
    time_window: int = 3600
):
    """Get APM performance summary"""
    summary = await apm.get_performance_summary(time_window)
    return summary


@router.post("/apm/transaction/start")
async def start_transaction(
    name: str,
    trace_id: str
):
    """Start monitoring transaction"""
    trace_id = await apm.start_transaction(name, trace_id)
    return {
        "trace_id": trace_id,
        "started": True
    }


@router.post("/apm/transaction/end")
async def end_transaction(
    trace_id: str,
    status: str = "success"
):
    """End transaction monitoring"""
    trace = await apm.end_transaction(trace_id, status)
    
    return {
        "trace_id": trace.trace_id,
        "duration_ms": trace.duration_ms,
        "status": trace.status
    }


@router.post("/apm/metric")
async def record_metric(
    name: str,
    value: float,
    unit: str = "count",
    tags: Optional[Dict[str, str]] = None
):
    """Record performance metric"""
    await apm.record_metric(name, value, unit, tags)
    return {"success": True}


@router.post("/testing/generate")
async def generate_tests(
    function_name: str
):
    """Generate unit tests for function"""
    tests = await test_gen.generate_unit_tests(function_name)
    
    return {
        "function_name": function_name,
        "tests_generated": len(tests),
        "tests": [
            {
                "name": t.name,
                "type": t.test_type,
                "code": t.code
            }
            for t in tests
        ]
    }
