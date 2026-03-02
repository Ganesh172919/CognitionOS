"""v4 API router aggregation."""

from fastapi import APIRouter

from cognitionos_platform.api.v4.routers.system import router as system_router
from cognitionos_platform.api.v4.routers.agent_runs import router as agent_runs_router
from cognitionos_platform.api.v4.routers.billing import router as billing_router
from cognitionos_platform.api.v4.routers.tenants import router as tenants_router
from cognitionos_platform.api.v4.routers.api_keys import router as api_keys_router


api_router = APIRouter()
api_router.include_router(system_router)
api_router.include_router(agent_runs_router)
api_router.include_router(billing_router)
api_router.include_router(tenants_router)
api_router.include_router(api_keys_router)
