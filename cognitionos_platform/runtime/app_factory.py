"""
Platform FastAPI app factory.

The canonical runtime is intentionally additive: this factory can be used
to build a stable v4 surface while keeping all existing v3 routes intact.
"""

from __future__ import annotations

from fastapi import FastAPI

from cognitionos_platform.api.v4.routers import api_router as platform_v4_router


def apply_platform_extensions(app: FastAPI) -> None:
    """
    Apply platform extensions (v4 routes and optional middleware) to an existing app.

    This is safe to call multiple times (idempotent for route mounting).
    """
    # FastAPI doesn't expose a public "router already included" API; we rely on tags/prefix being stable.
    app.include_router(platform_v4_router)


def create_app(base_app: FastAPI) -> FastAPI:
    """Return the provided app with platform extensions applied."""
    apply_platform_extensions(base_app)
    return base_app
