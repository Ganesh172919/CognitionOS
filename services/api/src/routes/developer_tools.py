"""
API Routes for Developer Tools
Exposes SDK generation and API documentation generation.
"""

from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from infrastructure.sdk.auto_generator import (
    SDKAutoGenerator,
    SDKLanguage,
    SDKGenerationConfig,
    SDKStyle
)
from infrastructure.dev_tools.api_doc_generator import (
    APIDocumentationGenerator,
    DocFormat,
    DocStyle,
    DocGenerationConfig
)

router = APIRouter(prefix="/api/v3/developer-tools", tags=["Developer Tools"])

# Initialize systems
sdk_generator = SDKAutoGenerator()
doc_generator = APIDocumentationGenerator()


# SDK Generation endpoints

class GenerateSDKRequest(BaseModel):
    """Request to generate SDK"""
    openapi_spec: dict = Field(..., description="OpenAPI specification")
    language: SDKLanguage = Field(..., description="Target language")
    package_name: str = Field(default="cognition_sdk", description="Package name")
    version: str = Field(default="1.0.0", description="SDK version")
    style: SDKStyle = Field(default=SDKStyle.BOTH, description="SDK style (async, sync, or both)")
    include_examples: bool = Field(default=True, description="Include example code")
    include_tests: bool = Field(default=True, description="Include unit tests")


@router.post("/sdk/generate")
async def generate_sdk(request: GenerateSDKRequest):
    """
    Generate SDK from OpenAPI specification

    Automatically generates production-ready SDK with:
    - Type-safe client
    - Async/await support
    - Retry logic
    - Rate limiting
    - Examples and tests
    """
    try:
        config = SDKGenerationConfig(
            language=request.language,
            style=request.style,
            package_name=request.package_name,
            version=request.version,
            include_examples=request.include_examples,
            include_tests=request.include_tests
        )

        sdk = await sdk_generator.generate_sdk(request.openapi_spec, config)

        return {
            "success": True,
            "language": sdk.language.value,
            "package_name": sdk.package_name,
            "version": sdk.version,
            "files": list(sdk.files.keys()),
            "file_count": len(sdk.files),
            "has_examples": len(sdk.examples) > 0,
            "has_tests": len(sdk.tests) > 0,
            "generated_at": sdk.generated_at.isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


class GenerateMultiSDKRequest(BaseModel):
    """Request to generate SDKs for multiple languages"""
    openapi_spec: dict = Field(..., description="OpenAPI specification")
    languages: List[SDKLanguage] = Field(..., description="Target languages")
    package_name: str = Field(default="cognition_sdk", description="Base package name")
    version: str = Field(default="1.0.0", description="SDK version")
    include_examples: bool = Field(default=True, description="Include examples")
    include_tests: bool = Field(default=True, description="Include tests")


@router.post("/sdk/generate-multi")
async def generate_multi_language_sdks(request: GenerateMultiSDKRequest):
    """
    Generate SDKs for multiple languages simultaneously

    Useful for maintaining consistency across language implementations
    """
    try:
        base_config = SDKGenerationConfig(
            language=SDKLanguage.PYTHON,  # Will be overridden per language
            package_name=request.package_name,
            version=request.version,
            include_examples=request.include_examples,
            include_tests=request.include_tests
        )

        sdks = await sdk_generator.generate_multi_language_sdks(
            request.openapi_spec,
            request.languages,
            base_config
        )

        return {
            "success": True,
            "languages": [lang.value for lang in request.languages],
            "sdks": {
                lang.value: {
                    "package_name": sdk.package_name,
                    "version": sdk.version,
                    "file_count": len(sdk.files)
                }
                for lang, sdk in sdks.items()
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/sdk/supported-languages")
async def get_supported_languages():
    """
    Get list of supported SDK languages

    Returns available languages and their capabilities
    """
    return {
        "success": True,
        "languages": [
            {
                "language": lang.value,
                "features": [
                    "Type safety",
                    "Async support",
                    "Retry logic",
                    "Rate limiting",
                    "Examples",
                    "Unit tests"
                ]
            }
            for lang in SDKLanguage
        ]
    }


# API Documentation endpoints

class GenerateDocsRequest(BaseModel):
    """Request to generate API documentation"""
    source_paths: List[str] = Field(..., description="Paths to scan for API routes")
    format: DocFormat = Field(default=DocFormat.MARKDOWN, description="Output format")
    style: DocStyle = Field(default=DocStyle.TECHNICAL, description="Documentation style")
    include_examples: bool = Field(default=True, description="Include code examples")
    include_sdk_snippets: bool = Field(default=True, description="Include SDK snippets")
    include_authentication_guide: bool = Field(default=True, description="Include auth guide")
    include_rate_limiting: bool = Field(default=True, description="Include rate limiting info")
    include_changelog: bool = Field(default=True, description="Include API changelog")
    include_error_codes: bool = Field(default=True, description="Include error codes")
    interactive: bool = Field(default=True, description="Generate interactive docs")
    dark_mode: bool = Field(default=True, description="Use dark mode theme")


@router.post("/docs/generate")
async def generate_documentation(request: GenerateDocsRequest):
    """
    Generate comprehensive API documentation

    Automatically scans source code and generates:
    - Complete endpoint documentation
    - Request/response schemas
    - Authentication guides
    - Code examples in multiple languages
    - Error codes reference
    - Interactive API playground
    """
    try:
        config = DocGenerationConfig(
            format=request.format,
            style=request.style,
            include_examples=request.include_examples,
            include_sdk_snippets=request.include_sdk_snippets,
            include_authentication_guide=request.include_authentication_guide,
            include_rate_limiting=request.include_rate_limiting,
            include_changelog=request.include_changelog,
            include_error_codes=request.include_error_codes,
            interactive=request.interactive,
            dark_mode=request.dark_mode
        )

        source_paths = [Path(p) for p in request.source_paths]
        doc = await doc_generator.generate_documentation(source_paths, config)

        return {
            "success": True,
            "title": doc.title,
            "version": doc.version,
            "endpoints_documented": len(doc.endpoints),
            "models_documented": len(doc.models),
            "format": request.format.value,
            "generated_at": doc.generated_at.isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/docs/generate-openapi")
async def generate_openapi_spec(source_paths: List[str] = Query(...)):
    """
    Generate OpenAPI/Swagger specification

    Creates standard OpenAPI 3.0 specification from source code
    """
    try:
        config = DocGenerationConfig(format=DocFormat.OPENAPI)
        paths = [Path(p) for p in source_paths]
        doc = await doc_generator.generate_documentation(paths, config)

        # Export as OpenAPI
        output_path = Path("/tmp/api-docs")
        result = await doc_generator.export_documentation(doc, output_path, config)

        return {
            "success": True,
            "format": "openapi",
            "endpoints": len(doc.endpoints),
            **result
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/docs/generate-playground")
async def generate_interactive_playground(source_paths: List[str] = Query(...)):
    """
    Generate interactive API playground

    Creates Swagger UI-based playground for testing APIs
    """
    try:
        config = DocGenerationConfig()
        paths = [Path(p) for p in source_paths]
        doc = await doc_generator.generate_documentation(paths, config)

        # Generate playground
        output_path = Path("/tmp/api-playground")
        result = await doc_generator.generate_interactive_playground(doc, output_path)

        return {
            "success": True,
            **result
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/docs/formats")
async def get_supported_formats():
    """
    Get supported documentation formats

    Returns list of available output formats
    """
    return {
        "success": True,
        "formats": [
            {
                "format": fmt.value,
                "description": {
                    "markdown": "GitHub-flavored Markdown documentation",
                    "html": "Standalone HTML documentation with styling",
                    "pdf": "PDF documentation (requires wkhtmltopdf)",
                    "openapi": "OpenAPI 3.0 specification (JSON)",
                    "postman": "Postman collection"
                }.get(fmt.value, "")
            }
            for fmt in DocFormat
        ]
    }


@router.get("/health")
async def health_check():
    """Health check endpoint for developer tools"""
    return {
        "success": True,
        "service": "developer-tools",
        "components": {
            "sdk_generator": "operational",
            "doc_generator": "operational"
        }
    }
