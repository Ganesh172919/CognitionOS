"""
API Documentation Auto-Generator
Automatically generates comprehensive API documentation from code annotations, OpenAPI specs, and docstrings.
Produces interactive documentation with examples, authentication guides, and SDKs.
"""

import ast
import inspect
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import markdown


class DocFormat(str, Enum):
    """Documentation output formats"""
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    OPENAPI = "openapi"
    POSTMAN = "postman"


class DocStyle(str, Enum):
    """Documentation style"""
    TECHNICAL = "technical"
    BEGINNER_FRIENDLY = "beginner_friendly"
    ENTERPRISE = "enterprise"


@dataclass
class APIEndpoint:
    """Parsed API endpoint"""
    path: str
    method: str
    function_name: str
    summary: str
    description: str
    parameters: List[Dict[str, Any]]
    request_body: Optional[Dict[str, Any]]
    responses: Dict[int, Dict[str, Any]]
    tags: List[str]
    deprecated: bool
    examples: List[Dict[str, Any]]
    authentication: List[str]


@dataclass
class APIDocumentation:
    """Complete API documentation"""
    title: str
    version: str
    description: str
    base_url: str
    endpoints: List[APIEndpoint]
    models: Dict[str, Dict[str, Any]]
    authentication: Dict[str, Any]
    rate_limiting: Optional[Dict[str, Any]]
    pagination: Optional[Dict[str, Any]]
    error_codes: Dict[str, Any]
    changelog: List[Dict[str, Any]]
    examples: Dict[str, List[str]]
    generated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DocGenerationConfig:
    """Configuration for documentation generation"""
    format: DocFormat = DocFormat.MARKDOWN
    style: DocStyle = DocStyle.TECHNICAL
    include_examples: bool = True
    include_sdk_snippets: bool = True
    include_authentication_guide: bool = True
    include_rate_limiting: bool = True
    include_changelog: bool = True
    include_error_codes: bool = True
    interactive: bool = True
    dark_mode: bool = True


class APIDocumentationGenerator:
    """
    Automatically generates API documentation from code

    Features:
    - Parses FastAPI routes and extracts metadata
    - Generates comprehensive endpoint documentation
    - Creates interactive examples
    - Includes authentication guides
    - Generates SDK code snippets
    - Creates OpenAPI/Swagger specs
    - Exports to multiple formats (Markdown, HTML, PDF)
    """

    def __init__(self):
        self.endpoints: List[APIEndpoint] = []
        self.models: Dict[str, Dict[str, Any]] = {}

    async def generate_documentation(
        self,
        source_paths: List[Path],
        config: DocGenerationConfig
    ) -> APIDocumentation:
        """
        Generate comprehensive API documentation

        Args:
            source_paths: List of paths to scan for API routes
            config: Documentation generation configuration

        Returns:
            Complete API documentation
        """
        # Parse source files
        for path in source_paths:
            if path.is_file():
                await self._parse_file(path)
            elif path.is_dir():
                for file_path in path.rglob("*.py"):
                    await self._parse_file(file_path)

        # Build documentation
        doc = APIDocumentation(
            title="CognitionOS API",
            version="3.0.0",
            description="Comprehensive API for CognitionOS platform",
            base_url="https://api.cognitionos.ai",
            endpoints=self.endpoints,
            models=self.models,
            authentication=self._generate_auth_section(),
            rate_limiting=self._generate_rate_limiting_section() if config.include_rate_limiting else None,
            pagination=self._generate_pagination_section(),
            error_codes=self._generate_error_codes() if config.include_error_codes else {},
            changelog=self._generate_changelog() if config.include_changelog else [],
            examples=self._generate_examples() if config.include_examples else {}
        )

        return doc

    async def _parse_file(self, file_path: Path):
        """Parse Python file and extract API endpoints"""
        try:
            with open(file_path, 'r') as f:
                source = f.read()

            tree = ast.parse(source)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    endpoint = self._extract_endpoint_from_function(node, source)
                    if endpoint:
                        self.endpoints.append(endpoint)

                elif isinstance(node, ast.ClassDef):
                    # Extract Pydantic models
                    model_info = self._extract_model_from_class(node, source)
                    if model_info:
                        self.models[node.name] = model_info

        except Exception as e:
            print(f"Error parsing {file_path}: {e}")

    def _extract_endpoint_from_function(
        self,
        node: ast.FunctionDef,
        source: str
    ) -> Optional[APIEndpoint]:
        """Extract endpoint information from function"""
        # Check for route decorators (@router.get, @router.post, etc.)
        method = None
        path = None

        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call):
                if hasattr(decorator.func, 'attr'):
                    method_name = decorator.func.attr
                    if method_name in ['get', 'post', 'put', 'delete', 'patch']:
                        method = method_name.upper()

                        # Extract path
                        if decorator.args and isinstance(decorator.args[0], ast.Constant):
                            path = decorator.args[0].value

        if not method or not path:
            return None

        # Extract docstring
        docstring = ast.get_docstring(node) or ""

        # Parse docstring
        summary, description = self._parse_docstring(docstring)

        # Extract parameters
        parameters = self._extract_parameters(node)

        # Extract response types
        responses = self._extract_responses(node, docstring)

        # Extract tags
        tags = self._extract_tags(path)

        # Generate examples
        examples = self._generate_endpoint_examples(path, method, parameters)

        return APIEndpoint(
            path=path,
            method=method,
            function_name=node.name,
            summary=summary,
            description=description,
            parameters=parameters,
            request_body=None,  # TODO: Extract from Pydantic models
            responses=responses,
            tags=tags,
            deprecated=False,
            examples=examples,
            authentication=["bearer"]
        )

    def _parse_docstring(self, docstring: str) -> tuple[str, str]:
        """Parse docstring into summary and description"""
        lines = docstring.strip().split('\n')
        if not lines:
            return "", ""

        summary = lines[0].strip()
        description = '\n'.join(lines[1:]).strip()

        return summary, description

    def _extract_parameters(self, node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Extract function parameters"""
        parameters = []

        for arg in node.args.args:
            if arg.arg in ['self', 'cls']:
                continue

            param_info = {
                "name": arg.arg,
                "type": self._get_type_annotation(arg.annotation) if arg.annotation else "string",
                "required": True,
                "in": "query",  # Default to query param
                "description": ""
            }

            parameters.append(param_info)

        return parameters

    def _get_type_annotation(self, annotation) -> str:
        """Get type from annotation"""
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Constant):
            return str(annotation.value)
        else:
            return "string"

    def _extract_responses(
        self,
        node: ast.FunctionDef,
        docstring: str
    ) -> Dict[int, Dict[str, Any]]:
        """Extract response information"""
        responses = {
            200: {
                "description": "Successful response",
                "content": {
                    "application/json": {
                        "schema": {"type": "object"}
                    }
                }
            }
        }

        return responses

    def _extract_tags(self, path: str) -> List[str]:
        """Extract tags from path"""
        parts = path.strip('/').split('/')
        if parts and parts[0]:
            # Use first part of path as tag
            return [parts[0].replace('-', ' ').title()]
        return []

    def _generate_endpoint_examples(
        self,
        path: str,
        method: str,
        parameters: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate example requests for endpoint"""
        examples = []

        # cURL example
        curl_example = self._generate_curl_example(path, method, parameters)
        examples.append({
            "language": "curl",
            "code": curl_example
        })

        # Python example
        python_example = self._generate_python_example(path, method, parameters)
        examples.append({
            "language": "python",
            "code": python_example
        })

        # JavaScript example
        js_example = self._generate_javascript_example(path, method, parameters)
        examples.append({
            "language": "javascript",
            "code": js_example
        })

        return examples

    def _generate_curl_example(
        self,
        path: str,
        method: str,
        parameters: List[Dict[str, Any]]
    ) -> str:
        """Generate cURL example"""
        base = f"curl -X {method} \\\n  https://api.cognitionos.ai{path}"

        if method in ["POST", "PUT", "PATCH"]:
            base += " \\\n  -H 'Content-Type: application/json' \\\n  -H 'Authorization: Bearer YOUR_API_KEY'"
            base += " \\\n  -d '{\n    \"example\": \"data\"\n  }'"
        else:
            base += " \\\n  -H 'Authorization: Bearer YOUR_API_KEY'"

        return base

    def _generate_python_example(
        self,
        path: str,
        method: str,
        parameters: List[Dict[str, Any]]
    ) -> str:
        """Generate Python example"""
        return f'''import httpx

async def example():
    async with httpx.AsyncClient() as client:
        response = await client.{method.lower()}(
            "https://api.cognitionos.ai{path}",
            headers={{"Authorization": "Bearer YOUR_API_KEY"}}
        )
        return response.json()
'''

    def _generate_javascript_example(
        self,
        path: str,
        method: str,
        parameters: List[Dict[str, Any]]
    ) -> str:
        """Generate JavaScript example"""
        return f'''const response = await fetch('https://api.cognitionos.ai{path}', {{
  method: '{method}',
  headers: {{
    'Authorization': 'Bearer YOUR_API_KEY',
    'Content-Type': 'application/json'
  }}
}});

const data = await response.json();
console.log(data);
'''

    def _extract_model_from_class(
        self,
        node: ast.ClassDef,
        source: str
    ) -> Optional[Dict[str, Any]]:
        """Extract Pydantic model information"""
        # Check if it's a Pydantic model (has BaseModel parent)
        is_model = False
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == 'BaseModel':
                is_model = True

        if not is_model:
            return None

        model_info = {
            "name": node.name,
            "description": ast.get_docstring(node) or "",
            "properties": {},
            "required": []
        }

        # Extract fields
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                field_name = item.target.id
                field_type = self._get_type_annotation(item.annotation) if item.annotation else "string"

                model_info["properties"][field_name] = {
                    "type": field_type,
                    "description": ""
                }

                # Check if required (no default value)
                if item.value is None:
                    model_info["required"].append(field_name)

        return model_info

    def _generate_auth_section(self) -> Dict[str, Any]:
        """Generate authentication documentation"""
        return {
            "type": "bearer",
            "scheme": "bearer",
            "bearer_format": "JWT",
            "description": "Use Bearer token authentication. Include your API key in the Authorization header.",
            "example": "Authorization: Bearer YOUR_API_KEY",
            "how_to_get": "Generate an API key from your dashboard at https://dashboard.cognitionos.ai/api-keys"
        }

    def _generate_rate_limiting_section(self) -> Dict[str, Any]:
        """Generate rate limiting documentation"""
        return {
            "default_limit": "100 requests per minute",
            "burst_limit": "10 requests per second",
            "headers": {
                "X-RateLimit-Limit": "Maximum requests per window",
                "X-RateLimit-Remaining": "Remaining requests in window",
                "X-RateLimit-Reset": "Time when limit resets (Unix timestamp)"
            },
            "upgrade_for_higher_limits": "https://cognitionos.ai/pricing"
        }

    def _generate_pagination_section(self) -> Dict[str, Any]:
        """Generate pagination documentation"""
        return {
            "style": "cursor-based",
            "parameters": {
                "limit": {
                    "type": "integer",
                    "default": 20,
                    "max": 100,
                    "description": "Number of items per page"
                },
                "cursor": {
                    "type": "string",
                    "description": "Cursor for next page (from previous response)"
                }
            },
            "response_format": {
                "data": "Array of items",
                "pagination": {
                    "next_cursor": "Cursor for next page",
                    "has_more": "Boolean indicating more pages"
                }
            }
        }

    def _generate_error_codes(self) -> Dict[str, Any]:
        """Generate error codes documentation"""
        return {
            "400": {
                "name": "Bad Request",
                "description": "Invalid request parameters",
                "example": {
                    "error": "validation_error",
                    "message": "Invalid parameter: limit must be between 1 and 100"
                }
            },
            "401": {
                "name": "Unauthorized",
                "description": "Missing or invalid API key",
                "example": {
                    "error": "unauthorized",
                    "message": "Invalid API key"
                }
            },
            "403": {
                "name": "Forbidden",
                "description": "Insufficient permissions",
                "example": {
                    "error": "forbidden",
                    "message": "This resource requires enterprise plan"
                }
            },
            "404": {
                "name": "Not Found",
                "description": "Resource not found",
                "example": {
                    "error": "not_found",
                    "message": "Resource with ID 'xyz' not found"
                }
            },
            "429": {
                "name": "Too Many Requests",
                "description": "Rate limit exceeded",
                "example": {
                    "error": "rate_limit_exceeded",
                    "message": "Rate limit exceeded. Retry after 60 seconds."
                }
            },
            "500": {
                "name": "Internal Server Error",
                "description": "Server error",
                "example": {
                    "error": "internal_error",
                    "message": "An unexpected error occurred"
                }
            }
        }

    def _generate_changelog(self) -> List[Dict[str, Any]]:
        """Generate API changelog"""
        return [
            {
                "version": "3.0.0",
                "date": "2024-01-15",
                "changes": [
                    "Added autonomous agent endpoints",
                    "New revenue systems API",
                    "Enhanced performance optimization endpoints",
                    "Added engagement systems (recommendations, referrals)"
                ]
            },
            {
                "version": "2.0.0",
                "date": "2023-12-01",
                "changes": [
                    "Complete API redesign",
                    "Improved authentication",
                    "Better error handling"
                ]
            }
        ]

    def _generate_examples(self) -> Dict[str, List[str]]:
        """Generate usage examples"""
        return {
            "quickstart": [
                "# Quick Start Guide\n\n1. Get your API key\n2. Make your first request\n3. Explore endpoints"
            ],
            "authentication": [
                "# Authentication\n\nInclude your API key in all requests:\n\n```\nAuthorization: Bearer YOUR_API_KEY\n```"
            ]
        }

    async def export_documentation(
        self,
        doc: APIDocumentation,
        output_path: Path,
        config: DocGenerationConfig
    ) -> Dict[str, Any]:
        """Export documentation to file"""
        if config.format == DocFormat.MARKDOWN:
            content = self._generate_markdown(doc, config)
            output_file = output_path / "API_DOCUMENTATION.md"
        elif config.format == DocFormat.HTML:
            content = self._generate_html(doc, config)
            output_file = output_path / "api-docs.html"
        elif config.format == DocFormat.OPENAPI:
            content = self._generate_openapi(doc)
            output_file = output_path / "openapi.json"
        else:
            raise ValueError(f"Unsupported format: {config.format}")

        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(content)

        return {
            "success": True,
            "output_file": str(output_file),
            "format": config.format.value,
            "endpoints_documented": len(doc.endpoints)
        }

    def _generate_markdown(self, doc: APIDocumentation, config: DocGenerationConfig) -> str:
        """Generate Markdown documentation"""
        lines = [
            f"# {doc.title}",
            "",
            f"**Version:** {doc.version}",
            f"**Base URL:** {doc.base_url}",
            f"**Generated:** {doc.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"{doc.description}",
            "",
            "## Table of Contents",
            "",
            "- [Authentication](#authentication)",
            "- [Rate Limiting](#rate-limiting)",
            "- [Pagination](#pagination)",
            "- [Error Codes](#error-codes)",
            "- [Endpoints](#endpoints)",
            "",
        ]

        # Authentication
        if config.include_authentication_guide:
            lines.extend([
                "## Authentication",
                "",
                f"**Type:** {doc.authentication['type']}",
                "",
                f"{doc.authentication['description']}",
                "",
                "**Example:**",
                "```",
                doc.authentication['example'],
                "```",
                "",
            ])

        # Rate Limiting
        if config.include_rate_limiting and doc.rate_limiting:
            lines.extend([
                "## Rate Limiting",
                "",
                f"- **Default Limit:** {doc.rate_limiting['default_limit']}",
                f"- **Burst Limit:** {doc.rate_limiting['burst_limit']}",
                "",
                "**Response Headers:**",
                "",
            ])
            for header, description in doc.rate_limiting['headers'].items():
                lines.append(f"- `{header}`: {description}")
            lines.append("")

        # Pagination
        if doc.pagination:
            lines.extend([
                "## Pagination",
                "",
                f"**Style:** {doc.pagination['style']}",
                "",
                "**Parameters:**",
                "",
            ])
            for param, details in doc.pagination['parameters'].items():
                lines.append(f"- `{param}` ({details['type']}): {details['description']}")
            lines.append("")

        # Error Codes
        if config.include_error_codes:
            lines.extend([
                "## Error Codes",
                "",
            ])
            for code, error_info in doc.error_codes.items():
                lines.extend([
                    f"### {code} - {error_info['name']}",
                    "",
                    error_info['description'],
                    "",
                    "**Example Response:**",
                    "```json",
                    json.dumps(error_info['example'], indent=2),
                    "```",
                    "",
                ])

        # Endpoints
        lines.extend([
            "## Endpoints",
            "",
        ])

        # Group endpoints by tag
        endpoints_by_tag = {}
        for endpoint in doc.endpoints:
            for tag in endpoint.tags:
                if tag not in endpoints_by_tag:
                    endpoints_by_tag[tag] = []
                endpoints_by_tag[tag].append(endpoint)

        for tag, endpoints in endpoints_by_tag.items():
            lines.extend([
                f"### {tag}",
                "",
            ])

            for endpoint in endpoints:
                lines.extend([
                    f"#### {endpoint.method} {endpoint.path}",
                    "",
                    f"**Summary:** {endpoint.summary}",
                    "",
                    endpoint.description,
                    "",
                ])

                if endpoint.parameters:
                    lines.extend([
                        "**Parameters:**",
                        "",
                    ])
                    for param in endpoint.parameters:
                        required = "required" if param['required'] else "optional"
                        lines.append(f"- `{param['name']}` ({param['type']}, {required}): {param['description']}")
                    lines.append("")

                if config.include_examples and endpoint.examples:
                    lines.extend([
                        "**Examples:**",
                        "",
                    ])
                    for example in endpoint.examples:
                        lines.extend([
                            f"**{example['language'].upper()}:**",
                            "```" + example['language'],
                            example['code'],
                            "```",
                            "",
                        ])

        return '\n'.join(lines)

    def _generate_html(self, doc: APIDocumentation, config: DocGenerationConfig) -> str:
        """Generate HTML documentation"""
        markdown_content = self._generate_markdown(doc, config)
        html_content = markdown.markdown(markdown_content, extensions=['fenced_code', 'tables'])

        template = f'''
<!DOCTYPE html>
<html>
<head>
    <title>{doc.title} - API Documentation</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: {"#1a1a1a" if config.dark_mode else "#ffffff"};
            color: {"#ffffff" if config.dark_mode else "#333333"};
        }}
        code {{
            background-color: {"#2d2d2d" if config.dark_mode else "#f4f4f4"};
            padding: 2px 6px;
            border-radius: 3px;
        }}
        pre {{
            background-color: {"#2d2d2d" if config.dark_mode else "#f4f4f4"};
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        h1, h2, h3, h4 {{
            color: {"#4a9eff" if config.dark_mode else "#0066cc"};
        }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>
'''
        return template

    def _generate_openapi(self, doc: APIDocumentation) -> str:
        """Generate OpenAPI specification"""
        spec = {
            "openapi": "3.0.0",
            "info": {
                "title": doc.title,
                "version": doc.version,
                "description": doc.description
            },
            "servers": [
                {"url": doc.base_url}
            ],
            "paths": {},
            "components": {
                "schemas": doc.models,
                "securitySchemes": {
                    "bearerAuth": {
                        "type": "http",
                        "scheme": "bearer",
                        "bearerFormat": "JWT"
                    }
                }
            },
            "security": [
                {"bearerAuth": []}
            ]
        }

        # Add endpoints
        for endpoint in doc.endpoints:
            if endpoint.path not in spec["paths"]:
                spec["paths"][endpoint.path] = {}

            spec["paths"][endpoint.path][endpoint.method.lower()] = {
                "summary": endpoint.summary,
                "description": endpoint.description,
                "tags": endpoint.tags,
                "parameters": endpoint.parameters,
                "responses": endpoint.responses
            }

        return json.dumps(spec, indent=2)

    async def generate_interactive_playground(
        self,
        doc: APIDocumentation,
        output_path: Path
    ) -> Dict[str, Any]:
        """Generate interactive API playground (HTML + JavaScript)"""
        html_content = self._generate_playground_html(doc)

        output_file = output_path / "api-playground.html"
        output_file.write_text(html_content)

        return {
            "success": True,
            "output_file": str(output_file),
            "endpoints": len(doc.endpoints)
        }

    def _generate_playground_html(self, doc: APIDocumentation) -> str:
        """Generate interactive playground HTML"""
        return f'''
<!DOCTYPE html>
<html>
<head>
    <title>{doc.title} - API Playground</title>
    <script src="https://unpkg.com/swagger-ui-dist@4/swagger-ui-bundle.js"></script>
    <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@4/swagger-ui.css" />
</head>
<body>
    <div id="swagger-ui"></div>
    <script>
        window.onload = function() {{
            SwaggerUIBundle({{
                url: './openapi.json',
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIBundle.SwaggerUIStandalonePreset
                ]
            }});
        }};
    </script>
</body>
</html>
'''
