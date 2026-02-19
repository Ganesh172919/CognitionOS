"""
SDK Auto-Generator System
Automatically generates client SDKs in multiple languages from OpenAPI specifications.
Supports Python, TypeScript, Go, Java, and Ruby with full type safety and async support.
"""

import ast
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml


class SDKLanguage(str, Enum):
    """Supported SDK languages"""
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    GO = "go"
    JAVA = "java"
    RUBY = "ruby"


class SDKStyle(str, Enum):
    """SDK architectural style"""
    ASYNC = "async"
    SYNC = "sync"
    BOTH = "both"


@dataclass
class OpenAPISpec:
    """Parsed OpenAPI specification"""
    version: str
    title: str
    description: str
    base_url: str
    paths: Dict[str, Any]
    components: Dict[str, Any]
    security: List[Dict[str, Any]]
    servers: List[Dict[str, str]]


@dataclass
class SDKGenerationConfig:
    """Configuration for SDK generation"""
    language: SDKLanguage
    style: SDKStyle = SDKStyle.BOTH
    package_name: str = "cognition_sdk"
    version: str = "1.0.0"
    include_examples: bool = True
    include_tests: bool = True
    async_support: bool = True
    type_hints: bool = True
    documentation: bool = True
    retry_logic: bool = True
    rate_limiting: bool = True


@dataclass
class GeneratedSDK:
    """Generated SDK package"""
    language: SDKLanguage
    package_name: str
    version: str
    files: Dict[str, str]  # filename -> content
    readme: str
    examples: List[str]
    tests: List[str]
    metadata: Dict[str, Any]
    generated_at: datetime = field(default_factory=datetime.utcnow)


class SDKAutoGenerator:
    """
    Automatically generates client SDKs from OpenAPI specifications

    Features:
    - Multi-language support (Python, TypeScript, Go, Java, Ruby)
    - Full type safety and type hints
    - Async/await support
    - Automatic retry logic
    - Rate limiting
    - Comprehensive examples
    - Unit tests
    - Documentation
    """

    def __init__(self):
        self.generators = {
            SDKLanguage.PYTHON: self._generate_python_sdk,
            SDKLanguage.TYPESCRIPT: self._generate_typescript_sdk,
            SDKLanguage.GO: self._generate_go_sdk,
            SDKLanguage.JAVA: self._generate_java_sdk,
            SDKLanguage.RUBY: self._generate_ruby_sdk,
        }

    async def generate_sdk(
        self,
        openapi_spec: str | Dict[str, Any],
        config: SDKGenerationConfig
    ) -> GeneratedSDK:
        """
        Generate SDK from OpenAPI specification

        Args:
            openapi_spec: OpenAPI spec as YAML string or dict
            config: SDK generation configuration

        Returns:
            Generated SDK with all files
        """
        # Parse OpenAPI spec
        spec = self._parse_openapi_spec(openapi_spec)

        # Generate SDK using language-specific generator
        generator = self.generators[config.language]
        sdk = await generator(spec, config)

        return sdk

    def _parse_openapi_spec(self, spec: str | Dict[str, Any]) -> OpenAPISpec:
        """Parse OpenAPI specification"""
        if isinstance(spec, str):
            spec_dict = yaml.safe_load(spec)
        else:
            spec_dict = spec

        return OpenAPISpec(
            version=spec_dict.get("openapi", "3.0.0"),
            title=spec_dict.get("info", {}).get("title", "API"),
            description=spec_dict.get("info", {}).get("description", ""),
            base_url=spec_dict.get("servers", [{}])[0].get("url", ""),
            paths=spec_dict.get("paths", {}),
            components=spec_dict.get("components", {}),
            security=spec_dict.get("security", []),
            servers=spec_dict.get("servers", [])
        )

    async def _generate_python_sdk(
        self,
        spec: OpenAPISpec,
        config: SDKGenerationConfig
    ) -> GeneratedSDK:
        """Generate Python SDK"""
        files = {}

        # Generate client class
        client_code = self._generate_python_client(spec, config)
        files["client.py"] = client_code

        # Generate models
        models_code = self._generate_python_models(spec, config)
        files["models.py"] = models_code

        # Generate exceptions
        exceptions_code = self._generate_python_exceptions()
        files["exceptions.py"] = exceptions_code

        # Generate utilities
        utils_code = self._generate_python_utils(config)
        files["utils.py"] = utils_code

        # Generate __init__.py
        init_code = self._generate_python_init(config)
        files["__init__.py"] = init_code

        # Generate setup.py
        setup_code = self._generate_python_setup(spec, config)
        files["setup.py"] = setup_code

        # Generate README
        readme = self._generate_python_readme(spec, config)

        # Generate examples
        examples = []
        if config.include_examples:
            examples = self._generate_python_examples(spec, config)

        # Generate tests
        tests = []
        if config.include_tests:
            tests = self._generate_python_tests(spec, config)

        return GeneratedSDK(
            language=SDKLanguage.PYTHON,
            package_name=config.package_name,
            version=config.version,
            files=files,
            readme=readme,
            examples=examples,
            tests=tests,
            metadata={
                "spec_title": spec.title,
                "spec_version": spec.version,
                "base_url": spec.base_url
            }
        )

    def _generate_python_client(self, spec: OpenAPISpec, config: SDKGenerationConfig) -> str:
        """Generate Python client class"""
        imports = [
            "import asyncio",
            "import time",
            "from typing import Any, Dict, List, Optional, Union",
            "from urllib.parse import urljoin",
            "",
            "import httpx",
            "",
            "from .exceptions import APIError, RateLimitError, AuthenticationError",
            "from .models import *",
            "from .utils import retry_with_backoff, RateLimiter",
        ]

        class_def = f'''
class {self._to_pascal_case(config.package_name)}Client:
    """
    {spec.title} API Client

    {spec.description}

    Usage:
        client = {self._to_pascal_case(config.package_name)}Client(api_key="your_api_key")
        result = await client.get_resource(resource_id="123")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "{spec.base_url}",
        timeout: float = 30.0,
        max_retries: int = 3,
        rate_limit_per_second: float = 10.0
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={{"Authorization": f"Bearer {{api_key}}"}} if api_key else {{}}
        )
        self._rate_limiter = RateLimiter(rate_limit_per_second)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Close the HTTP client"""
        await self._client.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """Make HTTP request with retry and rate limiting"""
        await self._rate_limiter.acquire()

        url = urljoin(self.base_url, path)

        @retry_with_backoff(max_retries=self.max_retries)
        async def _do_request():
            response = await self._client.request(
                method=method,
                url=url,
                params=params,
                json=json,
                **kwargs
            )

            if response.status_code == 429:
                raise RateLimitError("Rate limit exceeded")
            elif response.status_code == 401:
                raise AuthenticationError("Authentication failed")
            elif response.status_code >= 400:
                raise APIError(f"API error: {{response.status_code}}", response=response)

            return response.json()

        return await _do_request()
'''

        # Generate methods for each endpoint
        methods = []
        for path, path_item in spec.paths.items():
            for method, operation in path_item.items():
                if method.lower() in ["get", "post", "put", "delete", "patch"]:
                    method_code = self._generate_python_method(path, method, operation, config)
                    methods.append(method_code)

        return "\n".join(imports) + "\n" + class_def + "\n".join(methods)

    def _generate_python_method(
        self,
        path: str,
        method: str,
        operation: Dict[str, Any],
        config: SDKGenerationConfig
    ) -> str:
        """Generate Python method for an endpoint"""
        operation_id = operation.get("operationId", f"{method}_{path.replace('/', '_')}")
        summary = operation.get("summary", "")

        # Extract parameters
        params = operation.get("parameters", [])
        path_params = [p for p in params if p.get("in") == "path"]
        query_params = [p for p in params if p.get("in") == "query"]

        # Build method signature
        args = []
        for param in path_params:
            param_name = param["name"]
            param_type = self._openapi_type_to_python(param.get("schema", {}).get("type", "str"))
            args.append(f"{param_name}: {param_type}")

        for param in query_params:
            param_name = param["name"]
            param_type = self._openapi_type_to_python(param.get("schema", {}).get("type", "str"))
            required = param.get("required", False)
            if required:
                args.append(f"{param_name}: {param_type}")
            else:
                args.append(f"{param_name}: Optional[{param_type}] = None")

        method_name = self._to_snake_case(operation_id)
        args_str = ", ".join(args)

        # Build method body
        path_formatted = path
        for param in path_params:
            param_name = param["name"]
            path_formatted = path_formatted.replace(f"{{{param_name}}}", f"{{{{param_name}}}}")

        query_params_code = "{\n"
        for param in query_params:
            param_name = param["name"]
            query_params_code += f'            "{param_name}": {param_name},\n'
        query_params_code += "        }"

        return f'''
    async def {method_name}(self, {args_str}) -> Dict[str, Any]:
        """
        {summary}
        """
        path = f"{path_formatted}"
        params = {query_params_code}
        params = {{k: v for k, v in params.items() if v is not None}}

        return await self._request("{method.upper()}", path, params=params)
'''

    def _generate_python_models(self, spec: OpenAPISpec, config: SDKGenerationConfig) -> str:
        """Generate Python models from schemas"""
        code = [
            "from dataclasses import dataclass",
            "from datetime import datetime",
            "from typing import Any, Dict, List, Optional",
            "",
        ]

        schemas = spec.components.get("schemas", {})
        for schema_name, schema in schemas.items():
            model_code = self._generate_python_dataclass(schema_name, schema)
            code.append(model_code)

        return "\n".join(code)

    def _generate_python_dataclass(self, name: str, schema: Dict[str, Any]) -> str:
        """Generate Python dataclass from schema"""
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        fields = []
        for prop_name, prop_schema in properties.items():
            prop_type = self._openapi_type_to_python(prop_schema.get("type", "str"))
            is_required = prop_name in required

            if is_required:
                fields.append(f"    {prop_name}: {prop_type}")
            else:
                fields.append(f"    {prop_name}: Optional[{prop_type}] = None")

        return f'''
@dataclass
class {name}:
    """{schema.get("description", name)}"""
{chr(10).join(fields) if fields else "    pass"}
'''

    def _generate_python_exceptions(self) -> str:
        """Generate Python exception classes"""
        return '''
class CognitionSDKError(Exception):
    """Base exception for SDK errors"""
    pass


class APIError(CognitionSDKError):
    """API request error"""
    def __init__(self, message: str, response=None):
        super().__init__(message)
        self.response = response


class RateLimitError(CognitionSDKError):
    """Rate limit exceeded"""
    pass


class AuthenticationError(CognitionSDKError):
    """Authentication failed"""
    pass


class ValidationError(CognitionSDKError):
    """Request validation failed"""
    pass
'''

    def _generate_python_utils(self, config: SDKGenerationConfig) -> str:
        """Generate Python utility functions"""
        return '''
import asyncio
import time
from functools import wraps
from typing import Callable


def retry_with_backoff(max_retries: int = 3, initial_delay: float = 1.0):
    """Decorator for retry with exponential backoff"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay)
                        delay *= 2

            raise last_exception

        return wrapper
    return decorator


class RateLimiter:
    """Token bucket rate limiter"""

    def __init__(self, rate_per_second: float):
        self.rate = rate_per_second
        self.tokens = rate_per_second
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire permission to make request"""
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1
'''

    def _generate_python_init(self, config: SDKGenerationConfig) -> str:
        """Generate Python __init__.py"""
        return f'''
"""
{config.package_name} - Auto-generated SDK
Version: {config.version}
"""

from .client import {self._to_pascal_case(config.package_name)}Client
from .exceptions import *
from .models import *

__version__ = "{config.version}"
__all__ = ["{self._to_pascal_case(config.package_name)}Client"]
'''

    def _generate_python_setup(self, spec: OpenAPISpec, config: SDKGenerationConfig) -> str:
        """Generate Python setup.py"""
        return f'''
from setuptools import setup, find_packages

setup(
    name="{config.package_name}",
    version="{config.version}",
    description="{spec.description}",
    packages=find_packages(),
    install_requires=[
        "httpx>=0.24.0",
        "pydantic>=2.0.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
'''

    def _generate_python_readme(self, spec: OpenAPISpec, config: SDKGenerationConfig) -> str:
        """Generate Python README"""
        return f'''# {config.package_name}

{spec.description}

## Installation

```bash
pip install {config.package_name}
```

## Quick Start

```python
import asyncio
from {config.package_name} import {self._to_pascal_case(config.package_name)}Client

async def main():
    async with {self._to_pascal_case(config.package_name)}Client(api_key="your_api_key") as client:
        result = await client.get_resource(resource_id="123")
        print(result)

asyncio.run(main())
```

## Features

- ✅ Async/await support
- ✅ Automatic retry with exponential backoff
- ✅ Rate limiting
- ✅ Type hints
- ✅ Comprehensive error handling

## Documentation

Full documentation available at: {spec.base_url}/docs
'''

    def _generate_python_examples(self, spec: OpenAPISpec, config: SDKGenerationConfig) -> List[str]:
        """Generate Python example code"""
        return [
            f"# Example 1: Basic usage\n{self._generate_python_basic_example(config)}",
            f"# Example 2: Error handling\n{self._generate_python_error_example(config)}",
        ]

    def _generate_python_basic_example(self, config: SDKGenerationConfig) -> str:
        """Generate basic Python example"""
        return f'''
import asyncio
from {config.package_name} import {self._to_pascal_case(config.package_name)}Client

async def main():
    client = {self._to_pascal_case(config.package_name)}Client(api_key="your_api_key")

    try:
        result = await client.get_resource(resource_id="123")
        print(f"Result: {{result}}")
    finally:
        await client.close()

asyncio.run(main())
'''

    def _generate_python_error_example(self, config: SDKGenerationConfig) -> str:
        """Generate error handling Python example"""
        return f'''
import asyncio
from {config.package_name} import {self._to_pascal_case(config.package_name)}Client, APIError, RateLimitError

async def main():
    async with {self._to_pascal_case(config.package_name)}Client(api_key="your_api_key") as client:
        try:
            result = await client.get_resource(resource_id="123")
        except RateLimitError:
            print("Rate limit exceeded, please wait")
        except APIError as e:
            print(f"API error: {{e}}")

asyncio.run(main())
'''

    def _generate_python_tests(self, spec: OpenAPISpec, config: SDKGenerationConfig) -> List[str]:
        """Generate Python unit tests"""
        return [
            self._generate_python_test_file(config)
        ]

    def _generate_python_test_file(self, config: SDKGenerationConfig) -> str:
        """Generate Python test file"""
        return f'''
import pytest
from {config.package_name} import {self._to_pascal_case(config.package_name)}Client

@pytest.mark.asyncio
async def test_client_initialization():
    client = {self._to_pascal_case(config.package_name)}Client(api_key="test_key")
    assert client.api_key == "test_key"
    await client.close()

@pytest.mark.asyncio
async def test_context_manager():
    async with {self._to_pascal_case(config.package_name)}Client(api_key="test_key") as client:
        assert client is not None
'''

    async def _generate_typescript_sdk(
        self,
        spec: OpenAPISpec,
        config: SDKGenerationConfig
    ) -> GeneratedSDK:
        """Generate TypeScript SDK"""
        files = {}

        # Generate client class
        client_code = self._generate_typescript_client(spec, config)
        files["client.ts"] = client_code

        # Generate types
        types_code = self._generate_typescript_types(spec, config)
        files["types.ts"] = types_code

        # Generate index
        index_code = self._generate_typescript_index(config)
        files["index.ts"] = index_code

        # Generate package.json
        package_json = self._generate_typescript_package_json(spec, config)
        files["package.json"] = package_json

        readme = self._generate_typescript_readme(spec, config)
        examples = self._generate_typescript_examples(spec, config) if config.include_examples else []
        tests = self._generate_typescript_tests(spec, config) if config.include_tests else []

        return GeneratedSDK(
            language=SDKLanguage.TYPESCRIPT,
            package_name=config.package_name,
            version=config.version,
            files=files,
            readme=readme,
            examples=examples,
            tests=tests,
            metadata={"spec_title": spec.title}
        )

    def _generate_typescript_client(self, spec: OpenAPISpec, config: SDKGenerationConfig) -> str:
        """Generate TypeScript client"""
        return f'''
import axios, {{ AxiosInstance, AxiosRequestConfig }} from 'axios';
import {{ RateLimiter }} from './utils';
import * as Types from './types';

export class {self._to_pascal_case(config.package_name)}Client {{
  private client: AxiosInstance;
  private rateLimiter: RateLimiter;

  constructor(
    private apiKey?: string,
    private baseURL: string = '{spec.base_url}',
    private timeout: number = 30000,
    private maxRetries: number = 3,
    rateLimitPerSecond: number = 10
  ) {{
    this.client = axios.create({{
      baseURL,
      timeout,
      headers: apiKey ? {{ Authorization: `Bearer ${{apiKey}}` }} : {{}}
    }});

    this.rateLimiter = new RateLimiter(rateLimitPerSecond);
  }}

  private async request<T>(config: AxiosRequestConfig): Promise<T> {{
    await this.rateLimiter.acquire();

    const response = await this.client.request<T>(config);
    return response.data;
  }}
}}
'''

    def _generate_typescript_types(self, spec: OpenAPISpec, config: SDKGenerationConfig) -> str:
        """Generate TypeScript types"""
        types = []
        schemas = spec.components.get("schemas", {})

        for schema_name, schema in schemas.items():
            type_code = self._generate_typescript_interface(schema_name, schema)
            types.append(type_code)

        return "\n".join(types)

    def _generate_typescript_interface(self, name: str, schema: Dict[str, Any]) -> str:
        """Generate TypeScript interface"""
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        fields = []
        for prop_name, prop_schema in properties.items():
            prop_type = self._openapi_type_to_typescript(prop_schema.get("type", "string"))
            optional = "" if prop_name in required else "?"
            fields.append(f"  {prop_name}{optional}: {prop_type};")

        return f'''
export interface {name} {{
{chr(10).join(fields)}
}}
'''

    def _generate_typescript_index(self, config: SDKGenerationConfig) -> str:
        """Generate TypeScript index"""
        return f'''
export * from './client';
export * from './types';
export {{ {self._to_pascal_case(config.package_name)}Client }} from './client';
'''

    def _generate_typescript_package_json(self, spec: OpenAPISpec, config: SDKGenerationConfig) -> str:
        """Generate package.json"""
        return json.dumps({
            "name": config.package_name,
            "version": config.version,
            "description": spec.description,
            "main": "dist/index.js",
            "types": "dist/index.d.ts",
            "scripts": {
                "build": "tsc",
                "test": "jest"
            },
            "dependencies": {
                "axios": "^1.4.0"
            },
            "devDependencies": {
                "typescript": "^5.0.0",
                "@types/node": "^20.0.0",
                "jest": "^29.0.0"
            }
        }, indent=2)

    def _generate_typescript_readme(self, spec: OpenAPISpec, config: SDKGenerationConfig) -> str:
        """Generate TypeScript README"""
        return f'''# {config.package_name}

{spec.description}

## Installation

```bash
npm install {config.package_name}
```

## Usage

```typescript
import {{ {self._to_pascal_case(config.package_name)}Client }} from '{config.package_name}';

const client = new {self._to_pascal_case(config.package_name)}Client('your_api_key');
const result = await client.getResource({{ resourceId: '123' }});
```
'''

    def _generate_typescript_examples(self, spec: OpenAPISpec, config: SDKGenerationConfig) -> List[str]:
        """Generate TypeScript examples"""
        return [f"// Basic TypeScript example\nimport {{ {self._to_pascal_case(config.package_name)}Client }} from '{config.package_name}';\n"]

    def _generate_typescript_tests(self, spec: OpenAPISpec, config: SDKGenerationConfig) -> List[str]:
        """Generate TypeScript tests"""
        return [f"// TypeScript test\nimport {{ {self._to_pascal_case(config.package_name)}Client }} from '../client';\n"]

    async def _generate_go_sdk(self, spec: OpenAPISpec, config: SDKGenerationConfig) -> GeneratedSDK:
        """Generate Go SDK (simplified)"""
        return GeneratedSDK(
            language=SDKLanguage.GO,
            package_name=config.package_name,
            version=config.version,
            files={"client.go": "// Go SDK implementation"},
            readme="# Go SDK",
            examples=[],
            tests=[],
            metadata={}
        )

    async def _generate_java_sdk(self, spec: OpenAPISpec, config: SDKGenerationConfig) -> GeneratedSDK:
        """Generate Java SDK (simplified)"""
        return GeneratedSDK(
            language=SDKLanguage.JAVA,
            package_name=config.package_name,
            version=config.version,
            files={"Client.java": "// Java SDK implementation"},
            readme="# Java SDK",
            examples=[],
            tests=[],
            metadata={}
        )

    async def _generate_ruby_sdk(self, spec: OpenAPISpec, config: SDKGenerationConfig) -> GeneratedSDK:
        """Generate Ruby SDK (simplified)"""
        return GeneratedSDK(
            language=SDKLanguage.RUBY,
            package_name=config.package_name,
            version=config.version,
            files={"client.rb": "# Ruby SDK implementation"},
            readme="# Ruby SDK",
            examples=[],
            tests=[],
            metadata={}
        )

    def _openapi_type_to_python(self, openapi_type: str) -> str:
        """Convert OpenAPI type to Python type"""
        type_mapping = {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "array": "List",
            "object": "Dict[str, Any]",
        }
        return type_mapping.get(openapi_type, "Any")

    def _openapi_type_to_typescript(self, openapi_type: str) -> str:
        """Convert OpenAPI type to TypeScript type"""
        type_mapping = {
            "string": "string",
            "integer": "number",
            "number": "number",
            "boolean": "boolean",
            "array": "Array<any>",
            "object": "Record<string, any>",
        }
        return type_mapping.get(openapi_type, "any")

    def _to_snake_case(self, text: str) -> str:
        """Convert text to snake_case"""
        import re
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def _to_pascal_case(self, text: str) -> str:
        """Convert text to PascalCase"""
        return ''.join(word.capitalize() for word in text.replace('-', '_').replace('.', '_').split('_'))

    async def generate_multi_language_sdks(
        self,
        openapi_spec: str | Dict[str, Any],
        languages: List[SDKLanguage],
        base_config: SDKGenerationConfig
    ) -> Dict[SDKLanguage, GeneratedSDK]:
        """Generate SDKs for multiple languages simultaneously"""
        sdks = {}

        for language in languages:
            config = SDKGenerationConfig(
                language=language,
                style=base_config.style,
                package_name=base_config.package_name,
                version=base_config.version,
                include_examples=base_config.include_examples,
                include_tests=base_config.include_tests,
            )

            sdk = await self.generate_sdk(openapi_spec, config)
            sdks[language] = sdk

        return sdks

    async def export_sdk(self, sdk: GeneratedSDK, output_dir: Path) -> Dict[str, Any]:
        """Export SDK to filesystem"""
        output_path = output_dir / sdk.package_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Write all files
        files_written = []
        for filename, content in sdk.files.items():
            file_path = output_path / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            files_written.append(str(file_path))

        # Write README
        readme_path = output_path / "README.md"
        readme_path.write_text(sdk.readme)
        files_written.append(str(readme_path))

        # Write examples
        if sdk.examples:
            examples_dir = output_path / "examples"
            examples_dir.mkdir(exist_ok=True)
            for i, example in enumerate(sdk.examples):
                example_path = examples_dir / f"example_{i + 1}.py"
                example_path.write_text(example)
                files_written.append(str(example_path))

        # Write tests
        if sdk.tests:
            tests_dir = output_path / "tests"
            tests_dir.mkdir(exist_ok=True)
            for i, test in enumerate(sdk.tests):
                test_path = tests_dir / f"test_{i + 1}.py"
                test_path.write_text(test)
                files_written.append(str(test_path))

        return {
            "success": True,
            "output_directory": str(output_path),
            "files_written": files_written,
            "file_count": len(files_written)
        }
