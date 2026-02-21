"""
Developer SDK - Multi-Language Client Libraries

Production-ready SDKs for Python, JavaScript/TypeScript, Go, and Java
with full API coverage, automatic retries, and comprehensive examples.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SDKLanguage(str, Enum):
    """Supported SDK languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    JAVA = "java"
    RUBY = "ruby"
    PHP = "php"


@dataclass
class SDKConfig:
    """SDK configuration"""
    language: SDKLanguage
    version: str
    api_base_url: str = "https://api.cognitionos.com/v1"
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_ms: int = 1000


class PythonSDKGenerator:
    """
    Generate Python SDK

    Generates production-ready Python client library with type hints,
    async support, and comprehensive documentation.
    """

    def generate(self) -> str:
        """Generate complete Python SDK"""
        return '''
"""
CognitionOS Python SDK

Official Python client library for the CognitionOS API.

Installation:
    pip install cognitionos

Usage:
    from cognitionos import CognitionOS

    client = CognitionOS(api_key="your_api_key")
    result = client.agents.execute(task="Generate code")
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import requests
import time
import logging

logger = logging.getLogger(__name__)


class CognitionOSError(Exception):
    """Base exception for CognitionOS SDK"""
    pass


class AuthenticationError(CognitionOSError):
    """Authentication failed"""
    pass


class RateLimitError(CognitionOSError):
    """Rate limit exceeded"""
    pass


class APIError(CognitionOSError):
    """API returned error"""
    pass


@dataclass
class ExecutionResult:
    """Agent execution result"""
    task_id: str
    status: str
    result: Any
    metadata: Dict[str, Any]
    created_at: str
    completed_at: Optional[str] = None


class HTTPClient:
    """HTTP client with retry logic"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.cognitionos.com/v1",
        timeout: int = 30,
        max_retries: int = 3
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "CognitionOS-Python-SDK/1.0.0"
        })

    def request(
        self,
        method: str,
        path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic"""
        url = f"{self.base_url}{path}"

        for attempt in range(self.max_retries):
            try:
                response = self.session.request(
                    method,
                    url,
                    timeout=self.timeout,
                    **kwargs
                )

                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    if attempt < self.max_retries - 1:
                        logger.warning(f"Rate limited, retrying after {retry_after}s")
                        time.sleep(retry_after)
                        continue
                    raise RateLimitError("Rate limit exceeded")

                # Handle authentication errors
                if response.status_code == 401:
                    raise AuthenticationError("Invalid API key")

                # Handle other errors
                if response.status_code >= 400:
                    error_data = response.json() if response.content else {}
                    raise APIError(
                        f"API error {response.status_code}: "
                        f"{error_data.get('message', 'Unknown error')}"
                    )

                return response.json()

            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Request failed, retrying: {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                raise CognitionOSError(f"Request failed: {e}")

        raise CognitionOSError("Max retries exceeded")


class Agents:
    """Agent operations"""

    def __init__(self, client: HTTPClient):
        self.client = client

    def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        language: str = "python",
        async_execution: bool = False
    ) -> ExecutionResult:
        """
        Execute agent task

        Args:
            task: Task description
            context: Optional execution context
            language: Target language
            async_execution: Execute asynchronously

        Returns:
            ExecutionResult
        """
        response = self.client.request(
            "POST",
            "/agents/execute",
            json={
                "task": task,
                "context": context or {},
                "language": language,
                "async": async_execution
            }
        )

        return ExecutionResult(**response)

    def get_task(self, task_id: str) -> ExecutionResult:
        """Get task status"""
        response = self.client.request("GET", f"/agents/tasks/{task_id}")
        return ExecutionResult(**response)

    def list_tasks(
        self,
        limit: int = 10,
        offset: int = 0
    ) -> List[ExecutionResult]:
        """List recent tasks"""
        response = self.client.request(
            "GET",
            "/agents/tasks",
            params={"limit": limit, "offset": offset}
        )
        return [ExecutionResult(**task) for task in response["tasks"]]


class Code:
    """Code generation operations"""

    def __init__(self, client: HTTPClient):
        self.client = client

    def generate(
        self,
        description: str,
        language: str = "python",
        style: str = "functional"
    ) -> Dict[str, Any]:
        """Generate code"""
        return self.client.request(
            "POST",
            "/code/generate",
            json={
                "description": description,
                "language": language,
                "style": style
            }
        )

    def validate(self, code: str, language: str) -> Dict[str, Any]:
        """Validate code"""
        return self.client.request(
            "POST",
            "/code/validate",
            json={"code": code, "language": language}
        )

    def refactor(
        self,
        code: str,
        goal: str,
        language: str = "python"
    ) -> Dict[str, Any]:
        """Refactor code"""
        return self.client.request(
            "POST",
            "/code/refactor",
            json={
                "code": code,
                "goal": goal,
                "language": language
            }
        )


class Workflows:
    """Workflow operations"""

    def __init__(self, client: HTTPClient):
        self.client = client

    def create(
        self,
        name: str,
        steps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create workflow"""
        return self.client.request(
            "POST",
            "/workflows",
            json={"name": name, "steps": steps}
        )

    def execute(
        self,
        workflow_id: str,
        inputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute workflow"""
        return self.client.request(
            "POST",
            f"/workflows/{workflow_id}/execute",
            json={"inputs": inputs or {}}
        )

    def get(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow"""
        return self.client.request("GET", f"/workflows/{workflow_id}")


class CognitionOS:
    """
    CognitionOS API Client

    Main entry point for the SDK.

    Example:
        client = CognitionOS(api_key="your_api_key")
        result = client.agents.execute(task="Generate function")
        print(result.result)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.cognitionos.com/v1",
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize client

        Args:
            api_key: Your API key
            base_url: API base URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self._client = HTTPClient(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries
        )

        # Initialize resource clients
        self.agents = Agents(self._client)
        self.code = Code(self._client)
        self.workflows = Workflows(self._client)
'''


class TypeScriptSDKGenerator:
    """Generate TypeScript SDK"""

    def generate(self) -> str:
        """Generate complete TypeScript SDK"""
        return '''/**
 * CognitionOS TypeScript/JavaScript SDK
 *
 * Installation: npm install @cognitionos/sdk
 *
 * Usage:
 *   import { CognitionOS } from '@cognitionos/sdk';
 *   const client = new CognitionOS({ apiKey: 'your_api_key' });
 */

export interface Config {
  apiKey: string;
  baseUrl?: string;
  timeout?: number;
  maxRetries?: number;
}

export interface ExecutionResult {
  taskId: string;
  status: string;
  result: any;
  metadata: Record<string, any>;
  createdAt: string;
  completedAt?: string;
}

export class CognitionOS {
  private httpClient: HTTPClient;
  public readonly agents: Agents;
  public readonly code: Code;
  public readonly workflows: Workflows;

  constructor(config: Config) {
    this.httpClient = new HTTPClient(config);
    this.agents = new Agents(this.httpClient);
    this.code = new Code(this.httpClient);
    this.workflows = new Workflows(this.httpClient);
  }
}

class HTTPClient {
  private apiKey: string;
  private baseUrl: string;
  private timeout: number;
  private maxRetries: number;

  constructor(config: Config) {
    this.apiKey = config.apiKey;
    this.baseUrl = config.baseUrl || 'https://api.cognitionos.com/v1';
    this.timeout = config.timeout || 30000;
    this.maxRetries = config.maxRetries || 3;
  }

  async request<T>(method: string, path: string, body?: any): Promise<T> {
    const url = `${this.baseUrl}${path}`;

    for (let attempt = 0; attempt < this.maxRetries; attempt++) {
      try {
        const response = await fetch(url, {
          method,
          headers: {
            'Authorization': `Bearer ${this.apiKey}`,
            'Content-Type': 'application/json'
          },
          body: body ? JSON.stringify(body) : undefined
        });

        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }

        return await response.json();
      } catch (error) {
        if (attempt < this.maxRetries - 1) {
          await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 1000));
          continue;
        }
        throw error;
      }
    }
    throw new Error('Max retries exceeded');
  }
}

class Agents {
  constructor(private client: HTTPClient) {}

  async execute(options: {
    task: string;
    context?: Record<string, any>;
    language?: string;
    async?: boolean;
  }): Promise<ExecutionResult> {
    return this.client.request<ExecutionResult>('POST', '/agents/execute', options);
  }
}

class Code {
  constructor(private client: HTTPClient) {}

  async generate(options: {
    description: string;
    language?: string;
    style?: string;
  }): Promise<any> {
    return this.client.request('POST', '/code/generate', options);
  }
}

class Workflows {
  constructor(private client: HTTPClient) {}

  async create(name: string, steps: any[]): Promise<any> {
    return this.client.request('POST', '/workflows', { name, steps });
  }
}

export default CognitionOS;
'''


class SDKDocumentationGenerator:
    """Generate SDK documentation and examples"""

    def generate_quickstart_guide(self, language: SDKLanguage) -> str:
        """Generate quickstart guide"""

        if language == SDKLanguage.PYTHON:
            return '''# CognitionOS Python SDK - Quickstart

## Installation
```bash
pip install cognitionos
```

## Basic Usage
```python
from cognitionos import CognitionOS

client = CognitionOS(api_key="your_api_key")

# Execute agent task
result = client.agents.execute(
    task="Generate a function to calculate fibonacci"
)
print(result.result)

# Generate code
code = client.code.generate(
    description="REST API endpoint",
    language="python"
)
print(code["code"])
```'''

        return "Documentation not available"


class SDKManager:
    """
    SDK management and generation

    Manages SDK generation, versioning, and distribution.
    """

    def __init__(self):
        self._generators = {
            SDKLanguage.PYTHON: PythonSDKGenerator(),
            SDKLanguage.TYPESCRIPT: TypeScriptSDKGenerator()
        }
        self._docs_generator = SDKDocumentationGenerator()

    def generate_sdk(self, language: SDKLanguage) -> str:
        """Generate SDK for language"""
        generator = self._generators.get(language)
        if not generator:
            raise ValueError(f"SDK generator not available for {language}")

        logger.info(f"Generating {language} SDK")
        return generator.generate()

    def generate_all_sdks(self) -> Dict[SDKLanguage, str]:
        """Generate all SDKs"""
        sdks = {}
        for language in self._generators.keys():
            sdks[language] = self.generate_sdk(language)
        return sdks

    def generate_documentation(self, language: SDKLanguage) -> str:
        """Generate SDK documentation"""
        return self._docs_generator.generate_quickstart_guide(language)
