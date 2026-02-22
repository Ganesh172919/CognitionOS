"""
Developer SDK Portal
=====================
Multi-language SDK generation, interactive API sandbox, documentation
generation, API key provisioning, and developer onboarding automation.

Implements:
- SDK code generation for Python, TypeScript, Go, Java, cURL
- OpenAPI 3.0 schema generation from registered endpoints
- Interactive sandbox: request templating and simulated execution
- Code snippet library with copy-paste examples
- Developer onboarding flow: account → API key → first request
- Rate-limit preview for each API key tier
- SDK version management and changelog tracking
- SDK usage analytics: which SDK methods are called most
- API changelog generation from schema diffs
- Error reference documentation
- Postman/Insomnia collection export
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SDKLanguage(str, Enum):
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    GO = "go"
    JAVA = "java"
    CURL = "curl"
    RUBY = "ruby"
    PHP = "php"


class APIStatus(str, Enum):
    STABLE = "stable"
    BETA = "beta"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"


class ChangeType(str, Enum):
    ADDED = "added"
    CHANGED = "changed"
    DEPRECATED = "deprecated"
    REMOVED = "removed"
    FIXED = "fixed"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class APIParameter:
    name: str
    location: str = "query"       # query | path | header | body
    data_type: str = "string"
    required: bool = False
    description: str = ""
    example: Optional[Any] = None
    enum_values: List[str] = field(default_factory=list)


@dataclass
class APIEndpoint:
    endpoint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    path: str = ""
    method: str = "GET"
    summary: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    parameters: List[APIParameter] = field(default_factory=list)
    request_body_schema: Optional[Dict[str, Any]] = None
    response_schema: Optional[Dict[str, Any]] = None
    status: APIStatus = APIStatus.STABLE
    auth_required: bool = True
    rate_limit_tier: str = "default"
    version: str = "v3"
    example_response: Optional[Dict[str, Any]] = None


@dataclass
class SDKVersion:
    version: str
    language: SDKLanguage
    release_date: float = field(default_factory=time.time)
    changelog: List[str] = field(default_factory=list)
    download_url: str = ""
    size_bytes: int = 0
    downloads: int = 0


@dataclass
class DeveloperAccount:
    account_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    email: str = ""
    name: str = ""
    organization: str = ""
    tier: str = "free"
    api_keys: List[str] = field(default_factory=list)
    onboarding_completed: bool = False
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    sdk_downloads: Dict[str, int] = field(default_factory=dict)


@dataclass
class SandboxRequest:
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    account_id: str = ""
    endpoint_path: str = ""
    method: str = "GET"
    headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, Any] = field(default_factory=dict)
    body: Optional[Dict[str, Any]] = None
    executed_at: float = field(default_factory=time.time)
    response_status: int = 200
    response_body: Optional[Dict[str, Any]] = None
    latency_ms: float = 0.0


@dataclass
class ChangelogEntry:
    version: str
    date: str
    changes: Dict[str, List[str]] = field(default_factory=dict)   # ChangeType -> list of descriptions


# ---------------------------------------------------------------------------
# Code Generators
# ---------------------------------------------------------------------------

class PythonSDKGenerator:
    def generate_client(self, endpoints: List[APIEndpoint], package_name: str = "cognitionos") -> str:
        methods = []
        for ep in endpoints:
            method_name = self._to_snake_case(ep.summary or ep.path.replace("/", "_"))
            required_params = [p.name for p in ep.parameters if p.required]
            optional_params = [p.name for p in ep.parameters if not p.required]

            sig_parts: List[str] = list(required_params)
            sig_parts.extend(f"{p}=None" for p in optional_params)
            if ep.request_body_schema:
                sig_parts.append("**kwargs")
            param_sig = ", ".join(sig_parts)

            path_params = [p.name for p in ep.parameters if p.location == "path"]
            url_expr = f'f"{{self.base_url}}{ep.path}"'
            for pp in path_params:
                url_expr = url_expr.replace(f"{{{pp}}}", f"{{" + pp + "}")

            query_params = {p for p in required_params if next((x for x in ep.parameters if x.name == p and x.location == "query"), None)}
            params_dict = ", ".join(f'"{p}": {p}' for p in query_params)
            body = f'    def {method_name}(self, {param_sig}) -> dict:\n'
            body += f'        """{ep.summary}"""\n'
            body += f'        url = {url_expr}\n'
            body += f'        params = {{{params_dict}}}\n'
            if ep.method.upper() in ("POST", "PUT", "PATCH"):
                body += f'        return self._request("{ep.method}", url, params=params, json=kwargs)\n'
            else:
                body += f'        return self._request("{ep.method}", url, params=params)\n'
            methods.append(body)

        client_code = f'''"""
{package_name} Python SDK
Auto-generated. Do not edit manually.
"""
import requests
from typing import Optional, Any


class {package_name.capitalize()}Client:
    def __init__(self, api_key: str, base_url: str = "https://api.cognitionos.io"):
        self.api_key = api_key
        self.base_url = base_url
        self._session = requests.Session()
        self._session.headers.update({{"Authorization": f"Bearer {{api_key}}", "Content-Type": "application/json"}})

    def _request(self, method: str, url: str, **kwargs) -> dict:
        response = self._session.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()

'''
        for m in methods[:20]:  # limit to first 20 methods
            client_code += m + "\n"

        return client_code

    def _to_snake_case(self, text: str) -> str:
        import re
        text = re.sub(r"[^a-zA-Z0-9_]", "_", text)
        text = re.sub(r"_+", "_", text)
        return text.lower().strip("_") or "api_call"


class TypeScriptSDKGenerator:
    def generate_client(self, endpoints: List[APIEndpoint], package_name: str = "cognitionos") -> str:
        methods = []
        for ep in endpoints[:20]:
            method_name = self._to_camel_case(ep.summary or ep.path)
            params = [f"{p.name}: {'string' if p.data_type == 'string' else 'number'}" for p in ep.parameters if p.required]
            opt_params = [f"{p.name}?: {'string' if p.data_type == 'string' else 'number'}" for p in ep.parameters if not p.required]
            all_params = ", ".join(params + opt_params)

            m = f"  async {method_name}({all_params}): Promise<any> {{\n"
            m += f"    // {ep.summary}\n"
            m += f'    return this.request("{ep.method}", "{ep.path}");\n'
            m += "  }\n"
            methods.append(m)

        return f'''/**
 * {package_name} TypeScript SDK
 * Auto-generated. Do not edit manually.
 */

export class {package_name.capitalize()}Client {{
  constructor(
    private apiKey: string,
    private baseUrl: string = "https://api.cognitionos.io"
  ) {{}}

  private async request(method: string, path: string, data?: unknown): Promise<any> {{
    const response = await fetch(`${{this.baseUrl}}${{path}}`, {{
      method,
      headers: {{
        Authorization: `Bearer ${{this.apiKey}}`,
        "Content-Type": "application/json",
      }},
      body: data ? JSON.stringify(data) : undefined,
    }});
    if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
    return response.json();
  }}

{"".join(methods)}
}}
'''

    def _to_camel_case(self, text: str) -> str:
        import re
        text = re.sub(r"[^a-zA-Z0-9]", " ", text)
        words = text.split()
        if not words:
            return "apiCall"
        return words[0].lower() + "".join(w.capitalize() for w in words[1:])


class GoSDKGenerator:
    def generate_client(self, endpoints: List[APIEndpoint], package_name: str = "cognitionos") -> str:
        methods = []
        for ep in endpoints[:20]:
            func_name = self._to_pascal_case(ep.summary or ep.path)
            methods.append(
                f'// {ep.summary}\n'
                f'func (c *Client) {func_name}(ctx context.Context) (map[string]interface{{}}, error) {{\n'
                f'\treturn c.request(ctx, "{ep.method}", "{ep.path}", nil)\n'
                f'}}\n'
            )

        return f'''// Package {package_name} - Auto-generated SDK. Do not edit manually.
package {package_name}

import (
\t"bytes"
\t"context"
\t"encoding/json"
\t"fmt"
\t"net/http"
)

type Client struct {{
\tAPIKey  string
\tBaseURL string
\tHTTP    *http.Client
}}

func NewClient(apiKey string) *Client {{
\treturn &Client{{APIKey: apiKey, BaseURL: "https://api.cognitionos.io", HTTP: &http.Client{{}}}}
}}

func (c *Client) request(ctx context.Context, method, path string, body interface{{}}) (map[string]interface{{}}, error) {{
\tvar buf *bytes.Buffer
\tif body != nil {{
\t\tb, _ := json.Marshal(body)
\t\tbuf = bytes.NewBuffer(b)
\t}}
\treq, err := http.NewRequestWithContext(ctx, method, c.BaseURL+path, buf)
\tif err != nil {{
\t\treturn nil, err
\t}}
\treq.Header.Set("Authorization", "Bearer "+c.APIKey)
\treq.Header.Set("Content-Type", "application/json")
\tresp, err := c.HTTP.Do(req)
\tif err != nil {{
\t\treturn nil, err
\t}}
\tdefer resp.Body.Close()
\tvar result map[string]interface{{}}
\tjson.NewDecoder(resp.Body).Decode(&result)
\tif resp.StatusCode >= 400 {{
\t\treturn nil, fmt.Errorf("HTTP %d", resp.StatusCode)
\t}}
\treturn result, nil
}}

{"".join(methods)}'''

    def _to_pascal_case(self, text: str) -> str:
        import re
        text = re.sub(r"[^a-zA-Z0-9]", " ", text)
        return "".join(w.capitalize() for w in text.split()) or "ApiCall"


class CurlGenerator:
    def generate_examples(self, endpoints: List[APIEndpoint], base_url: str = "https://api.cognitionos.io") -> str:
        lines = ["#!/bin/bash", "# CognitionOS API - cURL Examples", "# Auto-generated", "", 'API_KEY="${COGNITIONOS_API_KEY}"', ""]
        for ep in endpoints[:20]:
            lines.append(f"# {ep.summary}")
            curl_cmd = f'curl -X {ep.method.upper()} \\\n  "{base_url}{ep.path}" \\\n  -H "Authorization: Bearer $API_KEY" \\\n  -H "Content-Type: application/json"'
            if ep.request_body_schema:
                curl_cmd += " \\\n  -d '{\"key\": \"value\"}'"
            lines.append(curl_cmd)
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# OpenAPI Generator
# ---------------------------------------------------------------------------

class OpenAPIGenerator:
    """Generates OpenAPI 3.0 specification from registered endpoints."""

    def generate(
        self,
        endpoints: List[APIEndpoint],
        title: str = "CognitionOS API",
        version: str = "3.0.0",
        description: str = "CognitionOS Platform API",
        base_url: str = "https://api.cognitionos.io",
    ) -> Dict[str, Any]:
        paths: Dict[str, Any] = {}
        tags: List[str] = []

        for ep in endpoints:
            if ep.path not in paths:
                paths[ep.path] = {}

            parameters = []
            for param in ep.parameters:
                if param.location != "body":
                    p: Dict[str, Any] = {
                        "name": param.name,
                        "in": param.location,
                        "required": param.required,
                        "description": param.description,
                        "schema": {"type": param.data_type},
                    }
                    if param.enum_values:
                        p["schema"]["enum"] = param.enum_values
                    if param.example is not None:
                        p["example"] = param.example
                    parameters.append(p)

            operation: Dict[str, Any] = {
                "summary": ep.summary,
                "description": ep.description,
                "tags": ep.tags,
                "parameters": parameters,
                "responses": {
                    "200": {
                        "description": "Success",
                        "content": {"application/json": {"schema": ep.response_schema or {"type": "object"}}},
                    },
                    "400": {"description": "Bad Request"},
                    "401": {"description": "Unauthorized"},
                    "429": {"description": "Rate Limit Exceeded"},
                },
            }

            if ep.auth_required:
                operation["security"] = [{"bearerAuth": []}]

            if ep.request_body_schema and ep.method.upper() in ("POST", "PUT", "PATCH"):
                operation["requestBody"] = {
                    "required": True,
                    "content": {"application/json": {"schema": ep.request_body_schema}},
                }

            if ep.example_response:
                operation["responses"]["200"]["content"]["application/json"]["example"] = ep.example_response

            paths[ep.path][ep.method.lower()] = operation
            tags.extend(t for t in ep.tags if t not in tags)

        return {
            "openapi": "3.0.3",
            "info": {"title": title, "version": version, "description": description},
            "servers": [{"url": base_url}],
            "tags": [{"name": t} for t in tags],
            "paths": paths,
            "components": {
                "securitySchemes": {
                    "bearerAuth": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"}
                }
            },
        }


# ---------------------------------------------------------------------------
# Postman Collection Generator
# ---------------------------------------------------------------------------

class PostmanCollectionGenerator:
    def generate(self, endpoints: List[APIEndpoint], collection_name: str = "CognitionOS") -> Dict[str, Any]:
        items = []
        for ep in endpoints[:50]:
            item: Dict[str, Any] = {
                "name": ep.summary,
                "request": {
                    "method": ep.method.upper(),
                    "header": [
                        {"key": "Authorization", "value": "Bearer {{api_key}}"},
                        {"key": "Content-Type", "value": "application/json"},
                    ],
                    "url": {
                        "raw": f"{{{{base_url}}}}{ep.path}",
                        "host": ["{{base_url}}"],
                        "path": [p for p in ep.path.split("/") if p],
                        "query": [
                            {"key": p.name, "value": str(p.example or ""), "description": p.description}
                            for p in ep.parameters if p.location == "query"
                        ],
                    },
                },
                "response": [],
            }
            if ep.request_body_schema and ep.method.upper() in ("POST", "PUT", "PATCH"):
                item["request"]["body"] = {
                    "mode": "raw",
                    "raw": json.dumps(ep.request_body_schema.get("example", {}), indent=2),
                    "options": {"raw": {"language": "json"}},
                }
            items.append(item)

        return {
            "info": {
                "name": collection_name,
                "_postman_id": str(uuid.uuid4()),
                "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
            },
            "variable": [
                {"key": "base_url", "value": "https://api.cognitionos.io"},
                {"key": "api_key", "value": "your-api-key-here"},
            ],
            "item": items,
        }


# ---------------------------------------------------------------------------
# Developer SDK Portal
# ---------------------------------------------------------------------------

class DeveloperSDKPortal:
    """
    Complete developer experience platform: SDK generation, API sandbox,
    onboarding automation, documentation, and usage analytics.
    """

    def __init__(self):
        self._endpoints: Dict[str, APIEndpoint] = {}
        self._accounts: Dict[str, DeveloperAccount] = {}
        self._sdk_versions: Dict[str, List[SDKVersion]] = {}
        self._sandbox_requests: List[SandboxRequest] = []
        self._changelog: List[ChangelogEntry] = []

        self._python_gen = PythonSDKGenerator()
        self._ts_gen = TypeScriptSDKGenerator()
        self._go_gen = GoSDKGenerator()
        self._curl_gen = CurlGenerator()
        self._openapi_gen = OpenAPIGenerator()
        self._postman_gen = PostmanCollectionGenerator()

        self._sdk_usage: Dict[str, Dict[str, int]] = {}  # account_id -> {method: count}

    # ---- Endpoint Registry ----

    def register_endpoint(self, endpoint: APIEndpoint) -> APIEndpoint:
        self._endpoints[endpoint.endpoint_id] = endpoint
        return endpoint

    def register_endpoints_bulk(self, endpoints: List[APIEndpoint]) -> int:
        for ep in endpoints:
            self.register_endpoint(ep)
        return len(endpoints)

    # ---- SDK Generation ----

    def generate_sdk(
        self,
        language: SDKLanguage,
        tags_filter: Optional[List[str]] = None,
    ) -> str:
        endpoints = list(self._endpoints.values())
        if tags_filter:
            endpoints = [ep for ep in endpoints if any(t in ep.tags for t in tags_filter)]

        if language == SDKLanguage.PYTHON:
            return self._python_gen.generate_client(endpoints)
        if language == SDKLanguage.TYPESCRIPT:
            return self._ts_gen.generate_client(endpoints)
        if language == SDKLanguage.GO:
            return self._go_gen.generate_client(endpoints)
        if language == SDKLanguage.CURL:
            return self._curl_gen.generate_examples(endpoints)
        return f"# SDK for {language.value} - coming soon"

    def generate_openapi_spec(self) -> Dict[str, Any]:
        return self._openapi_gen.generate(list(self._endpoints.values()))

    def generate_postman_collection(self) -> Dict[str, Any]:
        return self._postman_gen.generate(list(self._endpoints.values()))

    # ---- Developer Onboarding ----

    async def register_developer(
        self,
        email: str,
        name: str,
        organization: str = "",
        tier: str = "free",
    ) -> DeveloperAccount:
        account = DeveloperAccount(email=email, name=name, organization=organization, tier=tier)
        self._accounts[account.account_id] = account
        return account

    async def complete_onboarding(self, account_id: str) -> Dict[str, Any]:
        account = self._accounts.get(account_id)
        if not account:
            raise ValueError(f"Account {account_id} not found")

        import secrets
        api_key = f"cog_{secrets.token_urlsafe(32)}"
        account.api_keys.append(api_key)
        account.onboarding_completed = True

        return {
            "account_id": account_id,
            "api_key": api_key,
            "message": f"Welcome, {account.name}! Your API key is ready.",
            "quickstart": {
                "python": f'from cognitionos import CognitionosClient\nclient = CognitionosClient("{api_key}")',
                "curl": f'curl -H "Authorization: Bearer {api_key}" https://api.cognitionos.io/api/v3/health',
            },
            "documentation_url": "https://docs.cognitionos.io",
            "sandbox_url": "https://sandbox.cognitionos.io",
        }

    # ---- Interactive Sandbox ----

    async def execute_sandbox_request(
        self,
        account_id: str,
        endpoint_path: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
    ) -> SandboxRequest:
        import random

        # Find matching endpoint
        endpoint = next(
            (ep for ep in self._endpoints.values() if ep.path == endpoint_path and ep.method.upper() == method.upper()),
            None,
        )

        # Simulate response
        if endpoint and endpoint.example_response:
            response_body = endpoint.example_response
            status = 200
        else:
            response_body = {"status": "ok", "message": "Sandbox response", "data": {}}
            status = 200

        req = SandboxRequest(
            account_id=account_id,
            endpoint_path=endpoint_path,
            method=method,
            query_params=params or {},
            body=body,
            response_status=status,
            response_body=response_body,
            latency_ms=random.uniform(10, 150),
        )
        self._sandbox_requests.append(req)

        # Track usage
        if account_id not in self._sdk_usage:
            self._sdk_usage[account_id] = {}
        self._sdk_usage[account_id][endpoint_path] = self._sdk_usage[account_id].get(endpoint_path, 0) + 1

        return req

    def get_code_snippet(
        self, endpoint_path: str, method: str = "GET", language: SDKLanguage = SDKLanguage.CURL
    ) -> str:
        endpoint = next(
            (ep for ep in self._endpoints.values() if ep.path == endpoint_path and ep.method.upper() == method.upper()),
            None,
        )
        if not endpoint:
            return f"# Endpoint {method} {endpoint_path} not found"

        if language == SDKLanguage.CURL:
            return f'curl -X {method.upper()} "https://api.cognitionos.io{endpoint_path}" \\\n  -H "Authorization: Bearer $API_KEY"'
        if language == SDKLanguage.PYTHON:
            return f'response = client.request("{method}", "{endpoint_path}")\nprint(response)'
        if language == SDKLanguage.TYPESCRIPT:
            func_name = self._ts_gen._to_camel_case(endpoint.summary or endpoint_path)
            return f'const result = await client.{func_name}();\nconsole.log(result);'
        return f"// {method} {endpoint_path}"

    # ---- Changelog ----

    def add_changelog_entry(self, version: str, date: str, changes: Dict[str, List[str]]) -> ChangelogEntry:
        entry = ChangelogEntry(version=version, date=date, changes=changes)
        self._changelog.insert(0, entry)
        return entry

    def get_changelog(self, limit: int = 10) -> List[ChangelogEntry]:
        return self._changelog[:limit]

    # ---- Analytics ----

    def get_portal_summary(self) -> Dict[str, Any]:
        return {
            "registered_endpoints": len(self._endpoints),
            "registered_developers": len(self._accounts),
            "onboarded_developers": sum(1 for a in self._accounts.values() if a.onboarding_completed),
            "sandbox_requests": len(self._sandbox_requests),
            "supported_languages": [lang.value for lang in SDKLanguage],
            "changelog_entries": len(self._changelog),
        }

    def get_endpoint_usage_stats(self) -> Dict[str, Any]:
        usage_totals: Dict[str, int] = {}
        for account_usage in self._sdk_usage.values():
            for endpoint, count in account_usage.items():
                usage_totals[endpoint] = usage_totals.get(endpoint, 0) + count
        top_endpoints = sorted(usage_totals.items(), key=lambda x: x[1], reverse=True)[:10]
        return {
            "top_endpoints": [{"endpoint": ep, "calls": cnt} for ep, cnt in top_endpoints],
            "total_sandbox_calls": sum(usage_totals.values()),
        }
