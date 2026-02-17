"""
CognitionOS CLI - Management tool for tenants, subscriptions, and plugins.

Usage:
    cognition-cli tenant list
    cognition-cli tenant create <name> <slug>
    cognition-cli subscription show <tenant-slug>
    cognition-cli subscription upgrade <tenant-slug> <tier>
    cognition-cli plugin list
    cognition-cli plugin install <plugin-id> <tenant-slug>
"""

import asyncio
import sys
from typing import Optional
import httpx
import json


class CognitionCLI:
    """CLI client for CognitionOS API."""
    
    def __init__(self, api_url: str = "http://localhost:8100", api_key: Optional[str] = None):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.client = None
    
    def _get_headers(self, tenant_slug: Optional[str] = None):
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if tenant_slug:
            headers["X-Tenant-Slug"] = tenant_slug
        return headers
    
    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        return self
    
    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()
    
    # Tenant commands
    async def tenant_list(self):
        """List all tenants."""
        resp = await self.client.get(
            f"{self.api_url}/api/v3/tenants",
            headers=self._get_headers()
        )
        return resp.json()
    
    async def tenant_create(self, name: str, slug: str, owner_email: str, tier: str = "free"):
        """Create a new tenant."""
        data = {
            "name": name,
            "slug": slug,
            "owner_email": owner_email,
            "subscription_tier": tier
        }
        resp = await self.client.post(
            f"{self.api_url}/api/v3/tenants",
            headers=self._get_headers(),
            json=data
        )
        return resp.json()
    
    async def tenant_get(self, tenant_id: str):
        """Get tenant details."""
        resp = await self.client.get(
            f"{self.api_url}/api/v3/tenants/{tenant_id}",
            headers=self._get_headers()
        )
        return resp.json()
    
    # Subscription commands
    async def subscription_show(self, tenant_slug: str):
        """Show subscription for tenant."""
        resp = await self.client.get(
            f"{self.api_url}/api/v3/subscriptions/current",
            headers=self._get_headers(tenant_slug)
        )
        return resp.json()
    
    async def subscription_upgrade(self, tenant_slug: str, new_tier: str):
        """Upgrade subscription tier."""
        resp = await self.client.post(
            f"{self.api_url}/api/v3/subscriptions/upgrade",
            headers=self._get_headers(tenant_slug),
            json={"new_tier": new_tier}
        )
        return resp.json()
    
    async def subscription_usage(self, tenant_slug: str):
        """Get usage metrics."""
        resp = await self.client.get(
            f"{self.api_url}/api/v3/subscriptions/usage",
            headers=self._get_headers(tenant_slug)
        )
        return resp.json()
    
    # Plugin commands
    async def plugin_list(self, tenant_slug: Optional[str] = None):
        """List available plugins."""
        resp = await self.client.get(
            f"{self.api_url}/api/v3/plugins",
            headers=self._get_headers(tenant_slug)
        )
        return resp.json()
    
    async def plugin_get(self, plugin_id: str, tenant_slug: Optional[str] = None):
        """Get plugin details."""
        resp = await self.client.get(
            f"{self.api_url}/api/v3/plugins/{plugin_id}",
            headers=self._get_headers(tenant_slug)
        )
        return resp.json()
    
    async def plugin_install(self, plugin_id: str, tenant_slug: str):
        """Install plugin for tenant."""
        resp = await self.client.post(
            f"{self.api_url}/api/v3/plugins/{plugin_id}/install",
            headers=self._get_headers(tenant_slug)
        )
        return resp.json()


def print_json(data):
    """Pretty print JSON data."""
    print(json.dumps(data, indent=2))


async def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    command = sys.argv[1]
    subcommand = sys.argv[2] if len(sys.argv) > 2 else None
    
    # TODO: Load from config file
    api_url = "http://localhost:8100"
    api_key = None
    
    async with CognitionCLI(api_url, api_key) as cli:
        try:
            if command == "tenant":
                if subcommand == "list":
                    result = await cli.tenant_list()
                    print_json(result)
                elif subcommand == "create" and len(sys.argv) >= 5:
                    name = sys.argv[3]
                    slug = sys.argv[4]
                    email = sys.argv[5] if len(sys.argv) > 5 else f"admin@{slug}.com"
                    result = await cli.tenant_create(name, slug, email)
                    print_json(result)
                elif subcommand == "get" and len(sys.argv) >= 4:
                    tenant_id = sys.argv[3]
                    result = await cli.tenant_get(tenant_id)
                    print_json(result)
                else:
                    print("Usage: cognition-cli tenant [list|create|get]")
                    
            elif command == "subscription":
                if subcommand == "show" and len(sys.argv) >= 4:
                    tenant_slug = sys.argv[3]
                    result = await cli.subscription_show(tenant_slug)
                    print_json(result)
                elif subcommand == "upgrade" and len(sys.argv) >= 5:
                    tenant_slug = sys.argv[3]
                    new_tier = sys.argv[4]
                    result = await cli.subscription_upgrade(tenant_slug, new_tier)
                    print_json(result)
                elif subcommand == "usage" and len(sys.argv) >= 4:
                    tenant_slug = sys.argv[3]
                    result = await cli.subscription_usage(tenant_slug)
                    print_json(result)
                else:
                    print("Usage: cognition-cli subscription [show|upgrade|usage] <tenant-slug>")
                    
            elif command == "plugin":
                if subcommand == "list":
                    result = await cli.plugin_list()
                    print_json(result)
                elif subcommand == "get" and len(sys.argv) >= 4:
                    plugin_id = sys.argv[3]
                    result = await cli.plugin_get(plugin_id)
                    print_json(result)
                elif subcommand == "install" and len(sys.argv) >= 5:
                    plugin_id = sys.argv[3]
                    tenant_slug = sys.argv[4]
                    result = await cli.plugin_install(plugin_id, tenant_slug)
                    print_json(result)
                else:
                    print("Usage: cognition-cli plugin [list|get|install]")
                    
            else:
                print(__doc__)
                sys.exit(1)
                
        except httpx.HTTPError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
