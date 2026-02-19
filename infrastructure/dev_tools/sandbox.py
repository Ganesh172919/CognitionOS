"""Developer sandbox environment"""
from typing import Dict
from dataclasses import dataclass


@dataclass
class SandboxEnvironment:
    id: str
    name: str
    working_dir: str


class DeveloperSandbox:
    """Isolated developer sandbox"""
    
    def __init__(self):
        self.sandboxes: Dict[str, SandboxEnvironment] = {}
    
    async def create_sandbox(self, name: str) -> SandboxEnvironment:
        """Create sandbox"""
        sandbox = SandboxEnvironment(
            id=f"sandbox_{name}",
            name=name,
            working_dir="/tmp"
        )
        self.sandboxes[sandbox.id] = sandbox
        return sandbox
