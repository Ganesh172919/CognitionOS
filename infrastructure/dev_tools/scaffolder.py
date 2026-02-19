"""Code scaffolding system"""
from typing import Dict, List
from enum import Enum


class ProjectType(str, Enum):
    FASTAPI = "fastapi"
    CLI = "cli"


class CodeScaffolder:
    """Code scaffolding generator"""
    
    async def create_project(
        self,
        name: str,
        project_type: ProjectType
    ) -> Dict[str, str]:
        """Generate project scaffold"""
        return {
            "main.py": f"# {name} project",
            "README.md": f"# {name}"
        }
