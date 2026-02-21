"""
Context Manager - Codebase Context and Memory Management

Manages context about the codebase for intelligent code generation.
Maintains memory of previous decisions and patterns.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class CodebaseContext:
    """Context about the codebase"""
    project_name: str
    language: str
    framework: Optional[str] = None

    # File structure
    files: Dict[str, str] = field(default_factory=dict)  # path -> content
    file_tree: List[str] = field(default_factory=list)

    # Code patterns
    naming_conventions: Dict[str, str] = field(default_factory=dict)
    code_style: Dict[str, Any] = field(default_factory=dict)
    common_imports: List[str] = field(default_factory=list)

    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    dev_dependencies: List[str] = field(default_factory=list)

    # Architecture
    architecture_pattern: Optional[str] = None  # mvc, clean, layered, etc.
    components: List[str] = field(default_factory=list)

    # Metadata
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DecisionRecord:
    """Record of a design/implementation decision"""
    id: str
    timestamp: datetime
    decision: str
    rationale: str
    alternatives_considered: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)


@dataclass
class MemoryEntry:
    """Single memory entry"""
    id: str
    content: str
    entry_type: str  # decision, pattern, constraint, requirement
    timestamp: datetime
    relevance_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryStore:
    """
    Persistent memory store for agent decisions and learnings

    Stores and retrieves relevant context from past interactions.
    """

    def __init__(self, storage_backend: Optional[Any] = None):
        self.storage = storage_backend
        self._memory: List[MemoryEntry] = []
        self._decisions: Dict[str, DecisionRecord] = {}

    async def store(
        self,
        content: str,
        entry_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryEntry:
        """Store a memory entry"""

        entry = MemoryEntry(
            id=f"mem_{len(self._memory)}_{int(datetime.utcnow().timestamp())}",
            content=content,
            entry_type=entry_type,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )

        self._memory.append(entry)

        if self.storage:
            await self.storage.save_memory(entry)

        logger.info(f"Stored memory: {entry_type} - {content[:50]}...")
        return entry

    async def store_decision(
        self,
        decision: str,
        rationale: str,
        alternatives: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> DecisionRecord:
        """Store a design decision"""

        record = DecisionRecord(
            id=f"decision_{len(self._decisions)}",
            timestamp=datetime.utcnow(),
            decision=decision,
            rationale=rationale,
            alternatives_considered=alternatives or [],
            context=context or {}
        )

        self._decisions[record.id] = record

        if self.storage:
            await self.storage.save_decision(record)

        logger.info(f"Recorded decision: {decision[:50]}...")
        return record

    async def retrieve_relevant(
        self,
        query: str,
        entry_type: Optional[str] = None,
        limit: int = 5
    ) -> List[MemoryEntry]:
        """Retrieve relevant memories"""

        if self.storage and hasattr(self.storage, 'semantic_search'):
            # Use semantic search if available
            return await self.storage.semantic_search(query, entry_type, limit)

        # Simple keyword matching fallback
        query_lower = query.lower()
        matches = []

        for entry in self._memory:
            if entry_type and entry.entry_type != entry_type:
                continue

            # Simple relevance scoring
            if query_lower in entry.content.lower():
                score = entry.content.lower().count(query_lower) / len(entry.content)
                entry.relevance_score = score
                matches.append(entry)

        # Sort by relevance and timestamp
        matches.sort(key=lambda e: (e.relevance_score, e.timestamp), reverse=True)

        return matches[:limit]

    async def get_decisions(
        self,
        tags: Optional[Set[str]] = None
    ) -> List[DecisionRecord]:
        """Get decision records, optionally filtered by tags"""

        if not tags:
            return list(self._decisions.values())

        return [
            d for d in self._decisions.values()
            if tags.intersection(d.tags)
        ]


class ContextManager:
    """
    Manages codebase context and agent memory

    Provides intelligent context for code generation based on:
    - Existing codebase patterns
    - Previous decisions
    - Project conventions
    - Recent changes
    """

    def __init__(self, memory_store: Optional[MemoryStore] = None):
        self.memory = memory_store or MemoryStore()
        self.codebase_context: Optional[CodebaseContext] = None

    async def analyze_codebase(
        self,
        project_path: str,
        files: Dict[str, str]
    ) -> CodebaseContext:
        """
        Analyze codebase to extract context

        Args:
            project_path: Project root path
            files: Dictionary of file_path -> content

        Returns:
            CodebaseContext with extracted patterns
        """
        logger.info(f"Analyzing codebase at {project_path}")

        # Detect language and framework
        language = self._detect_language(files)
        framework = self._detect_framework(files, language)

        context = CodebaseContext(
            project_name=project_path.split('/')[-1],
            language=language,
            framework=framework,
            files=files,
            file_tree=list(files.keys())
        )

        # Analyze patterns
        context.naming_conventions = self._extract_naming_conventions(files)
        context.code_style = self._extract_code_style(files, language)
        context.common_imports = self._extract_common_imports(files, language)

        # Extract dependencies
        context.dependencies = self._extract_dependencies(files, language)

        # Detect architecture
        context.architecture_pattern = self._detect_architecture(files)

        self.codebase_context = context

        # Store in memory
        await self.memory.store(
            content=f"Analyzed codebase: {context.project_name}",
            entry_type="codebase_analysis",
            metadata={
                "language": language,
                "framework": framework,
                "file_count": len(files)
            }
        )

        return context

    def _detect_language(self, files: Dict[str, str]) -> str:
        """Detect primary programming language"""

        extension_counts = {}
        for path in files.keys():
            ext = path.split('.')[-1] if '.' in path else ''
            extension_counts[ext] = extension_counts.get(ext, 0) + 1

        # Map extensions to languages
        ext_to_lang = {
            'py': 'python',
            'ts': 'typescript',
            'js': 'javascript',
            'go': 'go',
            'rs': 'rust',
            'java': 'java',
            'cs': 'csharp'
        }

        most_common_ext = max(extension_counts.items(), key=lambda x: x[1])[0]
        return ext_to_lang.get(most_common_ext, 'unknown')

    def _detect_framework(self, files: Dict[str, str], language: str) -> Optional[str]:
        """Detect framework being used"""

        # Check for framework-specific files
        framework_indicators = {
            'fastapi': ['main.py', 'app.py'],
            'django': ['manage.py', 'settings.py'],
            'flask': ['app.py'],
            'react': ['package.json'],
            'vue': ['vue.config.js'],
            'nextjs': ['next.config.js']
        }

        for framework, indicators in framework_indicators.items():
            if any(any(ind in path for path in files.keys()) for ind in indicators):
                return framework

        return None

    def _extract_naming_conventions(self, files: Dict[str, str]) -> Dict[str, str]:
        """Extract naming conventions from code"""

        conventions = {
            "class_naming": "PascalCase",  # Default assumptions
            "function_naming": "snake_case",
            "variable_naming": "snake_case"
        }

        # Would do more sophisticated analysis in production
        return conventions

    def _extract_code_style(self, files: Dict[str, str], language: str) -> Dict[str, Any]:
        """Extract code style preferences"""

        style = {
            "indent_size": 4,
            "use_semicolons": language in ["typescript", "javascript"],
            "quote_style": "double"
        }

        return style

    def _extract_common_imports(self, files: Dict[str, str], language: str) -> List[str]:
        """Extract commonly used imports"""

        import_counts = {}

        for content in files.values():
            if language == "python":
                import re
                imports = re.findall(r'(?:from\s+([\w.]+)\s+)?import\s+([\w,\s]+)', content)
                for module, items in imports:
                    key = module if module else items.split(',')[0].strip()
                    import_counts[key] = import_counts.get(key, 0) + 1

        # Return top 10 most common
        sorted_imports = sorted(import_counts.items(), key=lambda x: x[1], reverse=True)
        return [imp[0] for imp in sorted_imports[:10]]

    def _extract_dependencies(self, files: Dict[str, str], language: str) -> List[str]:
        """Extract project dependencies"""

        dependencies = []

        # Look for dependency files
        if 'requirements.txt' in files:
            deps = files['requirements.txt'].strip().split('\n')
            dependencies.extend([d.split('==')[0].split('>=')[0] for d in deps if d])
        elif 'package.json' in files:
            import json
            try:
                pkg = json.loads(files['package.json'])
                dependencies.extend(pkg.get('dependencies', {}).keys())
            except:
                pass

        return dependencies

    def _detect_architecture(self, files: Dict[str, str]) -> Optional[str]:
        """Detect architecture pattern"""

        paths = list(files.keys())

        # Check for common patterns
        if any('domain' in p for p in paths) and any('infrastructure' in p for p in paths):
            return "clean_architecture"
        elif any('models' in p for p in paths) and any('views' in p for p in paths):
            return "mvc"
        elif any('services' in p for p in paths):
            return "microservices"

        return None

    async def get_context_for_task(
        self,
        task_description: str,
        file_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get relevant context for a task

        Args:
            task_description: Description of the task
            file_path: Optional target file path

        Returns:
            Dictionary with relevant context
        """
        context = {
            "codebase": self.codebase_context.to_dict() if self.codebase_context else {},
            "relevant_memories": [],
            "relevant_decisions": [],
            "similar_files": []
        }

        # Retrieve relevant memories
        memories = await self.memory.retrieve_relevant(task_description, limit=3)
        context["relevant_memories"] = [m.content for m in memories]

        # Get recent decisions
        decisions = await self.memory.get_decisions()
        context["relevant_decisions"] = [
            {"decision": d.decision, "rationale": d.rationale}
            for d in decisions[-5:]
        ]

        # Find similar files
        if file_path and self.codebase_context:
            context["similar_files"] = self._find_similar_files(file_path)

        return context

    def _find_similar_files(self, target_path: str) -> List[str]:
        """Find files similar to target path"""
        if not self.codebase_context:
            return []

        # Simple similarity: same directory or same extension
        target_dir = '/'.join(target_path.split('/')[:-1])
        target_ext = target_path.split('.')[-1]

        similar = []
        for path in self.codebase_context.file_tree:
            if target_dir in path or path.endswith(f".{target_ext}"):
                similar.append(path)

        return similar[:5]
