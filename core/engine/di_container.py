"""
Dependency Injection Container — CognitionOS Core Engine

Lightweight async-aware DI container with support for:
- Singleton, transient, and scoped service lifetimes
- Lazy initialization
- Factory functions
- Interface-to-implementation binding
- Circular dependency detection
- Thread-safe resolution
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any, Callable, Dict, Generic, List, Optional,
    Set, Type, TypeVar, Union, get_type_hints,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ServiceLifetime(str, Enum):
    """Defines how long a resolved service instance lives."""
    SINGLETON = "singleton"    # One instance for the entire container
    TRANSIENT = "transient"    # New instance every resolution
    SCOPED = "scoped"          # One instance per scope


@dataclass
class ServiceDescriptor:
    """Describes a registered service binding."""
    service_type: Type
    implementation: Union[Type, Callable, None] = None
    factory: Optional[Callable] = None
    instance: Optional[Any] = None
    lifetime: ServiceLifetime = ServiceLifetime.SINGLETON
    tags: List[str] = field(default_factory=list)
    registered_at: float = field(default_factory=time.time)
    resolved_count: int = 0
    last_resolved_at: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScopeContext:
    """Holds scoped instances for a particular scope."""
    scope_id: str
    instances: Dict[Type, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    parent: Optional["ScopeContext"] = None


class CircularDependencyError(Exception):
    """Raised when a circular dependency chain is detected."""
    def __init__(self, chain: List[str]):
        self.chain = chain
        super().__init__(
            f"Circular dependency detected: {' -> '.join(chain)}"
        )


class ServiceNotRegisteredError(Exception):
    """Raised when attempting to resolve an unregistered service."""
    def __init__(self, service_type: Type):
        self.service_type = service_type
        super().__init__(
            f"Service '{service_type.__name__}' is not registered in the container."
        )


class DependencyContainer:
    """
    Production-grade dependency injection container.

    Features:
    - Register services with singleton/transient/scoped lifetimes
    - Resolve services with automatic constructor injection
    - Factory-based registration for complex creation logic
    - Scoped containers for request-level isolation
    - Circular dependency detection at resolution time
    - Thread-safe singleton resolution
    - Async factory support
    """

    def __init__(self, name: str = "root"):
        self._name = name
        self._registry: Dict[Type, ServiceDescriptor] = {}
        self._singletons: Dict[Type, Any] = {}
        self._lock = threading.RLock()
        self._resolving: Set[Type] = set()  # For circular detection
        self._resolution_order: List[str] = []
        self._child_containers: List[DependencyContainer] = []
        self._disposed = False
        logger.debug("DependencyContainer '%s' created", name)

    # ── Registration ──

    def register_singleton(
        self,
        service_type: Type[T],
        implementation: Optional[Union[Type[T], Callable[..., T]]] = None,
        *,
        instance: Optional[T] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "DependencyContainer":
        """Register a singleton service (one instance for container lifetime)."""
        return self._register(
            service_type, implementation,
            instance=instance,
            lifetime=ServiceLifetime.SINGLETON,
            tags=tags or [],
            metadata=metadata or {},
        )

    def register_transient(
        self,
        service_type: Type[T],
        implementation: Optional[Union[Type[T], Callable[..., T]]] = None,
        *,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "DependencyContainer":
        """Register a transient service (new instance per resolution)."""
        return self._register(
            service_type, implementation,
            lifetime=ServiceLifetime.TRANSIENT,
            tags=tags or [],
            metadata=metadata or {},
        )

    def register_scoped(
        self,
        service_type: Type[T],
        implementation: Optional[Union[Type[T], Callable[..., T]]] = None,
        *,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "DependencyContainer":
        """Register a scoped service (one per scope)."""
        return self._register(
            service_type, implementation,
            lifetime=ServiceLifetime.SCOPED,
            tags=tags or [],
            metadata=metadata or {},
        )

    def register_factory(
        self,
        service_type: Type[T],
        factory: Callable[..., T],
        *,
        lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "DependencyContainer":
        """Register a service with a custom factory function."""
        descriptor = ServiceDescriptor(
            service_type=service_type,
            factory=factory,
            lifetime=lifetime,
            tags=tags or [],
            metadata=metadata or {},
        )
        with self._lock:
            self._registry[service_type] = descriptor
        logger.debug(
            "Registered factory for '%s' (%s)",
            service_type.__name__, lifetime.value,
        )
        return self

    def register_instance(
        self,
        service_type: Type[T],
        instance: T,
        *,
        tags: Optional[List[str]] = None,
    ) -> "DependencyContainer":
        """Register a pre-created instance as a singleton."""
        return self._register(
            service_type, None,
            instance=instance,
            lifetime=ServiceLifetime.SINGLETON,
            tags=tags or [],
        )

    def _register(
        self,
        service_type: Type,
        implementation: Optional[Union[Type, Callable]],
        *,
        instance: Optional[Any] = None,
        lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
        tags: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "DependencyContainer":
        """Internal registration logic."""
        self._check_disposed()
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation=implementation or service_type,
            instance=instance,
            lifetime=lifetime,
            tags=tags,
            metadata=metadata or {},
        )
        with self._lock:
            self._registry[service_type] = descriptor
            if instance is not None:
                self._singletons[service_type] = instance
        logger.debug(
            "Registered '%s' -> '%s' (%s)",
            service_type.__name__,
            (implementation or service_type).__name__
            if implementation or service_type else "instance",
            lifetime.value,
        )
        return self

    # ── Resolution ──

    def resolve(
        self,
        service_type: Type[T],
        scope: Optional[ScopeContext] = None,
    ) -> T:
        """
        Resolve a service instance synchronously.

        Performs constructor injection, respects lifetimes,
        and detects circular dependencies.
        """
        self._check_disposed()

        with self._lock:
            if service_type not in self._registry:
                raise ServiceNotRegisteredError(service_type)

            descriptor = self._registry[service_type]

            # Check circular dependency
            if service_type in self._resolving:
                chain = self._resolution_order + [service_type.__name__]
                raise CircularDependencyError(chain)

            # Singleton: return cached
            if descriptor.lifetime == ServiceLifetime.SINGLETON:
                if service_type in self._singletons:
                    descriptor.resolved_count += 1
                    descriptor.last_resolved_at = time.time()
                    return self._singletons[service_type]

            # Scoped: return from scope cache
            if descriptor.lifetime == ServiceLifetime.SCOPED and scope:
                if service_type in scope.instances:
                    descriptor.resolved_count += 1
                    descriptor.last_resolved_at = time.time()
                    return scope.instances[service_type]

            # Mark as resolving for circular detection
            self._resolving.add(service_type)
            self._resolution_order.append(service_type.__name__)

        try:
            instance = self._create_instance(descriptor, scope)

            with self._lock:
                descriptor.resolved_count += 1
                descriptor.last_resolved_at = time.time()

                if descriptor.lifetime == ServiceLifetime.SINGLETON:
                    self._singletons[service_type] = instance
                elif descriptor.lifetime == ServiceLifetime.SCOPED and scope:
                    scope.instances[service_type] = instance

            return instance

        finally:
            with self._lock:
                self._resolving.discard(service_type)
                if service_type.__name__ in self._resolution_order:
                    self._resolution_order.remove(service_type.__name__)

    async def resolve_async(
        self,
        service_type: Type[T],
        scope: Optional[ScopeContext] = None,
    ) -> T:
        """
        Resolve a service asynchronously.
        Supports async factory functions.
        """
        self._check_disposed()

        with self._lock:
            if service_type not in self._registry:
                raise ServiceNotRegisteredError(service_type)

            descriptor = self._registry[service_type]

            if service_type in self._resolving:
                chain = self._resolution_order + [service_type.__name__]
                raise CircularDependencyError(chain)

            if descriptor.lifetime == ServiceLifetime.SINGLETON:
                if service_type in self._singletons:
                    descriptor.resolved_count += 1
                    descriptor.last_resolved_at = time.time()
                    return self._singletons[service_type]

            if descriptor.lifetime == ServiceLifetime.SCOPED and scope:
                if service_type in scope.instances:
                    descriptor.resolved_count += 1
                    descriptor.last_resolved_at = time.time()
                    return scope.instances[service_type]

            self._resolving.add(service_type)
            self._resolution_order.append(service_type.__name__)

        try:
            instance = await self._create_instance_async(descriptor, scope)

            with self._lock:
                descriptor.resolved_count += 1
                descriptor.last_resolved_at = time.time()

                if descriptor.lifetime == ServiceLifetime.SINGLETON:
                    self._singletons[service_type] = instance
                elif descriptor.lifetime == ServiceLifetime.SCOPED and scope:
                    scope.instances[service_type] = instance

            return instance

        finally:
            with self._lock:
                self._resolving.discard(service_type)
                if service_type.__name__ in self._resolution_order:
                    self._resolution_order.remove(service_type.__name__)

    def _create_instance(
        self,
        descriptor: ServiceDescriptor,
        scope: Optional[ScopeContext],
    ) -> Any:
        """Create an instance using the descriptor's factory or constructor."""
        # Pre-created instance
        if descriptor.instance is not None:
            return descriptor.instance

        # Factory function
        if descriptor.factory is not None:
            if inspect.iscoroutinefunction(descriptor.factory):
                raise TypeError(
                    f"Factory for '{descriptor.service_type.__name__}' is async. "
                    "Use resolve_async() instead."
                )
            return descriptor.factory(self)

        # Constructor injection
        impl = descriptor.implementation
        if impl is None:
            raise ValueError(
                f"No implementation or factory for '{descriptor.service_type.__name__}'"
            )

        deps = self._resolve_constructor_deps(impl, scope)
        return impl(**deps)

    async def _create_instance_async(
        self,
        descriptor: ServiceDescriptor,
        scope: Optional[ScopeContext],
    ) -> Any:
        """Create instance with async support."""
        if descriptor.instance is not None:
            return descriptor.instance

        if descriptor.factory is not None:
            if inspect.iscoroutinefunction(descriptor.factory):
                return await descriptor.factory(self)
            return descriptor.factory(self)

        impl = descriptor.implementation
        if impl is None:
            raise ValueError(
                f"No implementation or factory for '{descriptor.service_type.__name__}'"
            )

        deps = self._resolve_constructor_deps(impl, scope)
        return impl(**deps)

    def _resolve_constructor_deps(
        self,
        impl: Type,
        scope: Optional[ScopeContext],
    ) -> Dict[str, Any]:
        """Resolve constructor dependencies via type hints."""
        deps = {}
        try:
            hints = get_type_hints(impl.__init__)
        except Exception:
            return deps

        for param_name, param_type in hints.items():
            if param_name in ("return", "self"):
                continue
            if param_type in self._registry:
                deps[param_name] = self.resolve(param_type, scope)

        return deps

    # ── Scoping ──

    def create_scope(self, scope_id: str = "") -> ScopeContext:
        """Create a new scope for scoped service resolution."""
        import uuid
        return ScopeContext(
            scope_id=scope_id or str(uuid.uuid4()),
        )

    def create_child_container(self, name: str = "") -> "DependencyContainer":
        """Create a child container that inherits parent registrations."""
        child = DependencyContainer(name=name or f"{self._name}.child")
        # Copy registrations (not instances)
        with self._lock:
            for stype, descriptor in self._registry.items():
                child._registry[stype] = ServiceDescriptor(
                    service_type=descriptor.service_type,
                    implementation=descriptor.implementation,
                    factory=descriptor.factory,
                    lifetime=descriptor.lifetime,
                    tags=descriptor.tags[:],
                    metadata=descriptor.metadata.copy(),
                )
            self._child_containers.append(child)
        return child

    # ── Query ──

    def is_registered(self, service_type: Type) -> bool:
        """Check if a service type is registered."""
        return service_type in self._registry

    def get_by_tag(self, tag: str) -> List[Any]:
        """Resolve all services with a specific tag."""
        results = []
        with self._lock:
            for stype, desc in self._registry.items():
                if tag in desc.tags:
                    try:
                        results.append(self.resolve(stype))
                    except Exception as exc:
                        logger.warning(
                            "Failed to resolve tagged service '%s': %s",
                            stype.__name__, exc,
                        )
        return results

    def get_all_descriptors(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata about all registered services."""
        with self._lock:
            return {
                stype.__name__: {
                    "lifetime": desc.lifetime.value,
                    "implementation": (
                        desc.implementation.__name__
                        if desc.implementation else "factory"
                    ),
                    "has_instance": desc.instance is not None
                    or stype in self._singletons,
                    "resolved_count": desc.resolved_count,
                    "last_resolved_at": desc.last_resolved_at,
                    "tags": desc.tags,
                    "metadata": desc.metadata,
                }
                for stype, desc in self._registry.items()
            }

    # ── Lifecycle ──

    def dispose(self):
        """
        Dispose all singleton instances that have a close/dispose method.
        Marks the container as disposed.
        """
        with self._lock:
            self._disposed = True
            for stype, instance in self._singletons.items():
                if hasattr(instance, "dispose"):
                    try:
                        instance.dispose()
                    except Exception as exc:
                        logger.error(
                            "Error disposing '%s': %s", stype.__name__, exc
                        )
                elif hasattr(instance, "close"):
                    try:
                        instance.close()
                    except Exception as exc:
                        logger.error(
                            "Error closing '%s': %s", stype.__name__, exc
                        )

            for child in self._child_containers:
                child.dispose()

            self._singletons.clear()
            self._child_containers.clear()
            logger.info("DependencyContainer '%s' disposed", self._name)

    async def dispose_async(self):
        """Async dispose with support for async close/dispose methods."""
        with self._lock:
            self._disposed = True

        for stype, instance in list(self._singletons.items()):
            if hasattr(instance, "dispose"):
                try:
                    if asyncio.iscoroutinefunction(instance.dispose):
                        await instance.dispose()
                    else:
                        instance.dispose()
                except Exception as exc:
                    logger.error(
                        "Error disposing '%s': %s", stype.__name__, exc
                    )
            elif hasattr(instance, "close"):
                try:
                    if asyncio.iscoroutinefunction(instance.close):
                        await instance.close()
                    else:
                        instance.close()
                except Exception as exc:
                    logger.error(
                        "Error closing '%s': %s", stype.__name__, exc
                    )

        for child in self._child_containers:
            await child.dispose_async()

        with self._lock:
            self._singletons.clear()
            self._child_containers.clear()
        logger.info("DependencyContainer '%s' async-disposed", self._name)

    def _check_disposed(self):
        if self._disposed:
            raise RuntimeError(
                f"Container '{self._name}' has been disposed and cannot be used."
            )

    # ── Stats ──

    def get_stats(self) -> Dict[str, Any]:
        """Return container stats for monitoring."""
        with self._lock:
            return {
                "name": self._name,
                "registered_services": len(self._registry),
                "active_singletons": len(self._singletons),
                "child_containers": len(self._child_containers),
                "disposed": self._disposed,
                "services": {
                    stype.__name__: {
                        "lifetime": desc.lifetime.value,
                        "resolved_count": desc.resolved_count,
                    }
                    for stype, desc in self._registry.items()
                },
            }

    def __repr__(self) -> str:
        return (
            f"DependencyContainer(name='{self._name}', "
            f"services={len(self._registry)}, "
            f"singletons={len(self._singletons)})"
        )


# ── Module-level convenience ──

_global_container: Optional[DependencyContainer] = None


def get_container() -> DependencyContainer:
    """Get or create the global DI container."""
    global _global_container
    if _global_container is None:
        _global_container = DependencyContainer(name="global")
    return _global_container


def reset_container():
    """Reset the global container (for testing)."""
    global _global_container
    if _global_container:
        _global_container.dispose()
    _global_container = None
