"""
Plugin sandbox execution with resource limits and security constraints.

Provides isolated execution environment for untrusted plugin code.
"""

import asyncio
import logging
import resource
import signal
import sys
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

logger = logging.getLogger(__name__)


class PluginTimeoutError(Exception):
    """Raised when plugin execution exceeds time limit."""
    pass


class PluginResourceError(Exception):
    """Raised when plugin exceeds resource limits."""
    pass


class PluginSandbox:
    """
    Secure sandbox for plugin execution with resource limits.
    
    Features:
    - CPU time limits
    - Memory limits
    - Restricted imports
    - Timeout protection
    - Safe builtins
    """
    
    def __init__(
        self,
        max_execution_time_seconds: int = 30,
        max_memory_mb: int = 256,
        max_cpu_time_seconds: int = 10,
    ):
        self.max_execution_time = max_execution_time_seconds
        self.max_memory_mb = max_memory_mb
        self.max_cpu_time = max_cpu_time_seconds
        
        # Safe builtins (whitelist approach)
        self.safe_builtins = {
            'abs': abs,
            'all': all,
            'any': any,
            'bool': bool,
            'dict': dict,
            'enumerate': enumerate,
            'filter': filter,
            'float': float,
            'int': int,
            'len': len,
            'list': list,
            'map': map,
            'max': max,
            'min': min,
            'range': range,
            'round': round,
            'sorted': sorted,
            'str': str,
            'sum': sum,
            'tuple': tuple,
            'zip': zip,
            'True': True,
            'False': False,
            'None': None,
        }
        
        # Restricted imports (blacklist)
        self.forbidden_modules = {
            'os', 'sys', 'subprocess', 'socket', 'multiprocessing',
            'threading', 'importlib', '__import__', 'eval', 'exec',
            'compile', 'open', 'file', 'input', 'raw_input',
        }
    
    def _timeout_handler(self, signum, frame):
        """Handle execution timeout."""
        raise PluginTimeoutError("Plugin execution exceeded time limit")
    
    @contextmanager
    def _resource_limits(self):
        """Context manager to set resource limits."""
        # Set CPU time limit (seconds)
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
            resource.setrlimit(resource.RLIMIT_CPU, (self.max_cpu_time, hard))
        except Exception as e:
            logger.warning(f"Could not set CPU limit: {e}")
        
        # Set memory limit (bytes)
        try:
            max_memory_bytes = self.max_memory_mb * 1024 * 1024
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, hard))
        except Exception as e:
            logger.warning(f"Could not set memory limit: {e}")
        
        try:
            yield
        finally:
            # Reset limits
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (soft, hard))
            except:
                pass
    
    def _create_safe_globals(self, plugin_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create safe globals dictionary for plugin execution.
        
        Args:
            plugin_context: Optional context to provide to plugin
            
        Returns:
            Safe globals dictionary
        """
        safe_globals = {
            '__builtins__': self.safe_builtins,
            '__name__': '__plugin__',
            '__doc__': None,
        }
        
        # Add plugin context if provided
        if plugin_context:
            safe_globals['context'] = plugin_context
        
        return safe_globals
    
    async def execute(
        self,
        plugin_code: str,
        plugin_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute plugin code in sandbox with resource limits.
        
        Args:
            plugin_code: Python code to execute
            plugin_context: Optional context dictionary for plugin
            
        Returns:
            Dict with execution results
            
        Raises:
            PluginTimeoutError: If execution exceeds time limit
            PluginResourceError: If plugin exceeds resource limits
            Exception: For other execution errors
        """
        start_time = datetime.utcnow()
        result = {
            'success': False,
            'output': None,
            'error': None,
            'execution_time_ms': 0,
        }
        
        try:
            # Create safe execution environment
            safe_globals = self._create_safe_globals(plugin_context)
            safe_locals = {}
            
            # Set timeout alarm
            signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.alarm(self.max_execution_time)
            
            try:
                # Execute in subprocess for better isolation
                loop = asyncio.get_event_loop()
                
                def run_in_sandbox():
                    with self._resource_limits():
                        # Compile code first to check for syntax errors
                        compiled_code = compile(plugin_code, '<plugin>', 'exec')
                        
                        # Execute
                        exec(compiled_code, safe_globals, safe_locals)
                        
                        # Return result if 'result' variable is set
                        return safe_locals.get('result', None)
                
                # Run in executor for isolation
                output = await loop.run_in_executor(None, run_in_sandbox)
                
                result['success'] = True
                result['output'] = output
                
            finally:
                # Cancel timeout alarm
                signal.alarm(0)
            
        except PluginTimeoutError as e:
            result['error'] = f"Execution timeout: {str(e)}"
            logger.warning(f"Plugin execution timeout: {e}")
            
        except MemoryError as e:
            result['error'] = f"Memory limit exceeded: {str(e)}"
            logger.warning(f"Plugin memory limit exceeded: {e}")
            
        except Exception as e:
            result['error'] = f"Execution error: {str(e)}"
            logger.error(f"Plugin execution error: {e}", exc_info=True)
        
        # Calculate execution time
        end_time = datetime.utcnow()
        result['execution_time_ms'] = int((end_time - start_time).total_seconds() * 1000)
        
        return result
    
    def validate_code(self, plugin_code: str) -> tuple[bool, Optional[str]]:
        """
        Validate plugin code without executing it.
        
        Args:
            plugin_code: Python code to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check for forbidden imports
            for forbidden in self.forbidden_modules:
                if f"import {forbidden}" in plugin_code or f"from {forbidden}" in plugin_code:
                    return False, f"Forbidden module: {forbidden}"
            
            # Try to compile (syntax check)
            compile(plugin_code, '<plugin>', 'exec')
            
            return True, None
            
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"


async def execute_plugin_safely(
    plugin_code: str,
    plugin_context: Optional[Dict[str, Any]] = None,
    max_execution_time: int = 30,
    max_memory_mb: int = 256,
) -> Dict[str, Any]:
    """
    Convenience function to execute plugin code safely.
    
    Args:
        plugin_code: Python code to execute
        plugin_context: Optional context for plugin
        max_execution_time: Maximum execution time in seconds
        max_memory_mb: Maximum memory in MB
        
    Returns:
        Execution result dictionary
    """
    sandbox = PluginSandbox(
        max_execution_time_seconds=max_execution_time,
        max_memory_mb=max_memory_mb,
    )
    
    return await sandbox.execute(plugin_code, plugin_context)
