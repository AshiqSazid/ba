"""
Memory Manager Service - Fallback version without psutil dependency
Provides basic memory monitoring functionality without external dependencies.
"""

import gc
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
import structlog
from contextlib import contextmanager

logger = structlog.get_logger(__name__)


class MemoryManagerFallback:
    """
    Fallback memory monitoring service that doesn't require psutil.
    Uses built-in Python modules for basic memory tracking.
    """

    def __init__(self,
                 warning_threshold_mb: int = 500,
                 critical_threshold_mb: int = 1000,
                 cleanup_interval_seconds: int = 60):
        self.warning_threshold_mb = warning_threshold_mb
        self.critical_threshold_mb = critical_threshold_mb
        self.cleanup_interval_seconds = cleanup_interval_seconds

        self._monitoring = False
        self._monitor_thread = None
        self._lock = threading.Lock()

        # Callbacks for cleanup
        self._cleanup_callbacks: Dict[str, Callable] = {}

        # Statistics
        self._peak_memory_mb = 0
        self._cleanup_count = 0
        self._last_cleanup = None

    def register_cleanup_callback(self, name: str, callback: Callable):
        """Register a cleanup callback function."""
        with self._lock:
            self._cleanup_callbacks[name] = callback
            logger.info(f"Registered cleanup callback: {name}")

    def unregister_cleanup_callback(self, name: str):
        """Unregister a cleanup callback function."""
        with self._lock:
            if name in self._cleanup_callbacks:
                del self._cleanup_callbacks[name]
                logger.info(f"Unregistered cleanup callback: {name}")

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics (fallback version)."""
        try:
            # Use resource module if available (Unix-like systems)
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            maxrss_mb = usage.ru_maxrss / 1024  # Linux: KB, macOS: bytes

            current_rss_mb = maxrss_mb
            current_vms_mb = maxrss_mb  # Same for fallback

        except (ImportError, AttributeError):
            # Fallback: estimate based on Python garbage collection stats
            gc_stats = gc.get_stats()
            total_objects = sum(sum(stat['alloc'] - stat['free'] for stat in generation.values())
                              for generation in gc_stats)
            # Rough estimate: 100 bytes per Python object
            current_rss_mb = (total_objects * 100) / 1024 / 1024
            current_vms_mb = current_rss_mb

        # Update peak
        if current_rss_mb > self._peak_memory_mb:
            self._peak_memory_mb = current_rss_mb

        return {
            "process": {
                "rss_mb": current_rss_mb,
                "vms_mb": current_vms_mb,
                "percent": 0.0,  # Not available in fallback
                "peak_mb": self._peak_memory_mb
            },
            "system": {
                "total_mb": 0.0,  # Not available in fallback
                "available_mb": 0.0,  # Not available in fallback
                "percent": 0.0  # Not available in fallback
            },
            "cleanup_stats": {
                "cleanup_count": self._cleanup_count,
                "last_cleanup": self._last_cleanup.isoformat() if self._last_cleanup else None
            }
        }

    def _check_memory_and_cleanup(self):
        """Check memory usage and trigger cleanup if needed."""
        memory_stats = self.get_memory_usage()
        current_mb = memory_stats["process"]["rss_mb"]

        # Warning threshold
        if current_mb > self.warning_threshold_mb:
            logger.warning(f"High memory usage detected: {current_mb:.1f}MB (threshold: {self.warning_threshold_mb}MB)")

        # Critical threshold - trigger aggressive cleanup
        if current_mb > self.critical_threshold_mb:
            logger.error(f"Critical memory usage detected: {current_mb:.1f}MB (threshold: {self.critical_threshold_mb}MB)")
            self._trigger_cleanup("critical_threshold")

    def _trigger_cleanup(self, reason: str):
        """Trigger all registered cleanup callbacks."""
        with self._lock:
            logger.info(f"Triggering cleanup due to: {reason}")

            cleanup_count = 0

            # Execute all cleanup callbacks
            for name, callback in self._cleanup_callbacks.items():
                try:
                    logger.info(f"Running cleanup callback: {name}")
                    callback()
                    cleanup_count += 1
                except Exception as e:
                    logger.error(f"Cleanup callback {name} failed: {e}")

            # Force garbage collection
            collected = gc.collect()

            # Update stats
            self._cleanup_count += 1
            self._last_cleanup = datetime.now()

            # Log results
            after_memory = self.get_memory_usage()
            memory_reduction = after_memory["process"]["rss_mb"] - self.get_memory_usage()["process"]["rss_mb"]

            logger.info(f"Cleanup completed: {cleanup_count} callbacks executed, "
                       f"{collected} objects collected, memory change: {memory_reduction:+.1f}MB")

    def start_monitoring(self):
        """Start background memory monitoring."""
        if self._monitoring:
            logger.warning("Memory monitoring already started")
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info(f"Memory monitoring started (interval: {self.cleanup_interval_seconds}s)")

    def stop_monitoring(self):
        """Stop background memory monitoring."""
        if not self._monitoring:
            return

        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Memory monitoring stopped")

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                self._check_memory_and_cleanup()
                time.sleep(self.cleanup_interval_seconds)
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(10)  # Brief pause on error

    def force_cleanup(self, reason: str = "manual"):
        """Force immediate cleanup."""
        logger.info(f"Manual cleanup triggered: {reason}")
        self._trigger_cleanup(f"manual_{reason}")

    @contextmanager
    def memory_monitor(self, operation: str, log_threshold_mb: float = 50.0):
        """Context manager to monitor memory usage during an operation."""
        start_stats = self.get_memory_usage()
        start_memory = start_stats["process"]["rss_mb"]

        try:
            yield
        finally:
            end_stats = self.get_memory_usage()
            end_memory = end_stats["process"]["rss_mb"]
            memory_delta = end_memory - start_memory

            if abs(memory_delta) >= log_threshold_mb:
                logger.info(f"Memory usage during {operation}: {memory_delta:+.1f}MB "
                           f"({start_memory:.1f}MB → {end_memory:.1f}MB)")

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        memory_stats = self.get_memory_usage()
        current_mb = memory_stats["process"]["rss_mb"]

        # Determine status
        if current_mb > self.critical_threshold_mb:
            status = "critical"
        elif current_mb > self.warning_threshold_mb:
            status = "warning"
        else:
            status = "healthy"

        return {
            "status": status,
            "monitoring": self._monitoring,
            "thresholds": {
                "warning_mb": self.warning_threshold_mb,
                "critical_mb": self.critical_threshold_mb
            },
            "memory": memory_stats,
            "callbacks": list(self._cleanup_callbacks.keys()),
            "note": "Using fallback memory manager (psutil not available)"
        }


# Try to import psutil version first, fallback to this version
try:
    from .memory_manager import MemoryManager
    logger.info("Using psutil-based memory manager")
except ImportError:
    MemoryManager = MemoryManagerFallback
    logger.warning("Using fallback memory manager (psutil not available)")


# Global memory manager instance
_global_memory_manager = None
_manager_lock = threading.Lock()

def get_memory_manager() -> MemoryManager:
    """Get global memory manager instance."""
    global _global_memory_manager

    if _global_memory_manager is None:
        with _manager_lock:
            if _global_memory_manager is None:
                _global_memory_manager = MemoryManager()
                logger.info("Global memory manager initialized")

    return _global_memory_manager

def initialize_memory_monitoring():
    """Initialize memory monitoring with default callbacks."""
    manager = get_memory_manager()

    # Register Python garbage collection
    def python_gc_cleanup():
        collected = gc.collect()
        logger.debug(f"Python garbage collection: {collected} objects collected")

    manager.register_cleanup_callback("python_gc", python_gc_cleanup)

    # Start monitoring
    manager.start_monitoring()

    logger.info("Memory monitoring initialized with default cleanup callbacks")

def cleanup_memory_manager():
    """Cleanup global memory manager."""
    global _global_memory_manager

    with _manager_lock:
        if _global_memory_manager is not None:
            _global_memory_manager.stop_monitoring()
            _global_memory_manager = None
            logger.info("Global memory manager cleaned up")