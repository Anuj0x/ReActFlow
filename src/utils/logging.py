"""Structured logging configuration using structlog."""

import sys
import logging
from typing import Any, Dict

import structlog


def setup_logging(
    log_level: str = "INFO",
    enable_structured: bool = True,
    enable_json: bool = False
) -> None:
    """Set up structured logging for the application."""

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if enable_structured:
        # Add additional processors for structured logging
        processors.extend([
            structlog.processors.JSONRenderer()
            if enable_json
            else structlog.dev.ConsoleRenderer(colors=True),
        ])
    else:
        processors.append(structlog.write_logs)

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.WriteLoggerFactory(),
    )


def get_logger(name: str) -> Any:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


def log_execution_error(
    logger: Any,
    error: Exception,
    context: Dict[str, Any] = None
) -> None:
    """Log execution errors with context."""
    context = context or {}
    logger.error(
        "Execution error occurred",
        error=str(error),
        error_type=type(error).__name__,
        **context
    )


def log_performance_metrics(
    logger: Any,
    operation: str,
    duration: float,
    metadata: Dict[str, Any] = None
) -> None:
    """Log performance metrics."""
    metadata = metadata or {}
    logger.info(
        "Performance metric",
        operation=operation,
        duration=duration,
        duration_ms=duration * 1000,
        **metadata
    )
