"""
Centralized Logging Configuration

Provides consistent logging across all face recognition modules.
Replaces print statements with structured logging while maintaining
the [OK], [WARNING], [ERROR] prefix style.

Usage:
    from logger import logger
    
    logger.info("Models loaded")           # [OK] Models loaded
    logger.warning("No face detected")     # [WARNING] No face detected
    logger.error("Connection failed")      # [ERROR] Connection failed
    logger.debug("Processing image...")    # [DEBUG] Processing image...
"""

import os
import sys
import logging
from typing import Optional
from pathlib import Path


class PrefixFormatter(logging.Formatter):
    """
    Custom formatter that uses [OK], [WARNING], [ERROR] prefixes
    to match existing print statement style.
    """
    
    LEVEL_PREFIXES = {
        logging.DEBUG: "[DEBUG]",
        logging.INFO: "[OK]",
        logging.WARNING: "[WARNING]",
        logging.ERROR: "[ERROR]",
        logging.CRITICAL: "[CRITICAL]",
    }
    
    def format(self, record: logging.LogRecord) -> str:
        # Get prefix based on log level
        prefix = self.LEVEL_PREFIXES.get(record.levelno, "[INFO]")
        
        # Format the message
        message = record.getMessage()
        
        # Add exception info if present
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            message = f"{message}\n{record.exc_text}"
        
        return f"{prefix} {message}"


class TimestampFormatter(logging.Formatter):
    """
    Formatter with timestamps for file logging.
    """
    
    LEVEL_PREFIXES = {
        logging.DEBUG: "DEBUG",
        logging.INFO: "INFO",
        logging.WARNING: "WARNING",
        logging.ERROR: "ERROR",
        logging.CRITICAL: "CRITICAL",
    }
    
    def format(self, record: logging.LogRecord) -> str:
        prefix = self.LEVEL_PREFIXES.get(record.levelno, "INFO")
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        message = record.getMessage()
        
        formatted = f"[{prefix}] {timestamp} - {record.name} - {message}"
        
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            formatted = f"{formatted}\n{record.exc_text}"
        
        return formatted


def get_log_level() -> int:
    """Get log level from environment variable."""
    level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return levels.get(level_str, logging.INFO)


def setup_logger(
    name: str = "face_recognition",
    log_file: Optional[str] = None,
    level: Optional[int] = None,
) -> logging.Logger:
    """
    Set up and return a configured logger.
    
    Args:
        name: Logger name (default: "face_recognition")
        log_file: Optional path to log file
        level: Log level (default: from LOG_LEVEL env var or INFO)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Set level
    log_level = level if level is not None else get_log_level()
    logger.setLevel(log_level)
    
    # Console handler with prefix formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(PrefixFormatter())
    logger.addHandler(console_handler)
    
    # Optional file handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(TimestampFormatter())
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


# Default logger instance
logger = setup_logger()


# Convenience functions for module-level logging
def debug(msg: str, *args, **kwargs) -> None:
    """Log debug message."""
    logger.debug(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs) -> None:
    """Log info message (shows as [OK])."""
    logger.info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs) -> None:
    """Log warning message."""
    logger.warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs) -> None:
    """Log error message."""
    logger.error(msg, *args, **kwargs)


def exception(msg: str, *args, **kwargs) -> None:
    """Log error message with exception traceback."""
    logger.exception(msg, *args, **kwargs)


def critical(msg: str, *args, **kwargs) -> None:
    """Log critical message."""
    logger.critical(msg, *args, **kwargs)


# Child logger factory for module-specific loggers
def get_logger(module_name: str) -> logging.Logger:
    """
    Get a child logger for a specific module.
    
    Args:
        module_name: Name of the module (e.g., "database", "hnsw")
    
    Returns:
        Child logger that inherits settings from main logger
    
    Example:
        from logger import get_logger
        logger = get_logger("database")
        logger.info("Connected to database")
    """
    return logging.getLogger(f"face_recognition.{module_name}")


__all__ = [
    "logger",
    "setup_logger",
    "get_logger",
    "debug",
    "info",
    "warning",
    "error",
    "exception",
    "critical",
]
