"""
Centralized Logging Configuration

Provides consistent logging across all face recognition modules.
Uses [OK], [WARNING], [ERROR] prefix style for console output.

Usage:
    from core.logger import get_logger
    logger = get_logger(__name__)
    
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


# =============================================================================
# FORMATTERS
# =============================================================================

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
        prefix = self.LEVEL_PREFIXES.get(record.levelno, "[INFO]")
        message = record.getMessage()
        
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            message = f"{message}\n{record.exc_text}"
        
        return f"{prefix} {message}"


class TimestampFormatter(logging.Formatter):
    """Formatter with timestamps for file logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        message = record.getMessage()
        
        formatted = f"[{record.levelname}] {timestamp} - {record.name} - {message}"
        
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            formatted = f"{formatted}\n{record.exc_text}"
        
        return formatted


# =============================================================================
# CONFIGURATION
# =============================================================================

# Root logger name - all module loggers are children of this
ROOT_LOGGER_NAME = "face_recognition"

# Track if we've initialized
_initialized = False


def get_log_level() -> int:
    """Get log level from LOG_LEVEL environment variable."""
    level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return levels.get(level_str, logging.INFO)


def _init_root_logger(log_file: Optional[str] = None) -> logging.Logger:
    """
    Initialize the root face_recognition logger with console handler.
    Called automatically on first import.
    """
    global _initialized
    
    root_logger = logging.getLogger(ROOT_LOGGER_NAME)
    
    # Skip if already configured
    if _initialized or root_logger.handlers:
        return root_logger
    
    log_level = get_log_level()
    root_logger.setLevel(log_level)
    
    # Console handler with prefix formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(PrefixFormatter())
    root_logger.addHandler(console_handler)
    
    # Optional file handler
    log_file = log_file or os.environ.get("LOG_FILE")
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(TimestampFormatter())
        root_logger.addHandler(file_handler)
    
    # Don't propagate to Python's root logger
    root_logger.propagate = False
    
    _initialized = True
    return root_logger


# =============================================================================
# PUBLIC API
# =============================================================================

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        name: Module name (e.g., __name__, "database", "hnsw").
              If None, returns the root face_recognition logger.
    
    Returns:
        Configured logger that inherits settings from root logger.
    
    Example:
        from core.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Connected to database")  # [OK] Connected to database
    """
    # Ensure root logger is initialized
    _init_root_logger()
    
    if name is None:
        return logging.getLogger(ROOT_LOGGER_NAME)
    
    # Strip common prefixes for cleaner names
    clean_name = name
    for prefix in ("app.", "face_recognition."):
        if clean_name.startswith(prefix):
            clean_name = clean_name[len(prefix):]
    
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{clean_name}")


def setup_logger(
    name: str = ROOT_LOGGER_NAME,
    log_file: Optional[str] = None,
    level: Optional[int] = None,
) -> logging.Logger:
    """
    Set up and return a configured logger (legacy compatibility).
    
    For new code, prefer get_logger(__name__) instead.
    """
    if level is not None:
        os.environ["LOG_LEVEL"] = logging.getLevelName(level)
    
    return _init_root_logger(log_file)


# Initialize root logger on import
logger = _init_root_logger()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

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


__all__ = [
    "logger",
    "get_logger",
    "setup_logger",
    "debug",
    "info",
    "warning",
    "error",
    "exception",
    "critical",
]
