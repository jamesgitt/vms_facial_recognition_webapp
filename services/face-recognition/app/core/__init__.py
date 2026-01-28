"""
Core module for Face Recognition API.

Contains centralized configuration, logging, exceptions, and application state.
"""

from .config import settings, get_settings, Settings
from .logger import logger, get_logger
from .exceptions import (
    FaceRecognitionError,
    NoFaceDetectedError,
    FeatureExtractionError,
    InvalidImageError,
    DatabaseConnectionError,
    IndexNotInitializedError,
)

__all__ = [
    # Config
    "settings",
    "get_settings",
    "Settings",
    # Logger
    "logger",
    "get_logger",
    # Exceptions
    "FaceRecognitionError",
    "NoFaceDetectedError",
    "FeatureExtractionError",
    "InvalidImageError",
    "DatabaseConnectionError",
    "IndexNotInitializedError",
]
