"""
Custom Exceptions for Face Recognition API

Provides domain-specific exceptions for better error handling and debugging.

Usage:
    from core.exceptions import NoFaceDetectedError
    
    if not faces:
        raise NoFaceDetectedError("No face found in uploaded image")
"""

from typing import Optional, Any


class FaceRecognitionError(Exception):
    """
    Base exception for all face recognition errors.
    
    All custom exceptions inherit from this class, allowing
    catch-all handling when needed.
    """
    
    def __init__(
        self,
        message: str = "Face recognition error occurred",
        details: Optional[Any] = None,
    ):
        self.message = message
        self.details = details
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class NoFaceDetectedError(FaceRecognitionError):
    """Raised when no face is detected in an image."""
    
    def __init__(
        self,
        message: str = "No face detected in image",
        details: Optional[Any] = None,
    ):
        super().__init__(message, details)


class MultipleFacesError(FaceRecognitionError):
    """Raised when multiple faces are detected but only one is expected."""
    
    def __init__(
        self,
        face_count: int,
        message: str = "Multiple faces detected",
        details: Optional[Any] = None,
    ):
        self.face_count = face_count
        super().__init__(f"{message}: found {face_count} faces", details)


class FeatureExtractionError(FaceRecognitionError):
    """Raised when face feature extraction fails."""
    
    def __init__(
        self,
        message: str = "Failed to extract face features",
        details: Optional[Any] = None,
    ):
        super().__init__(message, details)


class InvalidFeatureError(FaceRecognitionError):
    """Raised when feature vector is invalid (wrong dimension, NaN, etc.)."""
    
    def __init__(
        self,
        expected_dim: int = 128,
        actual_dim: Optional[int] = None,
        message: str = "Invalid feature vector",
        details: Optional[Any] = None,
    ):
        self.expected_dim = expected_dim
        self.actual_dim = actual_dim
        if actual_dim is not None:
            message = f"{message}: expected {expected_dim}D, got {actual_dim}D"
        super().__init__(message, details)


class InvalidImageError(FaceRecognitionError):
    """Raised when image is invalid, corrupted, or unsupported format."""
    
    def __init__(
        self,
        message: str = "Invalid or unsupported image",
        details: Optional[Any] = None,
    ):
        super().__init__(message, details)


class ImageTooLargeError(InvalidImageError):
    """Raised when image exceeds maximum size limits."""
    
    def __init__(
        self,
        width: int,
        height: int,
        max_width: int,
        max_height: int,
        message: str = "Image exceeds maximum size",
    ):
        self.width = width
        self.height = height
        self.max_width = max_width
        self.max_height = max_height
        details = f"Image: {width}x{height}, Max: {max_width}x{max_height}"
        super().__init__(message, details)


class UnsupportedFormatError(InvalidImageError):
    """Raised when image format is not supported."""
    
    def __init__(
        self,
        format: str,
        supported_formats: tuple = ("jpg", "jpeg", "png"),
        message: str = "Unsupported image format",
    ):
        self.format = format
        self.supported_formats = supported_formats
        details = f"Format: {format}, Supported: {', '.join(supported_formats)}"
        super().__init__(message, details)


class DatabaseConnectionError(FaceRecognitionError):
    """Raised when database connection fails."""
    
    def __init__(
        self,
        message: str = "Database connection failed",
        details: Optional[Any] = None,
    ):
        super().__init__(message, details)


class DatabaseQueryError(FaceRecognitionError):
    """Raised when a database query fails."""
    
    def __init__(
        self,
        message: str = "Database query failed",
        details: Optional[Any] = None,
    ):
        super().__init__(message, details)


class VisitorNotFoundError(FaceRecognitionError):
    """Raised when a visitor is not found in the database."""
    
    def __init__(
        self,
        visitor_id: str,
        message: str = "Visitor not found",
        details: Optional[Any] = None,
    ):
        self.visitor_id = visitor_id
        super().__init__(f"{message}: {visitor_id}", details)


class IndexNotInitializedError(FaceRecognitionError):
    """Raised when HNSW index is not available or not initialized."""
    
    def __init__(
        self,
        message: str = "HNSW index not initialized",
        details: Optional[Any] = None,
    ):
        super().__init__(message, details)


class IndexBuildError(FaceRecognitionError):
    """Raised when HNSW index building fails."""
    
    def __init__(
        self,
        message: str = "Failed to build HNSW index",
        details: Optional[Any] = None,
    ):
        super().__init__(message, details)


class ModelNotLoadedError(FaceRecognitionError):
    """Raised when ML model is not loaded."""
    
    def __init__(
        self,
        model_name: str,
        message: str = "Model not loaded",
        details: Optional[Any] = None,
    ):
        self.model_name = model_name
        super().__init__(f"{message}: {model_name}", details)


class ModelLoadError(FaceRecognitionError):
    """Raised when ML model fails to load."""
    
    def __init__(
        self,
        model_name: str,
        model_path: str,
        message: str = "Failed to load model",
        details: Optional[Any] = None,
    ):
        self.model_name = model_name
        self.model_path = model_path
        super().__init__(f"{message}: {model_name} from {model_path}", details)


__all__ = [
    "FaceRecognitionError",
    "NoFaceDetectedError",
    "MultipleFacesError",
    "FeatureExtractionError",
    "InvalidFeatureError",
    "InvalidImageError",
    "ImageTooLargeError",
    "UnsupportedFormatError",
    "DatabaseConnectionError",
    "DatabaseQueryError",
    "VisitorNotFoundError",
    "IndexNotInitializedError",
    "IndexBuildError",
    "ModelNotLoadedError",
    "ModelLoadError",
]
