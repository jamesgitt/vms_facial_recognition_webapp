"""
Pydantic Schemas for Face Recognition API

Contains request/response models for all API endpoints.
"""

from .detection import (
    DetectRequest,
    DetectionResponse,
    FeatureExtractionRequest,
    FeatureExtractionResponse,
)
from .recognition import (
    RecognizeRequest,
    VisitorMatch,
    VisitorRecognitionResponse,
)
from .comparison import (
    CompareRequest,
    CompareResponse,
)
from .common import (
    HealthResponse,
    ModelStatusResponse,
    ModelInfoResponse,
    HNSWStatusResponse,
    ValidateImageRequest,
    ValidateImageResponse,
    ErrorResponse,
)

__all__ = [
    # Detection
    "DetectRequest",
    "DetectionResponse",
    "FeatureExtractionRequest",
    "FeatureExtractionResponse",
    # Recognition
    "RecognizeRequest",
    "VisitorMatch",
    "VisitorRecognitionResponse",
    # Comparison
    "CompareRequest",
    "CompareResponse",
    # Common
    "HealthResponse",
    "ModelStatusResponse",
    "ModelInfoResponse",
    "HNSWStatusResponse",
    "ValidateImageRequest",
    "ValidateImageResponse",
    "ErrorResponse",
]
