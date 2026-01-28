"""
Common Schemas

Shared request/response models for health, status, and utility endpoints.
"""

from typing import Optional, Any, Tuple, Dict
from datetime import datetime
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response model for /health and /api/v1/health endpoints."""
    
    status: str = Field(
        default="ok",
        description="Service health status"
    )
    time: str = Field(
        description="Current server time (ISO 8601 format)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "time": "2024-01-15T10:30:00Z"
            }
        }


class ModelStatusResponse(BaseModel):
    """Response model for /models/status endpoint."""
    
    loaded: bool = Field(
        description="Whether ML models are loaded and ready"
    )
    details: Optional[Any] = Field(
        default=None,
        description="Additional details about loaded models"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "loaded": True,
                "details": {
                    "face_detector": "<class 'cv2.FaceDetectorYN'>",
                    "face_recognizer": "<class 'cv2.FaceRecognizerSF'>"
                }
            }
        }


class ModelInfo(BaseModel):
    """Information about a single ML model."""
    
    type: str = Field(description="Model type (e.g., 'YuNet', 'SFace')")
    model_path: str = Field(description="Path to model file")
    loaded: bool = Field(description="Whether model is loaded")
    input_size: Optional[Tuple[int, int]] = Field(
        default=None,
        description="Model input size (for detector)"
    )
    similarity_threshold: Optional[float] = Field(
        default=None,
        description="Similarity threshold (for recognizer)"
    )


class ModelInfoResponse(BaseModel):
    """Response model for /models/info endpoint."""
    
    detector: Any = Field(
        description="Face detector model information"
    )
    recognizer: Any = Field(
        description="Face recognizer model information"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "detector": {
                    "type": "YuNet",
                    "model_path": "models/face_detection_yunet_2023mar.onnx",
                    "input_size": [640, 640],
                    "loaded": True
                },
                "recognizer": {
                    "type": "SFace",
                    "model_path": "models/face_recognition_sface_2021dec.onnx",
                    "similarity_threshold": 0.55,
                    "loaded": True
                }
            }
        }


class HNSWStatusResponse(BaseModel):
    """Response model for /api/v1/hnsw/status endpoint."""
    
    available: bool = Field(
        description="Whether HNSW library is available"
    )
    initialized: bool = Field(
        description="Whether index is initialized"
    )
    total_vectors: int = Field(
        description="Number of vectors in index"
    )
    dimension: int = Field(
        default=128,
        description="Feature vector dimension"
    )
    index_type: str = Field(
        default="HNSW",
        description="Index type"
    )
    m: Optional[int] = Field(
        default=None,
        description="HNSW M parameter (bi-directional links)"
    )
    ef_construction: Optional[int] = Field(
        default=None,
        description="HNSW ef_construction parameter"
    )
    ef_search: Optional[int] = Field(
        default=None,
        description="HNSW ef_search parameter"
    )
    visitors_indexed: int = Field(
        description="Number of visitors indexed"
    )
    details: Optional[Any] = Field(
        default=None,
        description="Additional index details"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "available": True,
                "initialized": True,
                "total_vectors": 5000,
                "dimension": 128,
                "index_type": "HNSW",
                "m": 32,
                "ef_construction": 400,
                "ef_search": 400,
                "visitors_indexed": 5000
            }
        }


class ValidateImageRequest(BaseModel):
    """Request model for /validate-image endpoint (JSON body)."""
    
    image: str = Field(
        ...,
        description="Base64-encoded image data"
    )


class ValidateImageResponse(BaseModel):
    """Response model for /validate-image endpoint."""
    
    valid: bool = Field(
        description="Whether image is valid for processing"
    )
    format: Optional[str] = Field(
        default=None,
        description="Detected image format (e.g., 'jpeg', 'png')"
    )
    size: Optional[Tuple[int, int]] = Field(
        default=None,
        description="Image dimensions (width, height)"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if validation failed"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "valid": True,
                "format": "jpeg",
                "size": [640, 480],
                "error": None
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response model."""
    
    error: str = Field(
        description="Error message"
    )
    type: Optional[str] = Field(
        default=None,
        description="Error type/category"
    )
    details: Optional[Any] = Field(
        default=None,
        description="Additional error details"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "No face detected in image",
                "type": "NoFaceDetectedError",
                "details": None
            }
        }


class WebSocketMessage(BaseModel):
    """WebSocket message model for real-time detection."""
    
    type: str = Field(
        description="Message type: 'frame', 'results', 'error'"
    )
    image: Optional[str] = Field(
        default=None,
        description="Base64 image (for 'frame' type)"
    )
    faces: Optional[list] = Field(
        default=None,
        description="Detection results (for 'results' type)"
    )
    count: Optional[int] = Field(
        default=None,
        description="Face count (for 'results' type)"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message (for 'error' type)"
    )
    score_threshold: Optional[float] = Field(
        default=None,
        description="Detection threshold (for 'frame' type)"
    )
    return_landmarks: Optional[bool] = Field(
        default=None,
        description="Return landmarks flag (for 'frame' type)"
    )


__all__ = [
    "HealthResponse",
    "ModelStatusResponse",
    "ModelInfo",
    "ModelInfoResponse",
    "HNSWStatusResponse",
    "ValidateImageRequest",
    "ValidateImageResponse",
    "ErrorResponse",
    "WebSocketMessage",
]
