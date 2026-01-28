"""
Detection Schemas

Request/response models for face detection and feature extraction endpoints.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class DetectRequest(BaseModel):
    """Request model for /api/v1/detect endpoint."""
    
    image: str = Field(
        ...,
        description="Base64-encoded image data (with or without data URL prefix)"
    )
    score_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for face detection"
    )
    return_landmarks: bool = Field(
        default=False,
        description="Whether to return facial landmarks (5-point)"
    )


class FaceDetection(BaseModel):
    """Single face detection result."""
    
    bbox: List[float] = Field(
        description="Bounding box [x, y, width, height]"
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Detection confidence score"
    )
    landmarks: Optional[List[float]] = Field(
        default=None,
        description="Facial landmarks (10 values: 5 points x 2 coords)"
    )


class DetectionResponse(BaseModel):
    """Response model for /api/v1/detect endpoint."""
    
    faces: List[List[float]] = Field(
        description="List of detected faces (bounding boxes or full data with landmarks)"
    )
    count: int = Field(
        description="Number of faces detected"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "faces": [[100.0, 50.0, 200.0, 200.0]],
                "count": 1
            }
        }


class FeatureExtractionRequest(BaseModel):
    """Request model for /api/v1/extract-features endpoint (JSON body)."""
    
    image: str = Field(
        ...,
        description="Base64-encoded image data"
    )


class FeatureExtractionResponse(BaseModel):
    """Response model for /api/v1/extract-features endpoint."""
    
    features: List[List[float]] = Field(
        description="List of 128-dimensional feature vectors (one per detected face)"
    )
    num_faces: int = Field(
        description="Number of faces for which features were extracted"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [[0.1, -0.2, 0.3, "...128 values..."]],
                "num_faces": 1
            }
        }


__all__ = [
    "DetectRequest",
    "FaceDetection",
    "DetectionResponse",
    "FeatureExtractionRequest",
    "FeatureExtractionResponse",
]
