"""
Comparison Schemas

Request/response models for face comparison endpoints.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class CompareRequest(BaseModel):
    """Request model for /api/v1/compare endpoint."""
    
    image1: str = Field(
        ...,
        description="Base64-encoded first image"
    )
    image2: str = Field(
        ...,
        description="Base64-encoded second image"
    )
    threshold: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for match determination"
    )
    return_features: bool = Field(
        default=False,
        description="Whether to include feature vectors in response"
    )


class CompareResponse(BaseModel):
    """Response model for /api/v1/compare endpoint."""
    
    similarity_score: float = Field(
        description="Cosine similarity between the two faces (0.0 to 1.0)"
    )
    is_match: bool = Field(
        description="Whether similarity exceeds threshold"
    )
    features1: Optional[List[float]] = Field(
        default=None,
        description="128-dim feature vector of first face (if return_features=True)"
    )
    features2: Optional[List[float]] = Field(
        default=None,
        description="128-dim feature vector of second face (if return_features=True)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "similarity_score": 0.85,
                "is_match": True,
                "features1": None,
                "features2": None
            }
        }


__all__ = [
    "CompareRequest",
    "CompareResponse",
]
