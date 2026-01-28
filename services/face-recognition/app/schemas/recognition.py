"""
Recognition Schemas

Request/response models for visitor recognition endpoints.
"""

from typing import List, Optional, Any
from pydantic import BaseModel, Field


class RecognizeRequest(BaseModel):
    """Request model for /api/v1/recognize endpoint (JSON body)."""
    
    image: str = Field(
        ...,
        description="Base64-encoded image data"
    )
    threshold: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for a match"
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of matches to return"
    )


class VisitorMatch(BaseModel):
    """Single visitor match result."""
    
    visitor_id: str = Field(
        description="Unique visitor identifier"
    )
    match_score: float = Field(
        description="Cosine similarity score (0.0 to 1.0)"
    )
    is_match: bool = Field(
        description="Whether score exceeds threshold"
    )
    firstName: Optional[str] = Field(
        default=None,
        description="Visitor's first name"
    )
    lastName: Optional[str] = Field(
        default=None,
        description="Visitor's last name"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "visitor_id": "abc-123",
                "match_score": 0.87,
                "is_match": True,
                "firstName": "John",
                "lastName": "Doe"
            }
        }


class VisitorRecognitionResponse(BaseModel):
    """Response model for /api/v1/recognize endpoint."""
    
    # Primary result
    matched: bool = Field(
        default=False,
        description="Whether a matching visitor was found"
    )
    visitor_id: Optional[str] = Field(
        default=None,
        description="ID of best matching visitor (if matched)"
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence score of best match"
    )
    firstName: Optional[str] = Field(
        default=None,
        description="First name of matched visitor"
    )
    lastName: Optional[str] = Field(
        default=None,
        description="Last name of matched visitor"
    )
    
    # Top matches list
    matches: Optional[List[Any]] = Field(
        default=None,
        description="Top-K matching visitors sorted by score"
    )
    
    # Legacy fields (for backward compatibility)
    visitor: Optional[str] = Field(
        default=None,
        description="[Deprecated] Use visitor_id instead"
    )
    match_score: Optional[float] = Field(
        default=None,
        description="[Deprecated] Use confidence instead"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "matched": True,
                "visitor_id": "abc-123",
                "confidence": 0.87,
                "firstName": "John",
                "lastName": "Doe",
                "matches": [
                    {
                        "visitor_id": "abc-123",
                        "match_score": 0.87,
                        "is_match": True
                    }
                ]
            }
        }


__all__ = [
    "RecognizeRequest",
    "VisitorMatch",
    "VisitorRecognitionResponse",
]
