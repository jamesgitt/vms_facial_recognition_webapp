"""
Recognition Pipeline

Visitor recognition using HNSW index or linear search fallback.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

import numpy as np

from core.logger import get_logger
from core.config import settings
from core.state import app_state

logger = get_logger(__name__)

# Import ML modules
from ml import inference
from utils import image_loader
from db import database

# Import feature extraction
from .feature_extraction import (
    extract_single_feature,
    decode_feature_from_base64,
)


@dataclass
class VisitorMatch:
    """Single visitor match result."""
    visitor_id: str
    match_score: float
    is_match: bool
    firstName: Optional[str] = None
    lastName: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "visitor_id": self.visitor_id,
            "match_score": self.match_score,
            "is_match": self.is_match,
            "firstName": self.firstName,
            "lastName": self.lastName,
        }


@dataclass
class RecognitionResult:
    """Result of visitor recognition operation."""
    success: bool
    matched: bool = False
    best_match: Optional[VisitorMatch] = None
    matches: List[VisitorMatch] = field(default_factory=list)
    error: Optional[str] = None
    search_method: str = "none"  # "hnsw", "linear_db", "linear_cache"
    
    def to_response(self, top_k: int = 10) -> Dict[str, Any]:
        """Convert to API response format."""
        if self.matched and self.best_match:
            return {
                "matched": True,
                "visitor_id": self.best_match.visitor_id,
                "confidence": self.best_match.match_score,
                "firstName": self.best_match.firstName,
                "lastName": self.best_match.lastName,
                "visitor": self.best_match.visitor_id,  # Legacy
                "match_score": self.best_match.match_score,  # Legacy
                "matches": [m.to_dict() for m in self.matches[:top_k]],
            }
        return {
            "matched": False,
            "matches": [m.to_dict() for m in self.matches[:top_k]] if self.matches else [],
        }


def recognize_visitor(
    query_feature: np.ndarray,
    threshold: Optional[float] = None,
    top_k: int = 50,
) -> RecognitionResult:
    """
    Recognize visitor by matching feature against indexed visitors.
    
    Uses HNSW index for fast search if available, otherwise falls back
    to linear search.
    
    Args:
        query_feature: 128-dim query feature vector
        threshold: Similarity threshold for match
        top_k: Number of top matches to return
    
    Returns:
        RecognitionResult with matches
    """
    if query_feature is None:
        return RecognitionResult(
            success=False,
            error="Query feature is None"
        )
    
    threshold = threshold or settings.models.sface_similarity_threshold
    
    # Try HNSW search first
    result = _search_hnsw(query_feature, threshold, top_k)
    if result.matches:
        return result
    
    # Fallback: Linear search against database
    if app_state.use_database:
        result = _search_linear_database(query_feature, threshold, top_k)
        if result.matches:
            return result
    
    # Fallback: Linear search against cached features
    result = _search_linear_cache(query_feature, threshold)
    
    return result


def recognize_from_image(
    image: np.ndarray,
    threshold: Optional[float] = None,
    top_k: int = 50,
) -> RecognitionResult:
    """
    Recognize visitor from an image.
    
    Args:
        image: BGR image array
        threshold: Similarity threshold
        top_k: Number of top matches
    
    Returns:
        RecognitionResult
    """
    # Extract feature from query image
    query_feature = extract_single_feature(image)
    
    if query_feature is None:
        return RecognitionResult(
            success=True,
            matched=False,
            error="No face detected or feature extraction failed"
        )
    
    return recognize_visitor(query_feature, threshold, top_k)


def recognize_from_base64(
    image_b64: str,
    threshold: Optional[float] = None,
    top_k: int = 50,
) -> RecognitionResult:
    """
    Recognize visitor from base64-encoded image.
    
    Args:
        image_b64: Base64-encoded image
        threshold: Similarity threshold
        top_k: Number of top matches
    
    Returns:
        RecognitionResult
    """
    try:
        img = image_loader.load_image(image_b64, source_type="base64")
        max_size = settings.image.max_size
        image_loader.validate_image_size((img.shape[1], img.shape[0]), max_size)
    except ValueError as e:
        return RecognitionResult(success=False, error=f"Invalid image: {e}")
    except Exception as e:
        return RecognitionResult(success=False, error=f"Failed to load image: {e}")
    
    return recognize_from_image(img, threshold, top_k)


def _search_hnsw(
    query_feature: np.ndarray,
    threshold: float,
    top_k: int,
) -> RecognitionResult:
    """Search using HNSW index."""
    hnsw_manager = app_state.hnsw_manager
    
    if hnsw_manager is None or hnsw_manager.ntotal == 0:
        return RecognitionResult(
            success=True,
            matched=False,
            search_method="hnsw"
        )
    
    try:
        ann_results = hnsw_manager.search(query_feature, k=top_k)
        
        matches = []
        best_match = None
        best_score = 0.0
        
        for visitor_id, similarity, metadata in ann_results:
            is_match = similarity >= threshold
            match = VisitorMatch(
                visitor_id=visitor_id,
                match_score=similarity,
                is_match=is_match,
                firstName=metadata.get('firstName'),
                lastName=metadata.get('lastName'),
                metadata=metadata,
            )
            matches.append(match)
            
            if is_match and similarity > best_score:
                best_score = similarity
                best_match = match
        
        return RecognitionResult(
            success=True,
            matched=best_match is not None,
            best_match=best_match,
            matches=matches,
            search_method="hnsw"
        )
        
    except Exception as e:
        logger.warning(f"HNSW search error: {e}")
        return RecognitionResult(
            success=True,
            matched=False,
            error=str(e),
            search_method="hnsw"
        )


def _search_linear_database(
    query_feature: np.ndarray,
    threshold: float,
    limit: Optional[int] = None,
) -> RecognitionResult:
    """Linear search against database visitors."""
    db_config = settings.database
    
    try:
        visitors = database.get_visitor_images_from_db(
            table_name=db_config.table_name,
            visitor_id_column=db_config.visitor_id_column,
            image_column=db_config.image_column,
            limit=limit or db_config.visitor_limit,
        )
        
        matches = []
        best_match = None
        best_score = 0.0
        
        for visitor_data in visitors:
            visitor_id = str(visitor_data.get(db_config.visitor_id_column))
            base64_image = visitor_data.get(db_config.image_column)
            
            if not base64_image:
                continue
            
            # Try to decode as feature vector
            db_feature = decode_feature_from_base64(base64_image)
            
            # Otherwise extract from image
            if db_feature is None:
                try:
                    img_cv = image_loader.load_from_base64(base64_image)
                    db_feature = extract_single_feature(img_cv)
                    if db_feature is None:
                        continue
                except Exception:
                    continue
            
            # Compare features
            try:
                score, is_match_result = inference.compare_face_features(
                    query_feature,
                    db_feature,
                    threshold=threshold
                )
            except Exception:
                continue
            
            match = VisitorMatch(
                visitor_id=visitor_id,
                match_score=float(score),
                is_match=bool(is_match_result),
                firstName=visitor_data.get('firstName'),
                lastName=visitor_data.get('lastName'),
            )
            matches.append(match)
            
            if is_match_result and score > best_score:
                best_score = score
                best_match = match
        
        # Sort by score
        matches.sort(key=lambda x: x.match_score, reverse=True)
        
        return RecognitionResult(
            success=True,
            matched=best_match is not None,
            best_match=best_match,
            matches=matches,
            search_method="linear_db"
        )
        
    except Exception as e:
        logger.warning(f"Database search error: {e}")
        return RecognitionResult(
            success=True,
            matched=False,
            error=str(e),
            search_method="linear_db"
        )


def _search_linear_cache(
    query_feature: np.ndarray,
    threshold: float,
) -> RecognitionResult:
    """Linear search against in-memory cached features."""
    visitor_features = app_state.visitor_features
    
    if not visitor_features:
        return RecognitionResult(
            success=True,
            matched=False,
            search_method="linear_cache"
        )
    
    matches = []
    best_match = None
    best_score = 0.0
    
    for visitor_name, visitor_data in visitor_features.items():
        db_feature = visitor_data.get("feature")
        if db_feature is None:
            continue
        
        try:
            score, is_match_result = inference.compare_face_features(
                query_feature,
                db_feature,
                threshold=threshold
            )
        except Exception:
            continue
        
        match = VisitorMatch(
            visitor_id=visitor_name,
            match_score=float(score),
            is_match=bool(is_match_result),
        )
        matches.append(match)
        
        if is_match_result and score > best_score:
            best_score = score
            best_match = match
    
    # Sort by score
    matches.sort(key=lambda x: x.match_score, reverse=True)
    
    return RecognitionResult(
        success=True,
        matched=best_match is not None,
        best_match=best_match,
        matches=matches,
        search_method="linear_cache"
    )


__all__ = [
    "VisitorMatch",
    "RecognitionResult",
    "recognize_visitor",
    "recognize_from_image",
    "recognize_from_base64",
]
