"""
Comparison Pipeline

Face comparison business logic using SFace model.
"""

from typing import Optional, Tuple, List
from dataclasses import dataclass

import numpy as np

from core.logger import get_logger
from core.config import settings
from core.exceptions import (
    NoFaceDetectedError,
    FeatureExtractionError,
    InvalidImageError,
)

logger = get_logger(__name__)

# Import ML modules
from ml import inference
from utils import image_loader

# Import other pipelines
from .detection import detect_faces_in_image
from .feature_extraction import extract_single_feature


@dataclass
class ComparisonResult:
    """Result of face comparison operation."""
    success: bool
    similarity_score: float = 0.0
    is_match: bool = False
    threshold: float = 0.55
    error: Optional[str] = None
    feature1: Optional[np.ndarray] = None
    feature2: Optional[np.ndarray] = None


def compare_features(
    feature1: np.ndarray,
    feature2: np.ndarray,
    threshold: Optional[float] = None,
    raise_on_error: bool = False,
) -> ComparisonResult:
    """
    Compare two face feature vectors.
    
    Args:
        feature1: First 128-dim feature vector
        feature2: Second 128-dim feature vector
        threshold: Similarity threshold for match (default from settings)
        raise_on_error: If True, raise exceptions instead of returning error result
    
    Returns:
        ComparisonResult with similarity score
    
    Raises:
        FeatureExtractionError: If features are invalid (when raise_on_error=True)
    """
    if feature1 is None or feature2 is None:
        error_msg = "Both features must be provided"
        if raise_on_error:
            raise FeatureExtractionError(error_msg, details={"feature1": feature1 is not None, "feature2": feature2 is not None})
        return ComparisonResult(success=False, error=error_msg)
    
    threshold = threshold or settings.models.sface_similarity_threshold
    
    try:
        score, is_match = inference.compare_face_features(
            feature1,
            feature2,
            threshold=threshold
        )
        
        return ComparisonResult(
            success=True,
            similarity_score=float(score),
            is_match=bool(is_match),
            threshold=threshold,
        )
        
    except Exception as e:
        logger.error(f"Feature comparison failed: {e}")
        if raise_on_error:
            raise FeatureExtractionError(f"Feature comparison failed: {e}")
        return ComparisonResult(success=False, error=str(e))


def compare_two_faces(
    image1: np.ndarray,
    image2: np.ndarray,
    threshold: Optional[float] = None,
    return_features: bool = False,
    raise_on_error: bool = False,
) -> ComparisonResult:
    """
    Compare faces between two images.
    
    Args:
        image1: First BGR image
        image2: Second BGR image
        threshold: Similarity threshold
        return_features: Whether to include feature vectors in result
        raise_on_error: If True, raise exceptions instead of returning error result
    
    Returns:
        ComparisonResult with match details
    
    Raises:
        NoFaceDetectedError: If no face found in either image (when raise_on_error=True)
    """
    threshold = threshold or settings.models.sface_similarity_threshold
    
    # Detect and extract from first image
    feature1 = extract_single_feature(image1)
    if feature1 is None:
        error_msg = "No face detected in first image"
        if raise_on_error:
            raise NoFaceDetectedError(error_msg, details={"image": "image1"})
        return ComparisonResult(success=False, error=error_msg)
    
    # Detect and extract from second image
    feature2 = extract_single_feature(image2)
    if feature2 is None:
        error_msg = "No face detected in second image"
        if raise_on_error:
            raise NoFaceDetectedError(error_msg, details={"image": "image2"})
        return ComparisonResult(success=False, error=error_msg)
    
    # Compare features
    result = compare_features(feature1, feature2, threshold, raise_on_error)
    
    # Add features to result if requested
    if return_features and result.success:
        result.feature1 = feature1
        result.feature2 = feature2
    
    return result


def compare_from_base64(
    image1_b64: str,
    image2_b64: str,
    threshold: Optional[float] = None,
    return_features: bool = False,
    raise_on_error: bool = False,
) -> ComparisonResult:
    """
    Compare faces from two base64-encoded images.
    
    Args:
        image1_b64: First image as base64 string
        image2_b64: Second image as base64 string
        threshold: Similarity threshold
        return_features: Include features in result
        raise_on_error: If True, raise exceptions instead of returning error result
    
    Returns:
        ComparisonResult
    
    Raises:
        InvalidImageError: If image loading/validation fails (when raise_on_error=True)
        NoFaceDetectedError: If no face found (when raise_on_error=True)
    """
    max_size = settings.image.max_size
    
    try:
        # Load first image
        img1 = image_loader.load_image(image1_b64, source_type="base64")
        image_loader.validate_image_size((img1.shape[1], img1.shape[0]), max_size)
    except ValueError as e:
        error_msg = f"Invalid first image: {e}"
        if raise_on_error:
            raise InvalidImageError(error_msg, details={"image": "image1", "reason": str(e)})
        return ComparisonResult(success=False, error=error_msg)
    except Exception as e:
        error_msg = f"Failed to load first image: {e}"
        if raise_on_error:
            raise InvalidImageError(error_msg, details={"image": "image1", "reason": str(e)})
        return ComparisonResult(success=False, error=error_msg)
    
    try:
        # Load second image
        img2 = image_loader.load_image(image2_b64, source_type="base64")
        image_loader.validate_image_size((img2.shape[1], img2.shape[0]), max_size)
    except ValueError as e:
        error_msg = f"Invalid second image: {e}"
        if raise_on_error:
            raise InvalidImageError(error_msg, details={"image": "image2", "reason": str(e)})
        return ComparisonResult(success=False, error=error_msg)
    except Exception as e:
        error_msg = f"Failed to load second image: {e}"
        if raise_on_error:
            raise InvalidImageError(error_msg, details={"image": "image2", "reason": str(e)})
        return ComparisonResult(success=False, error=error_msg)
    
    return compare_two_faces(img1, img2, threshold, return_features, raise_on_error)


def batch_compare(
    query_feature: np.ndarray,
    candidate_features: List[Tuple[str, np.ndarray]],
    threshold: Optional[float] = None,
) -> List[Tuple[str, float, bool]]:
    """
    Compare one query feature against multiple candidates.
    
    Args:
        query_feature: Query 128-dim feature vector
        candidate_features: List of (id, feature) tuples
        threshold: Similarity threshold
    
    Returns:
        List of (id, similarity, is_match) tuples, sorted by similarity descending
    """
    if query_feature is None:
        return []
    
    threshold = threshold or settings.models.sface_similarity_threshold
    
    results = []
    for candidate_id, candidate_feature in candidate_features:
        if candidate_feature is None:
            continue
        
        try:
            score, is_match = inference.compare_face_features(
                query_feature,
                candidate_feature,
                threshold=threshold
            )
            results.append((candidate_id, float(score), bool(is_match)))
        except Exception as e:
            logger.warning(f"Comparison failed for {candidate_id}: {e}")
    
    # Sort by score descending
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results


__all__ = [
    "ComparisonResult",
    "compare_features",
    "compare_two_faces",
    "compare_from_base64",
    "batch_compare",
]
