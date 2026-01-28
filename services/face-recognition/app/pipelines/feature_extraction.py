"""
Feature Extraction Pipeline

Face feature extraction and encoding/decoding utilities.
"""

import base64
import pickle
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

import numpy as np

from core.logger import get_logger
from core.config import settings
from core.exceptions import FeatureExtractionError, InvalidFeatureError

logger = get_logger(__name__)

# Import ML modules
from ml import inference
from utils import image_loader
from db import database


# Constants
FEATURE_DIMENSION = 128


@dataclass
class FeatureExtractionResult:
    """Result of feature extraction operation."""
    success: bool
    features: List[np.ndarray]
    num_faces: int
    error: Optional[str] = None


def extract_features_from_image(
    image: np.ndarray,
    max_faces: Optional[int] = None,
) -> FeatureExtractionResult:
    """
    Extract face features from all faces in an image.
    
    Args:
        image: BGR image as numpy array
        max_faces: Maximum number of faces to process (None = all)
    
    Returns:
        FeatureExtractionResult with extracted features
    """
    try:
        # Detect faces with landmarks (required for feature extraction)
        faces = inference.detect_faces(image, return_landmarks=True)
        
        if faces is None or len(faces) == 0:
            return FeatureExtractionResult(
                success=True,
                features=[],
                num_faces=0
            )
        
        # Limit number of faces if specified
        faces_to_process = faces[:max_faces] if max_faces else faces
        
        features = []
        for face_row in faces_to_process:
            feature = inference.extract_face_features(image, face_row)
            if feature is not None:
                features.append(np.asarray(feature).flatten().astype(np.float32))
        
        return FeatureExtractionResult(
            success=True,
            features=features,
            num_faces=len(features)
        )
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        return FeatureExtractionResult(
            success=False,
            features=[],
            num_faces=0,
            error=str(e)
        )


def extract_single_feature(
    image: np.ndarray,
    face_row: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Extract feature vector from a single face.
    
    Args:
        image: BGR image array
        face_row: Face detection data (if None, will detect first face)
    
    Returns:
        128-dim feature vector or None
    """
    try:
        # Detect face if not provided
        if face_row is None:
            faces = inference.detect_faces(image, return_landmarks=True)
            if faces is None or len(faces) == 0:
                return None
            face_row = faces[0]
        
        # Extract feature
        feature = inference.extract_face_features(image, face_row)
        if feature is None:
            return None
        
        feature = np.asarray(feature).flatten().astype(np.float32)
        
        if feature.shape[0] != FEATURE_DIMENSION:
            logger.warning(f"Invalid feature dimension: {feature.shape[0]}")
            return None
        
        return feature
        
    except Exception as e:
        logger.error(f"Single feature extraction failed: {e}")
        return None


def extract_feature_from_visitor_data(
    visitor_data: Dict[str, Any],
    save_to_db: bool = True,
) -> Optional[np.ndarray]:
    """
    Extract 128-dim feature from visitor data.
    Priority: stored features > extract from image.
    
    Args:
        visitor_data: Dict with visitor info including image or features
        save_to_db: Whether to save extracted features to database
    
    Returns:
        128-dim feature vector or None
    """
    db_config = settings.database
    visitor_id = visitor_data.get(db_config.visitor_id_column, 'unknown')
    
    try:
        # Try stored features first
        stored_features = visitor_data.get(db_config.features_column)
        if stored_features:
            feature = decode_feature_from_base64(stored_features)
            if feature is not None:
                return feature
            logger.warning(f"Failed to decode stored features for {visitor_id}")
        
        # Extract from image
        base64_data = visitor_data.get(db_config.image_column)
        if not base64_data:
            return None
        
        img_cv = image_loader.load_from_base64(base64_data)
        feature = extract_single_feature(img_cv)
        
        if feature is None:
            return None
        
        # Save to database
        if save_to_db and db_config.use_database:
            try:
                database.update_visitor_features(
                    visitor_id=str(visitor_id),
                    features=feature,
                    table_name=db_config.table_name,
                    visitor_id_column=db_config.visitor_id_column,
                    features_column=db_config.features_column
                )
            except Exception as e:
                logger.warning(f"Failed to save features for {visitor_id}: {e}")
        
        return feature
        
    except Exception as e:
        logger.error(f"Feature extraction failed for {visitor_id}: {e}")
        return None


def decode_feature_from_base64(base64_data: str) -> Optional[np.ndarray]:
    """
    Try to decode base64 as a 128-dim feature vector.
    
    Attempts multiple formats:
    1. Raw float32 bytes (512 bytes = 128 * 4)
    2. Pickled numpy array
    
    Args:
        base64_data: Base64-encoded feature data
    
    Returns:
        128-dim feature vector or None
    """
    if not base64_data:
        return None
    
    try:
        feature_bytes = base64.b64decode(base64_data)
        
        # Try raw float32 bytes first (most common)
        if len(feature_bytes) == FEATURE_DIMENSION * 4:
            feature_array = np.frombuffer(feature_bytes, dtype=np.float32)
            if feature_array.shape[0] == FEATURE_DIMENSION:
                return feature_array.astype(np.float32)
        
        # Try pickle format
        try:
            feature_array = np.asarray(pickle.loads(feature_bytes)).flatten()
            if feature_array.shape[0] == FEATURE_DIMENSION:
                return feature_array.astype(np.float32)
        except Exception:
            pass
        
    except Exception:
        pass
    
    return None


def encode_feature_to_base64(feature: np.ndarray) -> str:
    """
    Encode feature vector to base64 string.
    
    Args:
        feature: 128-dim feature vector
    
    Returns:
        Base64-encoded string
    """
    feature_bytes = feature.astype(np.float32).tobytes()
    return base64.b64encode(feature_bytes).decode('utf-8')


def validate_feature(feature: np.ndarray) -> bool:
    """
    Validate that a feature vector is valid.
    
    Args:
        feature: Feature vector to validate
    
    Returns:
        True if valid, False otherwise
    """
    if feature is None:
        return False
    
    if not isinstance(feature, np.ndarray):
        return False
    
    feature = feature.flatten()
    
    if feature.shape[0] != FEATURE_DIMENSION:
        return False
    
    # Check for NaN or Inf
    if np.isnan(feature).any() or np.isinf(feature).any():
        return False
    
    # Check for zero vector
    if np.allclose(feature, 0):
        return False
    
    return True


__all__ = [
    "FEATURE_DIMENSION",
    "FeatureExtractionResult",
    "extract_features_from_image",
    "extract_single_feature",
    "extract_feature_from_visitor_data",
    "decode_feature_from_base64",
    "encode_feature_to_base64",
    "validate_feature",
]
