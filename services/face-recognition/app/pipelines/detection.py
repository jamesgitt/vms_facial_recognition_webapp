"""
Detection Pipeline

Face detection business logic using YuNet model.
"""

from typing import List, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np

from core.logger import get_logger
from core.config import settings
from core.exceptions import NoFaceDetectedError

logger = get_logger(__name__)

# Import ML modules
from ml import inference
from utils import image_loader


@dataclass
class FaceDetection:
    """Single face detection result."""
    bbox: Tuple[float, float, float, float]  # x, y, width, height
    confidence: Optional[float] = None
    landmarks: Optional[List[float]] = None
    raw_data: Optional[np.ndarray] = None


@dataclass
class DetectionResult:
    """Result of face detection operation."""
    success: bool
    faces: List[FaceDetection] = field(default_factory=list)
    count: int = 0
    error: Optional[str] = None
    
    @property
    def has_faces(self) -> bool:
        return self.count > 0
    
    def to_response_format(self, include_landmarks: bool = False) -> List[List[float]]:
        """Convert to API response format."""
        result = []
        for face in self.faces:
            if include_landmarks and face.raw_data is not None:
                result.append(face.raw_data.tolist() if hasattr(face.raw_data, 'tolist') else list(face.raw_data))
            else:
                result.append(list(face.bbox))
        return result


def detect_faces_in_image(
    image: np.ndarray,
    score_threshold: Optional[float] = None,
    return_landmarks: bool = False,
) -> DetectionResult:
    """
    Detect faces in an image using YuNet.
    
    Args:
        image: BGR image as numpy array
        score_threshold: Detection confidence threshold (default from settings)
        return_landmarks: Whether to include facial landmarks
    
    Returns:
        DetectionResult with detected faces
    """
    if image is None:
        return DetectionResult(
            success=False,
            error="Image is None"
        )
    
    threshold = score_threshold or settings.models.yunet_score_threshold
    
    try:
        results = inference.detect_faces(
            image,
            score_threshold=threshold,
            return_landmarks=return_landmarks,
        )
        
        if results is None or len(results) == 0:
            return DetectionResult(success=True, faces=[], count=0)
        
        faces = []
        for r in results:
            # Extract bounding box
            bbox = (float(r[0]), float(r[1]), float(r[2]), float(r[3]))
            
            # Extract confidence if available
            confidence = float(r[4]) if len(r) > 4 else None
            
            # Extract landmarks if available and requested
            landmarks = None
            if return_landmarks and len(r) > 14:
                landmarks = [float(x) for x in r[5:15]]
            
            faces.append(FaceDetection(
                bbox=bbox,
                confidence=confidence,
                landmarks=landmarks,
                raw_data=r if return_landmarks else None,
            ))
        
        return DetectionResult(
            success=True,
            faces=faces,
            count=len(faces),
        )
        
    except Exception as e:
        logger.error(f"Face detection failed: {e}")
        return DetectionResult(
            success=False,
            error=str(e)
        )


def load_and_detect(
    image_data: str,
    source_type: str = "base64",
    score_threshold: Optional[float] = None,
    return_landmarks: bool = False,
    validate_size: bool = True,
) -> Tuple[DetectionResult, Optional[np.ndarray]]:
    """
    Load image and detect faces.
    
    Args:
        image_data: Image data (base64 string, file path, or URL)
        source_type: Type of image source
        score_threshold: Detection threshold
        return_landmarks: Include landmarks in result
        validate_size: Validate image size limits
    
    Returns:
        Tuple of (DetectionResult, loaded_image or None)
    """
    try:
        # Load image
        img = image_loader.load_image(image_data, source_type=source_type)
        
        # Validate size
        if validate_size:
            max_size = settings.image.max_size
            image_loader.validate_image_size((img.shape[1], img.shape[0]), max_size)
        
        # Detect faces
        result = detect_faces_in_image(
            img,
            score_threshold=score_threshold,
            return_landmarks=return_landmarks,
        )
        
        return result, img
        
    except ValueError as e:
        return DetectionResult(success=False, error=f"Invalid image: {e}"), None
    except Exception as e:
        logger.error(f"Load and detect failed: {e}")
        return DetectionResult(success=False, error=str(e)), None


def require_single_face(
    image: np.ndarray,
    score_threshold: Optional[float] = None,
) -> Tuple[np.ndarray, FaceDetection]:
    """
    Detect faces and ensure exactly one face is found.
    
    Args:
        image: BGR image array
        score_threshold: Detection threshold
    
    Returns:
        Tuple of (image, face_detection)
    
    Raises:
        NoFaceDetectedError: If no face found
    """
    result = detect_faces_in_image(
        image,
        score_threshold=score_threshold,
        return_landmarks=True,
    )
    
    if not result.success:
        raise NoFaceDetectedError(result.error or "Detection failed")
    
    if result.count == 0:
        raise NoFaceDetectedError("No face detected in image")
    
    return image, result.faces[0]


__all__ = [
    "FaceDetection",
    "DetectionResult",
    "detect_faces_in_image",
    "load_and_detect",
    "require_single_face",
]
