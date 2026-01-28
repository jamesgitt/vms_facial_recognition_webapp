"""
API Routes

HTTP endpoints for face detection, recognition, and comparison.
"""

import io
import base64
import datetime

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from PIL import Image

from core.logger import get_logger
from core.config import settings
from core.state import app_state

from schemas import (
    # Detection
    DetectRequest,
    DetectionResponse,
    FaceDetection,
    FeatureExtractionRequest,
    FeatureExtractionResponse,
    # Recognition
    RecognizeRequest,
    VisitorMatch,
    VisitorRecognitionResponse,
    # Comparison
    CompareRequest,
    CompareResponse,
    # Common
    HealthResponse,
    ModelStatusResponse,
    ModelInfo,
    ModelInfoResponse,
    HNSWStatusResponse,
    ValidateImageRequest,
    ValidateImageResponse,
)

from pipelines import (
    detect_faces_in_image,
    extract_features_from_image,
    compare_from_base64,
    recognize_from_image,
)

from utils import image_loader
from ml import inference

from .deps import get_state, require_initialized, get_threshold

logger = get_logger(__name__)

router = APIRouter()


# =============================================================================
# HEALTH ENDPOINTS
# =============================================================================

@router.get("/health", response_model=HealthResponse, tags=["Health"])
@router.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        time=datetime.datetime.utcnow().isoformat() + "Z"
    )


# =============================================================================
# DETECTION ENDPOINTS
# =============================================================================

@router.post("/api/v1/detect", response_model=DetectionResponse, tags=["Detection"])
async def detect_faces_api(request: DetectRequest):
    """
    Detect faces in an image.
    
    Returns bounding boxes and optionally landmarks for all detected faces.
    """
    try:
        # Load and validate image
        img_np = image_loader.load_image(request.image, source_type="base64")
        image_loader.validate_image_size(
            (img_np.shape[1], img_np.shape[0]),
            settings.image.max_size
        )
        
        # Detect faces using pipeline
        result = detect_faces_in_image(
            img_np,
            score_threshold=request.score_threshold,
            return_landmarks=request.return_landmarks,
        )
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)
        
        # Convert to response format
        faces_list = result.to_response_format(include_landmarks=request.return_landmarks)
        
        return DetectionResponse(faces=faces_list, count=result.count)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")


@router.post("/api/v1/detect-structured", tags=["Detection"])
async def detect_faces_structured_api(request: DetectRequest):
    """
    Detect faces in an image with structured response.
    
    Returns FaceDetection objects with bbox, confidence, and landmarks.
    """
    try:
        img_np = image_loader.load_image(request.image, source_type="base64")
        image_loader.validate_image_size(
            (img_np.shape[1], img_np.shape[0]),
            settings.image.max_size
        )
        
        result = detect_faces_in_image(
            img_np,
            score_threshold=request.score_threshold,
            return_landmarks=request.return_landmarks,
        )
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)
        
        # Convert to structured FaceDetection objects
        faces = [
            FaceDetection(
                bbox=list(face.bbox),
                confidence=face.confidence,
                landmarks=face.landmarks,
            )
            for face in result.faces
        ]
        
        return {"faces": [f.model_dump() for f in faces], "count": len(faces)}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")


# =============================================================================
# FEATURE EXTRACTION ENDPOINTS
# =============================================================================

@router.post("/api/v1/extract-features", response_model=FeatureExtractionResponse, tags=["Features"])
async def extract_features_api(
    image: UploadFile = File(None),
    image_base64: str = Form(None),
):
    """
    Extract face feature vectors from an image.
    
    Accepts either file upload or base64 encoded image.
    Returns 128-dimensional feature vectors for all detected faces.
    """
    if image is None and not image_base64:
        raise HTTPException(status_code=400, detail="Image required")
    
    try:
        # Load image
        if image is not None:
            img_np = image_loader.load_from_upload(image)
        else:
            img_np = image_loader.load_image(image_base64, source_type="base64")
        
        image_loader.validate_image_size(
            (img_np.shape[1], img_np.shape[0]),
            settings.image.max_size
        )
        
        # Extract features using pipeline
        result = extract_features_from_image(img_np)
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)
        
        # Convert numpy arrays to lists
        features_list = [
            feat.tolist() if hasattr(feat, 'tolist') else list(feat)
            for feat in result.features
        ]
        
        return FeatureExtractionResponse(
            features=features_list,
            num_faces=result.num_faces
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")


@router.post("/api/v1/extract-features-json", response_model=FeatureExtractionResponse, tags=["Features"])
async def extract_features_json_api(request: FeatureExtractionRequest):
    """
    Extract face feature vectors from an image (JSON body).
    
    Accepts base64-encoded image in JSON request body.
    Returns 128-dimensional feature vectors for all detected faces.
    """
    try:
        img_np = image_loader.load_image(request.image, source_type="base64")
        
        image_loader.validate_image_size(
            (img_np.shape[1], img_np.shape[0]),
            settings.image.max_size
        )
        
        result = extract_features_from_image(img_np)
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)
        
        features_list = [
            feat.tolist() if hasattr(feat, 'tolist') else list(feat)
            for feat in result.features
        ]
        
        return FeatureExtractionResponse(
            features=features_list,
            num_faces=result.num_faces
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")


# =============================================================================
# COMPARISON ENDPOINTS
# =============================================================================

@router.post("/api/v1/compare", response_model=CompareResponse, tags=["Recognition"])
async def compare_faces_api(request: CompareRequest):
    """
    Compare faces between two images.
    
    Returns similarity score and match status based on threshold.
    """
    try:
        # Use comparison pipeline
        result = compare_from_base64(
            request.image1,
            request.image2,
            threshold=request.threshold,
            return_features=False,
        )
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error)
        
        return CompareResponse(
            similarity_score=result.similarity_score,
            is_match=result.is_match,
            threshold=result.threshold,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")


# =============================================================================
# RECOGNITION ENDPOINTS
# =============================================================================

@router.post("/api/v1/recognize", response_model=VisitorRecognitionResponse, tags=["Recognition"])
async def recognize_visitor_api(
    image: UploadFile = File(None),
    image_base64: str = Form(None),
    threshold: float = Form(None),
):
    """
    Recognize a visitor by matching against the database.
    
    Uses HNSW index for fast approximate search, with linear search fallback.
    """
    if image is None and not image_base64:
        raise HTTPException(status_code=400, detail="Image required")
    
    threshold = threshold or settings.models.sface_similarity_threshold
    
    try:
        # Load image
        if image is not None:
            img_np = image_loader.load_from_upload(image)
        else:
            img_np = image_loader.load_image(image_base64, source_type="base64")
        
        image_loader.validate_image_size(
            (img_np.shape[1], img_np.shape[0]),
            settings.image.max_size
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {e}")
    
    # Use recognition pipeline
    result = recognize_from_image(img_np, threshold=threshold, top_k=50)
    
    # Convert to response format using VisitorMatch schema
    matches = [
        VisitorMatch(
            visitor_id=m.visitor_id,
            match_score=m.match_score,
            is_match=m.is_match,
            firstName=m.firstName,
            lastName=m.lastName,
        )
        for m in result.matches[:10]
    ]
    
    return VisitorRecognitionResponse(
        matched=result.matched,
        visitor_id=result.best_match.visitor_id if result.best_match else None,
        confidence=result.best_match.match_score if result.best_match else None,
        firstName=result.best_match.firstName if result.best_match else None,
        lastName=result.best_match.lastName if result.best_match else None,
        matches=[m.model_dump() for m in matches],
    )


@router.post("/api/v1/recognize-json", response_model=VisitorRecognitionResponse, tags=["Recognition"])
async def recognize_visitor_json_api(request: RecognizeRequest):
    """
    Recognize a visitor by matching against the database (JSON body).
    
    Accepts base64-encoded image in JSON request body.
    Uses HNSW index for fast approximate search, with linear search fallback.
    """
    try:
        img_np = image_loader.load_image(request.image, source_type="base64")
        
        image_loader.validate_image_size(
            (img_np.shape[1], img_np.shape[0]),
            settings.image.max_size
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {e}")
    
    # Use recognition pipeline with request parameters
    result = recognize_from_image(
        img_np,
        threshold=request.threshold,
        top_k=request.top_k
    )
    
    # Convert to response format using VisitorMatch schema
    matches = [
        VisitorMatch(
            visitor_id=m.visitor_id,
            match_score=m.match_score,
            is_match=m.is_match,
            firstName=m.firstName,
            lastName=m.lastName,
        )
        for m in result.matches[:request.top_k]
    ]
    
    return VisitorRecognitionResponse(
        matched=result.matched,
        visitor_id=result.best_match.visitor_id if result.best_match else None,
        confidence=result.best_match.match_score if result.best_match else None,
        firstName=result.best_match.firstName if result.best_match else None,
        lastName=result.best_match.lastName if result.best_match else None,
        matches=[m.model_dump() for m in matches],
    )


# =============================================================================
# MODEL STATUS ENDPOINTS
# =============================================================================

@router.get("/models/status", response_model=ModelStatusResponse, tags=["Models"])
async def model_status():
    """Get ML model loading status."""
    loaded = app_state.models_loaded
    
    return ModelStatusResponse(
        loaded=loaded,
        details={
            "face_detector": str(type(app_state.face_detector)),
            "face_recognizer": str(type(app_state.face_recognizer))
        } if loaded else None
    )


@router.get("/models/info", response_model=ModelInfoResponse, tags=["Models"])
async def model_info():
    """Get ML model metadata."""
    return ModelInfoResponse(
        detector=ModelInfo(
            type="YuNet",
            model_path=str(inference.YUNET_PATH),
            input_size=list(inference.YUNET_INPUT_SIZE),
            loaded=app_state.face_detector is not None,
        ),
        recognizer=ModelInfo(
            type="SFace",
            model_path=str(inference.SFACE_PATH),
            similarity_threshold=inference.SFACE_SIMILARITY_THRESHOLD,
            loaded=app_state.face_recognizer is not None,
        ),
    )


# =============================================================================
# HNSW INDEX ENDPOINTS
# =============================================================================

@router.get("/api/v1/hnsw/status", response_model=HNSWStatusResponse, tags=["HNSW"])
async def hnsw_status():
    """Get HNSW index status and statistics."""
    hnsw_manager = app_state.hnsw_manager
    
    if hnsw_manager is None:
        return HNSWStatusResponse(
            available=False,
            initialized=False,
            total_vectors=0,
            dimension=128,
            index_type="HNSW",
            visitors_indexed=0,
            details={"error": "HNSW not initialized"}
        )
    
    try:
        stats = hnsw_manager.get_stats()
        return HNSWStatusResponse(
            available=True,
            initialized=True,
            total_vectors=stats.get('total_vectors', 0),
            dimension=stats.get('dimension', 128),
            index_type=stats.get('index_type', 'HNSW'),
            m=stats.get('m'),
            ef_construction=stats.get('ef_construction'),
            ef_search=stats.get('ef_search'),
            visitors_indexed=stats.get('visitors_indexed', 0),
            details=stats
        )
    except Exception as e:
        return HNSWStatusResponse(
            available=True,
            initialized=False,
            total_vectors=0,
            dimension=128,
            index_type="HNSW",
            visitors_indexed=0,
            details={"error": str(e)}
        )


# =============================================================================
# UTILITY ENDPOINTS
# =============================================================================

def _validate_image_data(image_data: bytes) -> ValidateImageResponse:
    """
    Internal helper to validate image bytes.
    
    Returns:
        ValidateImageResponse with validation result
    """
    try:
        img = Image.open(io.BytesIO(image_data))
        
        fmt = (img.format or "").lower()
        size = img.size
        max_size = settings.image.max_size
        allowed = settings.image.allowed_formats
        
        valid = (
            fmt in allowed and
            size[0] <= max_size[0] and
            size[1] <= max_size[1]
        )
        
        return ValidateImageResponse(valid=valid, format=fmt, size=size)
        
    except Exception:
        return ValidateImageResponse(valid=False, format=None, size=None)


@router.post("/validate-image", response_model=ValidateImageResponse, tags=["Utility"])
async def validate_image_api(
    image: UploadFile = File(None),
    image_base64: str = Form(None),
):
    """
    Validate an image before processing (multipart form).
    
    Accepts file upload or base64 form field.
    Checks format and size constraints.
    """
    if image is None and not image_base64:
        raise HTTPException(status_code=400, detail="Image required")
    
    if image is not None:
        contents = await image.read()
        await image.seek(0)
        return _validate_image_data(contents)
    else:
        # Handle data URI prefix
        if image_base64.startswith('data:'):
            image_base64 = image_base64.split(',', 1)[1]
        img_bytes = base64.b64decode(image_base64)
        return _validate_image_data(img_bytes)


@router.post("/api/v1/validate-image", response_model=ValidateImageResponse, tags=["Utility"])
async def validate_image_json_api(request: ValidateImageRequest):
    """
    Validate an image before processing (JSON body).
    
    Accepts base64-encoded image in JSON request body.
    Checks format and size constraints.
    """
    image_data = request.image
    
    # Handle data URI prefix
    if image_data.startswith('data:'):
        image_data = image_data.split(',', 1)[1]
    
    img_bytes = base64.b64decode(image_data)
    return _validate_image_data(img_bytes)


__all__ = ["router"]
