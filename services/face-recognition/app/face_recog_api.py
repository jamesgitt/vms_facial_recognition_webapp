"""
Face Recognition API

REST API for face detection and recognition using YuNet and SFace models.

Endpoints:
- POST /api/v1/detect           : Detect faces in an image
- POST /api/v1/extract-features : Extract face feature vectors
- POST /api/v1/compare          : Compare faces between two images
- POST /api/v1/recognize        : Recognize visitor from database
- GET  /api/v1/health           : Health check
- GET  /api/v1/hnsw/status      : HNSW index status
- GET  /models/status           : Model loading status
- GET  /models/info             : Model metadata
- POST /validate-image          : Validate image before processing
- WS   /ws/realtime             : Real-time face detection via websocket
"""

import os
import io
import json
import base64
import pickle
import datetime
from typing import List, Tuple, Optional, Any
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image

# Load environment variables
try:
    from dotenv import load_dotenv
    _SCRIPT_DIR = Path(__file__).parent
    for env_path in [_SCRIPT_DIR.parent / ".env.test", _SCRIPT_DIR / ".env.test"]:
        if env_path.exists():
            load_dotenv(env_path)
            break
except ImportError:
    pass

import inference
import image_loader

# Optional database integration
try:
    import database
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    print("[WARNING] Database module not available. Using test images fallback.")

# Optional HNSW index
try:
    from hnsw_index import HNSWIndexManager
    HNSW_AVAILABLE = True
except ImportError:
    HNSW_AVAILABLE = False
    HNSWIndexManager = None
    print("[WARNING] HNSW index not available. Using linear search.")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Thresholds
DEFAULT_SCORE_THRESHOLD = 0.7
DEFAULT_COMPARE_THRESHOLD = 0.550

# Image limits
MAX_IMAGE_SIZE = (1920, 1920)
ALLOWED_FORMATS = frozenset({"jpg", "jpeg", "png"})

# Paths
MODELS_PATH = os.environ.get("MODELS_PATH", "models")
VISITOR_IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_images")

# CORS
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",") if os.environ.get("CORS_ORIGINS") else ["*"]

# Database configuration
USE_DATABASE = os.environ.get("USE_DATABASE", "false").lower() == "true"
DB_TABLE_NAME = os.environ.get("DB_TABLE_NAME", 'public."Visitor"')
DB_VISITOR_ID_COLUMN = os.environ.get("DB_VISITOR_ID_COLUMN", "id")
DB_IMAGE_COLUMN = os.environ.get("DB_IMAGE_COLUMN", "base64Image")
DB_FEATURES_COLUMN = os.environ.get("DB_FEATURES_COLUMN", "facefeatures")
DB_VISITOR_LIMIT = int(os.environ.get("DB_VISITOR_LIMIT", "0")) or None


# =============================================================================
# GLOBAL STATE
# =============================================================================

face_detector = None
face_recognizer = None
hnsw_index_manager = None
VISITOR_FEATURES = {}  # {visitor_name: {'feature': np.array, 'path': str}}


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Face Recognition API",
    version="1.0.0",
    description="REST API for face detection and recognition using YuNet and SFace models.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_feature_from_visitor_data(visitor_data: dict) -> Optional[np.ndarray]:
    """
    Extract 128-dim feature from visitor data.
    Priority: stored features > extract from image.
    """
    try:
        visitor_id = visitor_data.get(DB_VISITOR_ID_COLUMN, 'unknown')
        
        # Try stored features first
        stored_features = visitor_data.get(DB_FEATURES_COLUMN)
        if stored_features:
            try:
                feature_bytes = base64.b64decode(stored_features)
                feature_array = np.frombuffer(feature_bytes, dtype=np.float32)
                if feature_array.shape[0] == 128:
                    return feature_array.astype(np.float32)
            except Exception as e:
                print(f"[WARNING] Error decoding features for {visitor_id}: {e}")
        
        # Extract from image
        base64_data = visitor_data.get(DB_IMAGE_COLUMN)
        if not base64_data:
            return None
        
        img_cv = image_loader.load_from_base64(base64_data)
        faces = inference.detect_faces(img_cv, return_landmarks=True)
        if faces is None or len(faces) == 0:
            return None
        
        feature = inference.extract_face_features(img_cv, faces[0])
        if feature is None:
            return None
        
        feature = np.asarray(feature).flatten().astype(np.float32)
        if feature.shape[0] != 128:
            return None
        
        # Save to database
        if USE_DATABASE and DB_AVAILABLE:
            try:
                database.update_visitor_features(
                    visitor_id=str(visitor_id),
                    features=feature,
                    table_name=DB_TABLE_NAME,
                    visitor_id_column=DB_VISITOR_ID_COLUMN,
                    features_column=DB_FEATURES_COLUMN
                )
            except Exception:
                pass
        
        return feature
    except Exception as e:
        print(f"[ERROR] Feature extraction failed for {visitor_data.get(DB_VISITOR_ID_COLUMN, 'unknown')}: {e}")
        return None


def decode_feature_from_base64(base64_data: str) -> Optional[np.ndarray]:
    """Try to decode base64 as a 128-dim feature vector."""
    try:
        feature_bytes = base64.b64decode(base64_data)
        
        # Try pickle first
        try:
            feature_array = np.asarray(pickle.loads(feature_bytes)).flatten()
        except Exception:
            # Try raw float32 bytes
            if len(feature_bytes) == 128 * 4:
                feature_array = np.frombuffer(feature_bytes, dtype=np.float32)
            else:
                return None
        
        if feature_array.shape[0] == 128:
            return feature_array.astype(np.float32)
    except Exception:
        pass
    return None


def init_database_connection() -> bool:
    """Initialize database connection. Returns True if successful."""
    global USE_DATABASE
    
    if not DB_AVAILABLE:
        if USE_DATABASE:
            print("[WARNING] Database module not available - using test_images")
        USE_DATABASE = False
        return False
    
    try:
        print("Connecting to database...")
        if database.test_connection():
            print("[OK] Database connection successful")
            database.init_connection_pool(min_conn=1, max_conn=5)
            USE_DATABASE = True
            return True
        else:
            print("[WARNING] Database connection failed - using test_images")
            USE_DATABASE = False
            return False
    except Exception as e:
        print(f"[WARNING] Database error: {e} - using test_images")
        USE_DATABASE = False
        return False


def init_hnsw_index() -> Optional[Any]:
    """Initialize HNSW index manager."""
    if not HNSW_AVAILABLE or HNSWIndexManager is None:
        return None
    
    try:
        index_dir = os.environ.get("HNSW_INDEX_DIR", MODELS_PATH)
        max_elements = int(os.environ.get("HNSW_MAX_ELEMENTS", "100000"))
        manager = HNSWIndexManager(index_dir=index_dir, max_elements=max_elements)
        print(f"[OK] HNSW index initialized (max_elements={max_elements})")
        return manager
    except Exception as e:
        print(f"[WARNING] HNSW init error: {e}")
        return None


def load_visitors_from_database(manager: Optional[Any]) -> int:
    """Load visitors from database and build HNSW index. Returns count."""
    try:
        visitors = database.get_visitor_images_from_db(
            table_name=DB_TABLE_NAME,
            visitor_id_column=DB_VISITOR_ID_COLUMN,
            image_column=DB_IMAGE_COLUMN,
            features_column=DB_FEATURES_COLUMN,
            limit=DB_VISITOR_LIMIT
        )
        print(f"[OK] Found {len(visitors)} visitors in database")
        
        if manager and visitors:
            print(f"Building HNSW index from {len(visitors)} visitors...")
            count = manager.rebuild_from_database(
                get_visitors_func=lambda: visitors,
                extract_feature_func=extract_feature_from_visitor_data
            )
            if count > 0:
                print(f"[OK] HNSW index built with {count} visitors")
            return count
        return len(visitors)
    except Exception as e:
        print(f"[ERROR] Loading visitors: {e}")
        return 0


def load_visitors_from_test_images(manager: Optional[Any]) -> int:
    """Load visitors from test_images directory. Returns count."""
    if not os.path.isdir(VISITOR_IMAGES_DIR):
        return 0
    
    print(f"Loading visitors from {VISITOR_IMAGES_DIR}")
    batch_data = []
    
    for fname in os.listdir(VISITOR_IMAGES_DIR):
        if not fname.lower().endswith(tuple(ALLOWED_FORMATS)):
            continue
        
        fpath = os.path.join(VISITOR_IMAGES_DIR, fname)
        try:
            img_cv = image_loader.load_from_path(fpath)
            faces = inference.detect_faces(img_cv, return_landmarks=True)
            if faces is None or len(faces) == 0:
                continue
            
            feature = inference.extract_face_features(img_cv, faces[0])
            if feature is not None:
                visitor_name = os.path.splitext(fname)[0]
                VISITOR_FEATURES[visitor_name] = {"feature": feature, "path": fpath}
                if manager:
                    batch_data.append((visitor_name, feature, {"path": fpath}))
        except Exception as e:
            print(f"[WARNING] Failed to process {fname}: {e}")
    
    print(f"[OK] Loaded {len(VISITOR_FEATURES)} visitors from test_images")
    
    if manager and batch_data:
        count = manager.add_visitors_batch(batch_data)
        if count > 0:
            manager.save()
            print(f"[OK] HNSW index built with {count} test_images visitors")
    
    return len(VISITOR_FEATURES)


# =============================================================================
# STARTUP
# =============================================================================

@app.on_event("startup")
def load_models():
    """Initialize models and load visitor data on startup."""
    global face_detector, face_recognizer, hnsw_index_manager, USE_DATABASE
    
    # Load ML models
    try:
        face_detector = inference.get_face_detector()
        face_recognizer = inference.get_face_recognizer()
        print("[OK] Models loaded")
    except Exception as e:
        print(f"[ERROR] Loading models: {e}")
        face_detector = None
        face_recognizer = None
    
    # Initialize HNSW index
    hnsw_index_manager = init_hnsw_index()
    
    # Initialize database
    init_database_connection()
    
    # Load visitors
    VISITOR_FEATURES.clear()
    
    # Check if HNSW index already has data (skip expensive database reload)
    index_has_data = False
    if hnsw_index_manager:
        try:
            count = hnsw_index_manager.ntotal
            if count > 0:
                print(f"[OK] HNSW index already has {count} vectors - skipping database reload")
                index_has_data = True
        except Exception as e:
            print(f"[DEBUG] Could not check index count: {e}")
    
    if USE_DATABASE and DB_AVAILABLE and not index_has_data:
        print("[OK] Using database for visitor recognition")
        if load_visitors_from_database(hnsw_index_manager) == 0:
            print("[WARNING] No visitors loaded from database - falling back to test_images")
            USE_DATABASE = False
    
    if not USE_DATABASE or not DB_AVAILABLE:
        if not index_has_data:
            load_visitors_from_test_images(hnsw_index_manager)


# =============================================================================
# PYDANTIC SCHEMAS
# =============================================================================

class DetectRequest(BaseModel):
    image: str
    score_threshold: float = DEFAULT_SCORE_THRESHOLD
    return_landmarks: bool = False

class CompareRequest(BaseModel):
    image1: str
    image2: str
    threshold: float = DEFAULT_COMPARE_THRESHOLD

class DetectionResponse(BaseModel):
    faces: List[List[float]]
    count: int

class FeatureExtractionResponse(BaseModel):
    features: List[List[float]]
    num_faces: int

class RecognitionResponse(BaseModel):
    similarity_score: float
    is_match: bool
    features1: Optional[List[float]] = None
    features2: Optional[List[float]] = None

class VisitorRecognitionResponse(BaseModel):
    visitor_id: Optional[str] = None
    confidence: Optional[float] = None
    matched: bool = False
    firstName: Optional[str] = None
    lastName: Optional[str] = None
    visitor: Optional[str] = None  # Legacy
    match_score: Optional[float] = None  # Legacy
    matches: Optional[list] = None

class ModelStatusResponse(BaseModel):
    loaded: bool
    details: Optional[Any] = None

class ModelInfoResponse(BaseModel):
    detector: Any
    recognizer: Any

class HNSWStatusResponse(BaseModel):
    available: bool
    initialized: bool
    total_vectors: int
    dimension: int
    index_type: str
    m: Optional[int] = None
    ef_construction: Optional[int] = None
    ef_search: Optional[int] = None
    visitors_indexed: int
    details: Optional[Any] = None

class ValidateImageResponse(BaseModel):
    valid: bool
    format: Optional[str]
    size: Optional[Tuple[int, int]]


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(ValueError)
async def valueerror_handler(request, exc):
    return JSONResponse(status_code=400, content={"error": str(exc), "type": "ValueError"})

@app.exception_handler(FileNotFoundError)
async def filenotfounderror_handler(request, exc):
    return JSONResponse(status_code=500, content={"error": str(exc), "type": "FileNotFoundError"})


# =============================================================================
# API ROUTES
# =============================================================================

@app.get("/health", tags=["Health"])
@app.get("/api/v1/health", tags=["Health"])
def health():
    return {"status": "ok", "time": datetime.datetime.utcnow().isoformat() + "Z"}


@app.post("/api/v1/detect", response_model=DetectionResponse, tags=["Detection"])
async def detect_faces_api(request: DetectRequest):
    """Detect faces in an image."""
    try:
        img_np = image_loader.load_image(request.image, source_type="base64")
        image_loader.validate_image_size((img_np.shape[1], img_np.shape[0]), MAX_IMAGE_SIZE)
        
        results = inference.detect_faces(
            img_np,
            score_threshold=request.score_threshold,
            return_landmarks=request.return_landmarks
        )
        
        if results is None:
            return DetectionResponse(faces=[], count=0)
        
        faces_list = []
        for r in results:
            if request.return_landmarks:
                faces_list.append(r.tolist() if hasattr(r, 'tolist') else list(r))
            else:
                faces_list.append([float(r[0]), float(r[1]), float(r[2]), float(r[3])])
        
        return DetectionResponse(faces=faces_list, count=len(faces_list))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")


@app.post("/api/v1/extract-features", response_model=FeatureExtractionResponse, tags=["Features"])
async def extract_features_api(
    image: UploadFile = File(None),
    image_base64: str = Form(None),
):
    """Extract face features from an image."""
    if image is None and not image_base64:
        raise HTTPException(status_code=400, detail="Image required")
    
    img_np = image_loader.load_from_upload(image) if image else image_loader.load_image(image_base64, source_type="base64")
    image_loader.validate_image_size((img_np.shape[1], img_np.shape[0]), MAX_IMAGE_SIZE)
    
    faces = inference.detect_faces(img_np, return_landmarks=True)
    if faces is None or len(faces) == 0:
        return FeatureExtractionResponse(features=[], num_faces=0)
    
    features_list = []
    for face_row in faces:
        feature = inference.extract_face_features(img_np, face_row)
        if feature is not None:
            features_list.append(feature.tolist() if hasattr(feature, 'tolist') else list(feature))
    
    return FeatureExtractionResponse(features=features_list, num_faces=len(features_list))


@app.post("/api/v1/compare", response_model=RecognitionResponse, tags=["Recognition"])
async def compare_faces_api(request: CompareRequest):
    """Compare faces between two images."""
    try:
        img1 = image_loader.load_image(request.image1, source_type="base64")
        img2 = image_loader.load_image(request.image2, source_type="base64")
        
        image_loader.validate_image_size((img1.shape[1], img1.shape[0]), MAX_IMAGE_SIZE)
        image_loader.validate_image_size((img2.shape[1], img2.shape[0]), MAX_IMAGE_SIZE)
        
        faces1 = inference.detect_faces(img1, return_landmarks=True)
        faces2 = inference.detect_faces(img2, return_landmarks=True)
        
        if faces1 is None or len(faces1) == 0:
            raise HTTPException(status_code=400, detail="No face in image1")
        if faces2 is None or len(faces2) == 0:
            raise HTTPException(status_code=400, detail="No face in image2")
        
        feature1 = inference.extract_face_features(img1, faces1[0])
        feature2 = inference.extract_face_features(img2, faces2[0])
        
        if feature1 is None or feature2 is None:
            raise HTTPException(status_code=400, detail="Failed to extract features")
        
        score, is_match = inference.compare_face_features(feature1, feature2, threshold=request.threshold)
        
        return RecognitionResponse(
            similarity_score=float(score),
            is_match=bool(is_match),
            features1=None,
            features2=None,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")


@app.post("/api/v1/recognize", response_model=VisitorRecognitionResponse, tags=["Recognition"])
async def recognize_visitor_api(
    image: UploadFile = File(None),
    image_base64: str = Form(None),
    threshold: float = Form(DEFAULT_COMPARE_THRESHOLD),
):
    """Recognize visitor by matching against database."""
    if image is None and not image_base64:
        raise HTTPException(status_code=400, detail="Image required")
    
    try:
        if image is not None:
            img_np = image_loader.load_from_upload(image)
        else:
            img_np = image_loader.load_image(image_base64, source_type="base64")
        image_loader.validate_image_size((img_np.shape[1], img_np.shape[0]), MAX_IMAGE_SIZE)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {e}")
    
    faces = inference.detect_faces(img_np, return_landmarks=True)
    if faces is None or len(faces) == 0:
        return VisitorRecognitionResponse(matched=False, matches=[])
    
    query_feature = inference.extract_face_features(img_np, faces[0])
    if query_feature is None:
        return VisitorRecognitionResponse(matched=False, matches=[])
    
    results = []
    best_match = None
    best_score = 0.0
    
    # Try HNSW search first
    if hnsw_index_manager and hnsw_index_manager.ntotal > 0:
        try:
            ann_results = hnsw_index_manager.search(query_feature, k=50)
            for visitor_id, similarity, metadata in ann_results:
                is_match = similarity >= threshold
                result = {
                    "visitor_id": visitor_id,
                    "match_score": float(similarity),
                    "is_match": is_match,
                    "firstName": metadata.get('firstName'),
                    "lastName": metadata.get('lastName'),
                }
                results.append(result)
                if is_match and similarity > best_score:
                    best_score = similarity
                    best_match = result
        except Exception as e:
            print(f"[WARNING] HNSW search error: {e}")
    
    # Fallback: Linear search
    if not results:
        if USE_DATABASE and DB_AVAILABLE:
            try:
                visitors = database.get_visitor_images_from_db(
                    table_name=DB_TABLE_NAME,
                    visitor_id_column=DB_VISITOR_ID_COLUMN,
                    image_column=DB_IMAGE_COLUMN,
                    limit=DB_VISITOR_LIMIT
                )
                
                for visitor_data in visitors:
                    visitor_id = str(visitor_data.get(DB_VISITOR_ID_COLUMN))
                    base64_image = visitor_data.get(DB_IMAGE_COLUMN)
                    if not base64_image:
                        continue
                    
                    # Try as feature vector first
                    db_feature = decode_feature_from_base64(base64_image)
                    
                    # Otherwise extract from image
                    if db_feature is None:
                        try:
                            img_cv_db = image_loader.load_from_base64(base64_image)
                            db_faces = inference.detect_faces(img_cv_db, return_landmarks=True)
                            if db_faces is None or len(db_faces) == 0:
                                continue
                            db_feature = inference.extract_face_features(img_cv_db, db_faces[0])
                            if db_feature is None:
                                continue
                            db_feature = np.asarray(db_feature).flatten().astype(np.float32)
                            if db_feature.shape[0] != 128:
                                continue
                        except Exception:
                            continue
                    
                    score, is_match = inference.compare_face_features(query_feature, db_feature, threshold=threshold)
                    result = {
                        "visitor_id": visitor_id,
                        "match_score": float(score),
                        "is_match": bool(is_match),
                        "firstName": visitor_data.get('firstName'),
                        "lastName": visitor_data.get('lastName'),
                    }
                    results.append(result)
                    if is_match and score > best_score:
                        best_score = score
                        best_match = result
            except Exception as e:
                print(f"[WARNING] Database query error: {e}")
        
        # Fallback: test_images
        if not results:
            for visitor_name, visitor in VISITOR_FEATURES.items():
                db_feature = visitor.get("feature")
                score, is_match = inference.compare_face_features(query_feature, db_feature, threshold=threshold)
                result = {
                    "visitor_id": visitor_name,
                    "visitor": visitor_name,
                    "match_score": float(score),
                    "is_match": bool(is_match),
                }
                results.append(result)
                if is_match and score > best_score:
                    best_score = score
                    best_match = result
    
    results.sort(key=lambda x: x.get("match_score", 0), reverse=True)
    
    if best_match and best_match.get("is_match"):
        return VisitorRecognitionResponse(
            visitor_id=best_match.get("visitor_id"),
            confidence=best_match.get("match_score"),
            matched=True,
            firstName=best_match.get("firstName"),
            lastName=best_match.get("lastName"),
            visitor=best_match.get("visitor"),
            match_score=best_match.get("match_score"),
            matches=results[:10]
        )
    
    return VisitorRecognitionResponse(matched=False, matches=results[:10] if results else [])


@app.get("/models/status", response_model=ModelStatusResponse, tags=["Models"])
async def model_status():
    loaded = face_detector is not None and face_recognizer is not None
    return ModelStatusResponse(
        loaded=loaded,
        details={"face_detector": str(type(face_detector)), "face_recognizer": str(type(face_recognizer))} if loaded else None
    )


@app.get("/models/info", response_model=ModelInfoResponse, tags=["Models"])
async def model_info():
    return ModelInfoResponse(
        detector={
            "type": "YuNet",
            "model_path": inference.YUNET_PATH,
            "input_size": inference.YUNET_INPUT_SIZE,
            "loaded": face_detector is not None
        },
        recognizer={
            "type": "SFace",
            "model_path": inference.SFACE_PATH,
            "similarity_threshold": inference.SFACE_SIMILARITY_THRESHOLD,
            "loaded": face_recognizer is not None
        }
    )


@app.get("/api/v1/hnsw/status", response_model=HNSWStatusResponse, tags=["HNSW"])
async def hnsw_status():
    """Get HNSW index status."""
    if not HNSW_AVAILABLE:
        return HNSWStatusResponse(
            available=False, initialized=False, total_vectors=0,
            dimension=128, index_type="HNSW", visitors_indexed=0,
            details={"error": "HNSW not available"}
        )
    
    if hnsw_index_manager is None:
        return HNSWStatusResponse(
            available=True, initialized=False, total_vectors=0,
            dimension=128, index_type="HNSW", visitors_indexed=0,
            details={"error": "HNSW not initialized"}
        )
    
    try:
        stats = hnsw_index_manager.get_stats()
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
            available=True, initialized=False, total_vectors=0,
            dimension=128, index_type="HNSW", visitors_indexed=0,
            details={"error": str(e)}
        )


@app.post("/validate-image", response_model=ValidateImageResponse, tags=["Utility"])
async def validate_image_api(image: UploadFile = File(None), image_base64: str = Form(None)):
    if image is None and not image_base64:
        raise HTTPException(status_code=400, detail="Image required")
    
    try:
        if image is not None:
            contents = image.file.read()
            image.file.seek(0)
            img = Image.open(io.BytesIO(contents))
        else:
            img_bytes = base64.b64decode(image_base64.split(',', 1)[1] if image_base64.startswith('data:') else image_base64)
            img = Image.open(io.BytesIO(img_bytes))
        
        fmt = (img.format or "").lower()
        size = img.size
        valid = fmt in ALLOWED_FORMATS and size[0] <= MAX_IMAGE_SIZE[0] and size[1] <= MAX_IMAGE_SIZE[1]
        return ValidateImageResponse(valid=valid, format=fmt, size=size)
    except Exception:
        return ValidateImageResponse(valid=False, format=None, size=None)


# =============================================================================
# WEBSOCKET
# =============================================================================

@app.websocket("/ws/realtime")
async def websocket_face_endpoint(websocket: WebSocket):
    """WebSocket for real-time face detection."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                req = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "error": "Invalid JSON"})
                continue
            
            if req.get("type") != "frame":
                await websocket.send_json({"type": "error", "error": "Invalid request type"})
                continue
            
            image_b64 = req.get("image")
            if not image_b64:
                await websocket.send_json({"type": "error", "error": "Missing image"})
                continue
            
            try:
                img_np = image_loader.load_image(image_b64, source_type="base64")
                image_loader.validate_image_size((img_np.shape[1], img_np.shape[0]), MAX_IMAGE_SIZE)
                
                score_threshold = float(req.get("score_threshold", DEFAULT_SCORE_THRESHOLD))
                return_landmarks = bool(req.get("return_landmarks", False))
                
                dets = inference.detect_faces(img_np, score_threshold=score_threshold, return_landmarks=return_landmarks)
                
                faces_list = []
                if dets:
                    for r in dets:
                        face_obj = {"bbox": [float(r[0]), float(r[1]), float(r[2]), float(r[3])]}
                        if len(r) > 4:
                            face_obj["confidence"] = float(r[4])
                        if return_landmarks and len(r) > 14:
                            face_obj["landmarks"] = [float(x) for x in r[5:15]]
                        faces_list.append(face_obj)
                
                await websocket.send_json({"type": "results", "faces": faces_list, "count": len(faces_list)})
            except Exception as ex:
                await websocket.send_json({"type": "error", "error": str(ex)})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("face_recog_api:app", host="0.0.0.0", port=8000, reload=True)
