"""
Face Recognition API
REST API for face detection and recognition using YuNet and Sface models.

Implements endpoints for:
- POST /api/v1/detect           : Detect faces in an image.
- POST /api/v1/extract-features : Extract face features/vectors for detected faces.
- POST /api/v1/compare          : Compare faces between two images.
- POST /api/v1/recognize : Recognize visitor (PostgreSQL database is used if properly set up, otherwise falls back to test_images. If the database connection fails, the database module is missing, or data loading from DB fails, the service loads visitors from the test_images directory instead and uses it for recognition. This ensures face recognition can still work for testing/demo.)
- GET  /api/v1/health           : Health check.
- GET  /api/v1/models/status    : Model loading status.
- GET  /api/v1/models/info      : Model metadata.
- POST /api/v1/validate-image   : Validate image before processing.
- WEBSOCKET /ws/realtime     : Real-time face detection/recognition via websocket.
"""

import os
import io
import json
import base64
import datetime
from typing import List, Tuple, Optional, Any
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from PIL import Image

# Load environment variables from .env.test file if it exists
try:
    from dotenv import load_dotenv
    _SCRIPT_DIR = Path(__file__).parent
    env_file = _SCRIPT_DIR.parent / ".env.test"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv(_SCRIPT_DIR / ".env.test")
except ImportError:
    # python-dotenv not installed, skip .env loading
    pass

import inference  # This should be your inference.py containing detection/recognition

# Image loading utilities (centralized)
import image_loader

# Database integration (optional - falls back to test_images if not configured)
try:
    import database
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    print("Warning: database module not available. Using test images fallback.")

# HNSW index for fast ANN search
try:
    from hnsw_index import HNSWIndexManager
    HNSW_AVAILABLE = True
except ImportError:
    HNSW_AVAILABLE = False
    HNSWIndexManager = None
    print("Warning: HNSW index manager not available. Using linear search.")

# --- CONFIGURATION ---
DEFAULT_SCORE_THRESHOLD = 0.6
DEFAULT_COMPARE_THRESHOLD = 0.363
MAX_IMAGE_SIZE = 1920, 1920
ALLOWED_FORMATS = {"jpg", "jpeg", "png"}
# In Docker: working directory is /app/app, models are in /app/app/models
# Use environment variable if set, otherwise default to "models" relative to working directory
MODELS_PATH = os.environ.get("MODELS_PATH", "models")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",") if os.environ.get("CORS_ORIGINS") else ["*"]

# Database configuration
USE_DATABASE = os.environ.get("USE_DATABASE", "false").lower() == "true"
DB_TABLE_NAME = os.environ.get("DB_TABLE_NAME", "visitors")
DB_VISITOR_ID_COLUMN = os.environ.get("DB_VISITOR_ID_COLUMN", "id")  # Changed from visitor_id to id
DB_IMAGE_COLUMN = os.environ.get("DB_IMAGE_COLUMN", "base64Image")
DB_ACTIVE_ONLY = os.environ.get("DB_ACTIVE_ONLY", "false").lower() == "true"  # Not used in minimal schema
DB_VISITOR_LIMIT = int(os.environ.get("DB_VISITOR_LIMIT", "0")) or None  # 0 or None = no limit

# --- GLOBAL TEST VISITORS SETUP ---
# This section will load all reference visitors/faces from test_images on startup

VISITOR_IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_images")
VISITOR_FEATURES = {}  # Dict[str, Dict]: {visitor_name: {'feature': np.array, 'path': str}}

# --- FASTAPI APP INITIALIZATION ---
app = FastAPI(
    title="Face Recognition API",
    version="1.0.0",
    description="REST API for face detection and recognition using YuNet and Sface models.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL INFERENCE/MODEL LOADING ---
face_detector = None
face_recognizer = None
hnsw_index_manager = None  # HNSW index for fast ANN search

@app.on_event("startup")
def load_models():
    global face_detector, face_recognizer, VISITOR_FEATURES, hnsw_index_manager
    try:
        face_detector = inference.get_face_detector(MODELS_PATH)
        face_recognizer = inference.get_face_recognizer(MODELS_PATH)
    except Exception as e:
        print("Error loading models:", e)
        face_detector = None
        face_recognizer = None
    
    # Initialize HNSW index manager
    if HNSW_AVAILABLE and HNSWIndexManager is not None:
        try:
            hnsw_index_manager = HNSWIndexManager(index_dir=MODELS_PATH)
            print("✓ HNSW index manager initialized")
        except Exception as e:
            print(f"⚠ Error initializing HNSW index: {e}")
            hnsw_index_manager = None
    else:
        hnsw_index_manager = None

    # Initialize database connection - auto-detect if available
    global USE_DATABASE
    if DB_AVAILABLE:
        # Try to connect to database even if USE_DATABASE is not explicitly set
        # This allows automatic detection of Docker database when running locally
        try:
            print("Attempting to connect to database...")
            if database.test_connection():
                print("✓ Database connection successful - enabling database mode")
                USE_DATABASE = True
                # Initialize connection pool for better performance
                database.init_connection_pool(min_conn=1, max_conn=5)
            else:
                if USE_DATABASE:
                    # USE_DATABASE was explicitly set to true but connection failed
                    print("⚠ Database connection failed, falling back to test_images")
                    print("Falling back to test_images because the database connection could not be established (check DB host, credentials, or server status).")
                else:
                    # Database not explicitly enabled, silently fall back
                    print("ℹ Database not available or not configured - using test_images")
                USE_DATABASE = False
        except Exception as e:
            if USE_DATABASE:
                print(f"⚠ Database initialization error: {e}, falling back to test_images")
                print("Falling back to test_images because the database module was found but could not initialize a working connection. Likely a connection issue, bad pool config, or missing DB server dependency.")
            else:
                print(f"ℹ Database not available ({type(e).__name__}) - using test_images")
            USE_DATABASE = False
    elif USE_DATABASE:
        # USE_DATABASE was set but database module is not available
        print("⚠ Database module not available (psycopg2 not installed) - falling back to test_images")
        USE_DATABASE = False

    # Load visitor images from database or test_images (fallback)
    VISITOR_FEATURES.clear()
    
    def extract_feature_from_visitor_data(visitor_data):
        """Helper function to extract feature from visitor data."""
        try:
            base64_image = visitor_data.get(DB_IMAGE_COLUMN)
            if not base64_image:
                return None
            
            img_cv_db = image_loader.load_from_base64(base64_image)
            db_faces = inference.detect_faces(img_cv_db, return_landmarks=True)
            if db_faces is None or len(db_faces) == 0:
                return None
            
            db_feature = inference.extract_face_features(img_cv_db, db_faces[0])
            return db_feature
        except Exception as e:
            print(f"Error extracting feature: {e}")
            return None
    
    if USE_DATABASE and DB_AVAILABLE:
        # Database mode: Build HNSW index if available, otherwise use on-the-fly extraction
        print("✓ Using database for visitor recognition")
        try:
            visitors = database.get_visitor_images_from_db(
                table_name=DB_TABLE_NAME,
                visitor_id_column=DB_VISITOR_ID_COLUMN,
                image_column=DB_IMAGE_COLUMN,
                limit=DB_VISITOR_LIMIT,
                active_only=DB_ACTIVE_ONLY
            )
            print(f"✓ Database has {len(visitors)} visitors available for recognition")
            
            # Build HNSW index if available
            if hnsw_index_manager and len(visitors) > 0:
                print(f"Building HNSW index from {len(visitors)} database visitors (this may take a while)...")
                def get_visitors():
                    return visitors
                
                count = hnsw_index_manager.rebuild_from_database(
                    get_visitors_func=get_visitors,
                    extract_feature_func=extract_feature_from_visitor_data
                )
                if count > 0:
                    print(f"✓ HNSW index built with {count} visitors (fast ANN search enabled)")
                else:
                    print("⚠ HNSW index build failed, falling back to linear search")
        except Exception as e:
            print(f"⚠ Error loading visitors from database: {e}, falling back to test_images")
            print("Falling back to test_images because an error occurred loading visitor images from the database (could be a bad query, missing table/column, or unexpected DB exception).")
            USE_DATABASE = False
    
    # Fallback to test_images if database not available or failed
    if not USE_DATABASE or not DB_AVAILABLE:
        # Print fallback reason
        if not DB_AVAILABLE:
            print("Falling back to test_images because the database module is not available (maybe not installed or import failed).")
        else:
            print("Falling back to test_images because database mode is not enabled or previous database step failed.")
        if os.path.isdir(VISITOR_IMAGES_DIR):
            print(f"Loading gallery visitors from {VISITOR_IMAGES_DIR}")
            batch_data = []
            
            for fname in os.listdir(VISITOR_IMAGES_DIR):
                if not fname.lower().endswith(tuple(ALLOWED_FORMATS)):
                    continue
                fpath = os.path.join(VISITOR_IMAGES_DIR, fname)
                try:
                    # Use image_loader to load image from file path
                    img_cv = image_loader.load_from_path(fpath)

                    # Detect face(s)
                    faces = inference.detect_faces(img_cv, return_landmarks=True)
                    if faces is None or len(faces) == 0:
                        print(f"No face found in {fname}")
                        continue
                    # Take only first face for that file
                    feature = inference.extract_face_features(img_cv, faces[0])
                    if feature is not None:
                        visitor_name = os.path.splitext(fname)[0]
                        VISITOR_FEATURES[visitor_name] = {
                            "feature": feature,
                            "path": fpath
                        }
                        
                        # Add to HNSW index if available
                        if hnsw_index_manager:
                            batch_data.append((visitor_name, feature, {"path": fpath}))
                except Exception as e:
                    print(f"Failed to process {fname}: {e}")
            
            print(f"Loaded {len(VISITOR_FEATURES)} visitors from test_images")
            
            # Build HNSW index from test_images if available
            if hnsw_index_manager and batch_data:
                print(f"Building HNSW index from {len(batch_data)} test_images visitors...")
                count = hnsw_index_manager.add_visitors_batch(batch_data)
                if count > 0:
                    hnsw_index_manager.save()
                    print(f"✓ HNSW index built with {count} test_images visitors (fast ANN search enabled)")
                else:
                    print("⚠ HNSW index build failed for test_images, using linear search.")

# --- PYDANTIC SCHEMAS ---
class DetectionRequest(BaseModel):
    image_base64: Optional[str] = None
    score_threshold: Optional[float] = Field(DEFAULT_SCORE_THRESHOLD, ge=0, le=1)
    return_landmarks: Optional[bool] = False

class BoundingBox(BaseModel):
    x: int
    y: int
    w: int
    h: int

class DetectionResponse(BaseModel):
    faces: List[List[float]]  # List of [x, y, w, h] bounding boxes
    count: int  # Number of faces detected

class FeatureExtractionRequest(BaseModel):
    image_base64: Optional[str] = None
    return_face_count: Optional[bool] = False

class FeatureExtractionResponse(BaseModel):
    features: List[List[float]]
    num_faces: int

class RecognitionRequest(BaseModel):
    image1_base64: Optional[str] = None
    image2_base64: Optional[str] = None
    threshold: Optional[float] = Field(DEFAULT_COMPARE_THRESHOLD, ge=0, le=1)

class RecognitionResponse(BaseModel):
    similarity_score: float
    is_match: bool
    features1: Optional[List[float]] = None
    features2: Optional[List[float]] = None

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

class ValidateImageRequest(BaseModel):
    image_base64: Optional[str] = None

class ValidateImageResponse(BaseModel):
    valid: bool
    format: Optional[str]
    size: Optional[Tuple[int, int]]

class VisitorRecognitionResponse(BaseModel):
    visitor_id: Optional[str] = None  # Database visitor ID
    confidence: Optional[float] = None  # Match confidence score
    matched: bool = False  # Whether a match was found above threshold
    # Legacy fields for backward compatibility
    visitor: Optional[str] = None  # Deprecated: use visitor_id
    match_score: Optional[float] = None  # Deprecated: use confidence
    matches: Optional[list] = None  # Additional match details (optional)

# --- IMAGE PROCESSING UTILITIES ---
# Use centralized image_loader module instead of duplicate functions
# All image loading now goes through image_loader for consistency

# --- ERROR HANDLING ---
@app.exception_handler(ValueError)
async def valueerror_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": str(exc), "type": "ValueError"},
    )

@app.exception_handler(FileNotFoundError)
async def filenotfounderror_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": "FileNotFoundError"},
    )

# --- API ROUTES ---

@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "time": datetime.datetime.utcnow().isoformat() + "Z"}

@app.get("/api/v1/health", tags=["Health"])
def health_v1():
    return {"status": "ok", "time": datetime.datetime.utcnow().isoformat() + "Z"}

class DetectRequest(BaseModel):
    image: str
    score_threshold: float = DEFAULT_SCORE_THRESHOLD
    return_landmarks: bool = False

class CompareRequest(BaseModel):
    image1: str
    image2: str
    threshold: float = DEFAULT_COMPARE_THRESHOLD

@app.post("/api/v1/detect", response_model=DetectionResponse, tags=["Detection"])
async def detect_faces_api_v1(request: DetectRequest):
    """Detect faces endpoint that accepts JSON with base64 image"""
    try:
        # Use image_loader to handle base64 (including data URL format)
        img_np = image_loader.load_image(request.image, source_type="base64")
        image_loader.validate_image_size((img_np.shape[1], img_np.shape[0]), MAX_IMAGE_SIZE)
        results = inference.detect_faces(
            img_np, score_threshold=request.score_threshold, return_landmarks=request.return_landmarks
        )
        
        if results is None:
            return DetectionResponse(faces=[], count=0)
        
        faces_list = []
        for r in results:
            if request.return_landmarks:
                # If landmarks requested, return full array
                faces_list.append(r.tolist() if hasattr(r, 'tolist') else list(r))
            else:
                # Just return bounding box [x, y, w, h]
                faces_list.append([float(r[0]), float(r[1]), float(r[2]), float(r[3])])
        return DetectionResponse(faces=faces_list, count=len(faces_list))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/api/v1/extract-features", response_model=FeatureExtractionResponse, tags=["Features"])
async def extract_features_api(
    image: UploadFile = File(None),
    image_base64: str = Form(None),
):
    if image is None and not image_base64:
        raise HTTPException(status_code=400, detail="An image (file or base64) must be provided.")
    # Load image using unified image_loader
    if image is not None:
        img_np = image_loader.load_from_upload(image)
    else:
        img_np = image_loader.load_image(image_base64, source_type="base64")
    # Validate image size
    image_loader.validate_image_size((img_np.shape[1], img_np.shape[0]), MAX_IMAGE_SIZE)
    
    # Detect faces first
    faces = inference.detect_faces(img_np, return_landmarks=True)
    if faces is None or len(faces) == 0:
        return FeatureExtractionResponse(features=[], num_faces=0)
    
    # Extract features for each detected face
    features_list = []
    for face_row in faces:
        feature = inference.extract_face_features(img_np, face_row)
        if feature is not None:
            features_list.append(feature.tolist() if hasattr(feature, 'tolist') else list(feature))
    
    return FeatureExtractionResponse(
        features=features_list,
        num_faces=len(features_list)
    )

@app.post("/api/v1/compare", response_model=RecognitionResponse, tags=["Recognition"])
async def compare_faces_api_v1(request: CompareRequest):
    """Compare faces endpoint that accepts JSON with base64 images"""
    try:
        # Use image_loader to handle base64 (including data URL format)
        img1 = image_loader.load_image(request.image1, source_type="base64")
        img2 = image_loader.load_image(request.image2, source_type="base64")
        
        image_loader.validate_image_size((img1.shape[1], img1.shape[0]), MAX_IMAGE_SIZE)
        image_loader.validate_image_size((img2.shape[1], img2.shape[0]), MAX_IMAGE_SIZE)
        
        # Detect faces in both images
        faces1 = inference.detect_faces(img1, return_landmarks=True)
        faces2 = inference.detect_faces(img2, return_landmarks=True)
        
        if faces1 is None or len(faces1) == 0:
            raise HTTPException(status_code=400, detail="No face detected in image1")
        if faces2 is None or len(faces2) == 0:
            raise HTTPException(status_code=400, detail="No face detected in image2")
        
        # Extract features from first face in each image
        feature1 = inference.extract_face_features(img1, faces1[0])
        feature2 = inference.extract_face_features(img2, faces2[0])
        
        if feature1 is None or feature2 is None:
            raise HTTPException(status_code=400, detail="Failed to extract features from one or both images")
        
        # Compare features
        score, is_match = inference.compare_face_features(feature1, feature2, threshold=request.threshold)
        
        return RecognitionResponse(
            similarity_score=float(score),
            is_match=bool(is_match),
            features1=None,  # Don't return features for security/performance
            features2=None,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/api/v1/recognize", response_model=VisitorRecognitionResponse, tags=["Recognition"])
async def recognize_visitor_api(
    image: UploadFile = File(None),
    image_base64: str = Form(None),
    threshold: float = Form(DEFAULT_COMPARE_THRESHOLD),
):
    """
    Recognize visitor by matching the input image against database visitors.
    Returns visitor_id, confidence, and matched status.
    Falls back to test_images if database is not configured.

    Note:
    Falls back to the test_images directory if:
    - The database connection cannot be established (connection refused, bad credentials, no DB server running),
    - The database module is not available (e.g. not installed in the environment or ImportError),
    - A database query or loading error occurs,
    - Database integration is not enabled/configured.
    """
    if image is None and not image_base64:
        raise HTTPException(status_code=400, detail="An image (file or base64) must be provided.")
    
    # Load received image using unified image_loader
    if image is not None:
        img_np = image_loader.load_from_upload(image)
    else:
        img_np = image_loader.load_image(image_base64, source_type="base64")
    
    # Validate image size
    image_loader.validate_image_size((img_np.shape[1], img_np.shape[0]), MAX_IMAGE_SIZE)

    # Detect faces in input
    faces = inference.detect_faces(img_np, return_landmarks=True)
    if faces is None or len(faces) == 0:
        return VisitorRecognitionResponse(
            visitor_id=None, confidence=None, matched=False,
            visitor=None, match_score=None, matches=[]
        )

    # Extract feature for first detected face
    query_feature = inference.extract_face_features(img_np, faces[0])
    if query_feature is None:
        return VisitorRecognitionResponse(
            visitor_id=None, confidence=None, matched=False,
            visitor=None, match_score=None, matches=[]
        )

    results = []
    best_match = None
    best_score = 0.0

    # Use HNSW ANN search if available, otherwise fall back to linear search
    if hnsw_index_manager and hnsw_index_manager.index and hnsw_index_manager.ntotal > 0:
        # Fast ANN search using HNSW
        try:
            # Search for top 50 candidates using HNSW
            ann_results = hnsw_index_manager.search(query_feature, k=50)
            
            for visitor_id, cosine_similarity, metadata in ann_results:
                # cosine_similarity from HNSW is already in range [-1, 1]
                # OpenCV's compare_face_features returns values in similar range
                # Use cosine_similarity directly as match_score
                score_float = float(cosine_similarity)
                is_match = score_float >= threshold
                
                results.append({
                    "visitor_id": visitor_id,
                    "match_score": score_float,
                    "is_match": bool(is_match),
                    **metadata  # Include any additional metadata
                })
                
                # Track best match
                if is_match and score_float > best_score:
                    best_score = score_float
                    best_match = {
                        "visitor_id": visitor_id,
                        "match_score": score_float,
                        "is_match": True
                    }
        except Exception as e:
            print(f"HNSW search error: {e}, falling back to linear search")
            # Fall through to linear search
    
    # Fallback: Linear search (original implementation)
    if len(results) == 0:
        # Database mode: Query and extract features on-the-fly
        if USE_DATABASE and DB_AVAILABLE:
            try:
                visitors = database.get_visitor_images_from_db(
                    table_name=DB_TABLE_NAME,
                    visitor_id_column=DB_VISITOR_ID_COLUMN,
                    image_column=DB_IMAGE_COLUMN,
                    limit=DB_VISITOR_LIMIT,
                    active_only=DB_ACTIVE_ONLY
                )
                
                for visitor_data in visitors:
                    visitor_id = str(visitor_data.get(DB_VISITOR_ID_COLUMN))
                    base64_image = visitor_data.get(DB_IMAGE_COLUMN)
                    
                    if not base64_image:
                        continue
                    
                    try:
                        # Use image_loader to decode base64 image from database
                        img_cv_db = image_loader.load_from_base64(base64_image)
                        
                        # Detect and extract features on-the-fly
                        db_faces = inference.detect_faces(img_cv_db, return_landmarks=True)
                        if db_faces is None or len(db_faces) == 0:
                            continue
                        
                        db_feature = inference.extract_face_features(img_cv_db, db_faces[0])
                        if db_feature is None:
                            continue
                        
                        # Compare features
                        score, is_match = inference.compare_face_features(query_feature, db_feature, threshold=threshold)
                        score_float = float(score)
                        
                        results.append({
                            "visitor_id": visitor_id,
                            "match_score": score_float,
                            "is_match": bool(is_match),
                        })
                        
                        # Track best match
                        if is_match and score_float > best_score:
                            best_score = score_float
                            best_match = {
                                "visitor_id": visitor_id,
                                "match_score": score_float,
                                "is_match": True
                            }
                            
                    except Exception as e:
                        print(f"Error processing visitor {visitor_id}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Database query error: {e}")
                print("Falling back to test_images because error occurred during database query or feature extraction step.")
                # Fall through to test_images fallback
    
    # Fallback: Use pre-loaded test_images features
    if not USE_DATABASE or not DB_AVAILABLE or len(results) == 0:
        print("Using test_images fallback for face recognition because either database is not enabled, database module is missing, or all recognition/database attempts failed.")
        for visitor_name, visitor in VISITOR_FEATURES.items():
            db_feature = visitor.get("feature")
            score, is_match = inference.compare_face_features(query_feature, db_feature, threshold=threshold)
            score_float = float(score)
            
            results.append({
                "visitor": visitor_name,  # Legacy: name instead of ID
                "visitor_id": visitor_name,  # Use name as ID for test_images
                "match_score": score_float,
                "is_match": bool(is_match),
                "filename": os.path.basename(visitor.get("path", ""))
            })
            
            # Track best match
            if is_match and score_float > best_score:
                best_score = score_float
                best_match = {
                    "visitor_id": visitor_name,
                    "visitor": visitor_name,  # Legacy
                    "match_score": score_float,
                    "is_match": True
                }
    
    # Sort by match score descending
    results.sort(key=lambda x: x.get("match_score", 0), reverse=True)
    
    # Prepare response
    if best_match and best_match.get("is_match"):
        return VisitorRecognitionResponse(
            visitor_id=best_match.get("visitor_id"),
            confidence=best_match.get("match_score"),
            matched=True,
            visitor=best_match.get("visitor"),  # Legacy
            match_score=best_match.get("match_score"),  # Legacy
            matches=results[:10]  # Top 10 matches
        )
    else:
        return VisitorRecognitionResponse(
            visitor_id=None,
            confidence=None,
            matched=False,
            visitor=None,
            match_score=None,
            matches=results[:10] if results else []
        )

@app.post("/batch-detect", tags=["Batch"])
async def batch_detect_api(
    images: List[UploadFile] = File(...),
    score_threshold: float = Form(DEFAULT_SCORE_THRESHOLD),
):
    responses = []
    for imgfile in images:
        try:
            img_np = image_loader.load_from_upload(imgfile)
            image_loader.validate_image_size((img_np.shape[1], img_np.shape[0]), MAX_IMAGE_SIZE)
            results = inference.detect_faces(
                img_np, score_threshold=score_threshold, return_landmarks=False
            )
            faces = []
            for det in results.get("faces", []):
                x, y, w, h = det["box"]
                faces.append(BoundingBox(x=x, y=y, w=w, h=h))
            responses.append({
                "faces": [f.dict() for f in faces],
                "num_faces": len(faces),
                "image_size": [img_np.shape[1], img_np.shape[0]]
            })
        except Exception as e:
            responses.append({"error": str(e), "filename": getattr(imgfile, "filename", None)})
    return responses


@app.get("/models/status", response_model=ModelStatusResponse, tags=["Models"])
async def model_status():
    loaded = (face_detector is not None) and (face_recognizer is not None)
    detail = {
        "face_detector": str(type(face_detector)),
        "face_recognizer": str(type(face_recognizer)),
    } if loaded else None
    return ModelStatusResponse(loaded=loaded, details=detail)

@app.get("/models/info", response_model=ModelInfoResponse, tags=["Models"])
async def model_info():
    # Return basic model information
    info = {
        "detector": {
            "type": "YuNet",
            "model_path": inference.YUNET_PATH,
            "input_size": inference.YUNET_INPUT_SIZE,
            "loaded": face_detector is not None
        },
        "recognizer": {
            "type": "Sface",
            "model_path": inference.SFACE_PATH,
            "similarity_threshold": inference.SFACE_SIMILARITY_THRESHOLD,
            "loaded": face_recognizer is not None
        }
    }
    return ModelInfoResponse(detector=info.get("detector"), recognizer=info.get("recognizer"))

@app.get("/api/v1/hnsw/status", response_model=HNSWStatusResponse, tags=["HNSW"])
async def hnsw_status():
    """
    Get HNSW index status and statistics.
    Returns information about whether HNSW is available, initialized, and its current state.
    """
    global hnsw_index_manager
    
    if not HNSW_AVAILABLE:
        return HNSWStatusResponse(
            available=False,
            initialized=False,
            total_vectors=0,
            dimension=512,
            index_type="HNSW",
            visitors_indexed=0,
            details={"error": "HNSW library not available. Install with: pip install hnswlib"}
        )
    
    if hnsw_index_manager is None:
        return HNSWStatusResponse(
            available=True,
            initialized=False,
            total_vectors=0,
            dimension=512,
            index_type="HNSW",
            visitors_indexed=0,
            details={"error": "HNSW index manager not initialized"}
        )
    
    try:
        stats = hnsw_index_manager.get_stats()
        return HNSWStatusResponse(
            available=True,
            initialized=True,
            total_vectors=stats.get('total_vectors', 0),
            dimension=stats.get('dimension', 512),
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
            dimension=512,
            index_type="HNSW",
            visitors_indexed=0,
            details={"error": str(e)}
        )

@app.post("/validate-image", response_model=ValidateImageResponse, tags=["Utility"])
async def validate_image_api(image: UploadFile = File(None), image_base64: str = Form(None)):
    if image is None and not image_base64:
        raise HTTPException(status_code=400, detail="An image (file or base64) must be provided.")
    try:
        # Load image using image_loader
        if image is not None:
            img_np = image_loader.load_from_upload(image)
            # Get format from original file if available
            img = Image.open(io.BytesIO(image.file.read()))
            image.file.seek(0)  # Reset file pointer
        else:
            img_np = image_loader.load_image(image_base64, source_type="base64")
            # Decode to get format info
            img_bytes = base64.b64decode(image_base64.split(',', 1)[1] if image_base64.startswith('data:') else image_base64)
            img = Image.open(io.BytesIO(img_bytes))
        
        fmt = (img.format or "").lower()
        size = img.size
        valid = (fmt in ALLOWED_FORMATS) and (size[0] <= MAX_IMAGE_SIZE[0] and size[1] <= MAX_IMAGE_SIZE[1])
        return ValidateImageResponse(valid=valid, format=fmt, size=size)
    except Exception as e:
        return ValidateImageResponse(valid=False, format=None, size=None)

# --- WEBSOCKET FOR REAL-TIME FACE DETECTION/RECOGNITION ---
@app.websocket("/ws/realtime")
async def websocket_face_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time face detection/recognition.
    Expected message format: { "type": "frame", "image": "<base64>", ... }
    Response: { "type": "results", faces: [...], count: <number> }
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                req = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "error": "Invalid JSON input"
                })
                continue

            # --- Only support { type: 'frame', image: ... } format (per docs) ---
            req_type = req.get("type")
            if req_type != "frame":
                await websocket.send_json({
                    "type": "error",
                    "error": "Invalid request type. Must be { type: 'frame', image: ... }"
                })
                continue

            image_b64 = req.get("image")
            score_threshold = float(req.get("score_threshold", DEFAULT_SCORE_THRESHOLD))
            return_landmarks = bool(req.get("return_landmarks", False))
            if not image_b64:
                await websocket.send_json({
                    "type": "error",
                    "error": "Missing required field: image (base64)"
                })
                continue

            try:
                img_np = image_loader.load_image(image_b64, source_type="base64")
                image_loader.validate_image_size((img_np.shape[1], img_np.shape[0]), MAX_IMAGE_SIZE)
                dets = inference.detect_faces(img_np, score_threshold=score_threshold, return_landmarks=return_landmarks)
                faces_list = []
                if dets:
                    for r in dets:
                        # r shape: [x, y, w, h, score, ...] (landmarks after 5th field)
                        bbox = [float(r[0]), float(r[1]), float(r[2]), float(r[3])]
                        score = float(r[4]) if len(r) > 4 else None
                        face_obj = {"bbox": bbox}
                        if score is not None:
                            face_obj["confidence"] = score
                        if return_landmarks and len(r) > 9:
                            # YuNet returns landmarks after 5th index: [x, y, w, h, score, l0x, l0y, ... l4x, l4y]
                            landmarks = [float(x) for x in r[5:15]]
                            face_obj["landmarks"] = landmarks
                        faces_list.append(face_obj)
                response = {
                    "type": "results",
                    "faces": faces_list,
                    "count": len(faces_list)
                }
            except Exception as ex:
                response = {
                    "type": "error",
                    "error": str(ex)
                }

            await websocket.send_json(response)
    except WebSocketDisconnect:
        # Client disconnected; cleanup if needed
        pass
    except Exception as e:
        # Catch all, try to tell client if possible
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass

# --- MAIN ENTRYPOINT (for direct run, optional) ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("face_recog_api:app", host="0.0.0.0", port=8000, reload=True)
