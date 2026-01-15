"""
Face Recognition API
REST API for face detection and recognition using YuNet and Sface models.

Implements endpoints for:
- POST /detect           : Detect faces in an image.
- POST /extract-features : Extract face features/vectors for detected faces.
- POST /compare          : Compare faces between two images.
- POST /api/v1/recognize : Recognize visitor (PostgreSQL database or test_images fallback)
- GET  /health           : Health check.
- GET  /models/status    : Model loading status.
- GET  /models/info      : Model metadata.
- POST /validate-image   : Validate image before processing.
- WEBSOCKET /ws/face     : Real-time face detection/recognition via websocket.
"""

import os
import io
import json
import base64
import datetime
from typing import List, Tuple, Optional, Any

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from PIL import Image

import inference  # This should be your inference.py containing detection/recognition

# Database integration (optional - falls back to test_images if not configured)
try:
    import database
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    print("Warning: database module not available. Using test images fallback.")

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

@app.on_event("startup")
def load_models():
    global face_detector, face_recognizer, VISITOR_FEATURES
    try:
        face_detector = inference.get_face_detector(MODELS_PATH)
        face_recognizer = inference.get_face_recognizer(MODELS_PATH)
    except Exception as e:
        print("Error loading models:", e)
        face_detector = None
        face_recognizer = None

    # Initialize database connection if enabled
    global USE_DATABASE
    if USE_DATABASE and DB_AVAILABLE:
        try:
            if database.test_connection():
                print("✓ Database connection successful")
                # Initialize connection pool for better performance
                database.init_connection_pool(min_conn=1, max_conn=5)
            else:
                print("⚠ Database connection failed, falling back to test_images")
                USE_DATABASE = False
        except Exception as e:
            print(f"⚠ Database initialization error: {e}, falling back to test_images")
            USE_DATABASE = False

    # Load visitor images from database or test_images (fallback)
    VISITOR_FEATURES.clear()
    
    if USE_DATABASE and DB_AVAILABLE:
        # Database mode: Don't pre-load features, extract on-the-fly during recognition
        print("✓ Using database for visitor recognition (on-the-fly feature extraction)")
        try:
            visitors = database.get_visitor_images_from_db(
                table_name=DB_TABLE_NAME,
                visitor_id_column=DB_VISITOR_ID_COLUMN,
                image_column=DB_IMAGE_COLUMN,
                limit=DB_VISITOR_LIMIT,
                active_only=DB_ACTIVE_ONLY
            )
            print(f"✓ Database has {len(visitors)} visitors available for recognition")
        except Exception as e:
            print(f"⚠ Error loading visitors from database: {e}, falling back to test_images")
            USE_DATABASE = False
    
    # Fallback to test_images if database not available or failed
    if not USE_DATABASE or not DB_AVAILABLE:
        if os.path.isdir(VISITOR_IMAGES_DIR):
            print(f"Loading gallery visitors from {VISITOR_IMAGES_DIR}")
            for fname in os.listdir(VISITOR_IMAGES_DIR):
                if not fname.lower().endswith(tuple(ALLOWED_FORMATS)):
                    continue
                fpath = os.path.join(VISITOR_IMAGES_DIR, fname)
                try:
                    # Read via PIL and CV2
                    img = Image.open(fpath)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img_np = np.array(img)
                    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

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
                except Exception as e:
                    print(f"Failed to process {fname}: {e}")
            print(f"Loaded {len(VISITOR_FEATURES)} visitors from test_images")

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
def decode_base64_image(img_b64: str) -> np.ndarray:
    try:
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_np = np.array(img)
        # Pillow returns HWC, convert to opencv BGR
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    except Exception:
        raise ValueError("Invalid base64 image.")

def uploadfile_to_np(upload_file: UploadFile) -> np.ndarray:
    try:
        contents = upload_file.file.read()
        img = Image.open(io.BytesIO(contents))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_np = np.array(img)
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    except Exception:
        raise ValueError("Invalid uploaded image file.")

def validate_image_format(img: Image.Image):
    fmt = (img.format or "").lower()
    if fmt not in ALLOWED_FORMATS:
        raise ValueError(f"Image format '{fmt}' not allowed. Allowed: {ALLOWED_FORMATS}")

def validate_image_size(size: Tuple[int, int]):
    max_w, max_h = MAX_IMAGE_SIZE
    w, h = size
    if w > max_w or h > max_h:
        raise ValueError(f"Image dimensions too large: {w}x{h}, max is {MAX_IMAGE_SIZE}")

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
        # Handle data URL format (data:image/jpeg;base64,...) or plain base64
        image_b64 = request.image
        if image_b64.startswith('data:'):
            # Extract base64 part after comma
            image_b64 = image_b64.split(',', 1)[1] if ',' in image_b64 else image_b64
        
        img_np = decode_base64_image(image_b64)
        validate_image_size((img_np.shape[1], img_np.shape[0]))
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

@app.post("/detect", response_model=DetectionResponse, tags=["Detection"])
async def detect_faces_api(
    image: UploadFile = File(None),
    image_base64: str = Form(None),
    score_threshold: float = Form(DEFAULT_SCORE_THRESHOLD),
    return_landmarks: bool = Form(False),
):
    if image is None and not image_base64:
        raise HTTPException(status_code=400, detail="An image (file or base64) must be provided.")
    # Load image
    if image is not None:
        img_np = uploadfile_to_np(image)
    else:
        img_np = decode_base64_image(image_base64)
    # Validate image size
    validate_image_size((img_np.shape[1], img_np.shape[0]))
    results = inference.detect_faces(
        img_np, score_threshold=score_threshold, return_landmarks=return_landmarks
    )
    
    if results is None:
        return DetectionResponse(faces=[], count=0)
    
    faces_list = []
    for r in results:
        if return_landmarks:
            faces_list.append(r.tolist() if hasattr(r, 'tolist') else list(r))
        else:
            faces_list.append([float(r[0]), float(r[1]), float(r[2]), float(r[3])])
    return DetectionResponse(
        faces=faces_list,
        count=len(faces_list)
    )

@app.post("/extract-features", response_model=FeatureExtractionResponse, tags=["Features"])
async def extract_features_api(
    image: UploadFile = File(None),
    image_base64: str = Form(None),
):
    if image is None and not image_base64:
        raise HTTPException(status_code=400, detail="An image (file or base64) must be provided.")
    if image is not None:
        img_np = uploadfile_to_np(image)
    else:
        img_np = decode_base64_image(image_base64)
    # Validate image size
    validate_image_size((img_np.shape[1], img_np.shape[0]))
    
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
        # Handle data URL format
        image1_b64 = request.image1
        if image1_b64.startswith('data:'):
            image1_b64 = image1_b64.split(',', 1)[1] if ',' in image1_b64 else image1_b64
        
        image2_b64 = request.image2
        if image2_b64.startswith('data:'):
            image2_b64 = image2_b64.split(',', 1)[1] if ',' in image2_b64 else image2_b64
        
        img1 = decode_base64_image(image1_b64)
        img2 = decode_base64_image(image2_b64)
        
        validate_image_size((img1.shape[1], img1.shape[0]))
        validate_image_size((img2.shape[1], img2.shape[0]))
        
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

@app.post("/compare", response_model=RecognitionResponse, tags=["Recognition"])
async def compare_faces_api(
    image1: UploadFile = File(None),
    image2: UploadFile = File(None),
    image1_base64: str = Form(None),
    image2_base64: str = Form(None),
    threshold: float = Form(DEFAULT_COMPARE_THRESHOLD),
):
    if (image1 is None and not image1_base64) or (image2 is None and not image2_base64):
        raise HTTPException(status_code=400, detail="Both images must be provided (file or base64).")
    if image1 is not None:
        img1 = uploadfile_to_np(image1)
    else:
        img1 = decode_base64_image(image1_base64)
    if image2 is not None:
        img2 = uploadfile_to_np(image2)
    else:
        img2 = decode_base64_image(image2_base64)

    validate_image_size((img1.shape[1], img1.shape[0]))
    validate_image_size((img2.shape[1], img2.shape[0]))

    # Detect faces in both images
    faces1 = inference.detect_faces(img1, return_landmarks=True)
    faces2 = inference.detect_faces(img2, return_landmarks=True)
    
    if faces1 is None or len(faces1) == 0:
        raise HTTPException(status_code=400, detail="No face detected in image1")
    if faces2 is None or len(faces2) == 0:
        raise HTTPException(status_code=400, detail="No face detected in image2")
    
    # Extract features
    feature1 = inference.extract_face_features(img1, faces1[0])
    feature2 = inference.extract_face_features(img2, faces2[0])
    
    if feature1 is None or feature2 is None:
        raise HTTPException(status_code=400, detail="Failed to extract features")
    
    # Compare
    score, is_match = inference.compare_face_features(feature1, feature2, threshold=threshold)
    
    return RecognitionResponse(
        similarity_score=float(score),
        is_match=bool(is_match),
        features1=None,
        features2=None,
    )


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
    """
    if image is None and not image_base64:
        raise HTTPException(status_code=400, detail="An image (file or base64) must be provided.")
    
    # Load received image
    if image is not None:
        img_np = uploadfile_to_np(image)
    else:
        img_np = decode_base64_image(image_base64)
    
    # Validate image size
    validate_image_size((img_np.shape[1], img_np.shape[0]))

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
                    # Decode base64 image
                    img_bytes = base64.b64decode(base64_image)
                    img = Image.open(io.BytesIO(img_bytes))
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img_np_db = np.array(img)
                    img_cv_db = cv2.cvtColor(img_np_db, cv2.COLOR_RGB2BGR)
                    
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
            # Fall through to test_images fallback
    
    # Fallback: Use pre-loaded test_images features
    if not USE_DATABASE or not DB_AVAILABLE or len(results) == 0:
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
            img_np = uploadfile_to_np(imgfile)
            validate_image_size((img_np.shape[1], img_np.shape[0]))
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

@app.post("/validate-image", response_model=ValidateImageResponse, tags=["Utility"])
async def validate_image_api(image: UploadFile = File(None), image_base64: str = Form(None)):
    if image is None and not image_base64:
        raise HTTPException(status_code=400, detail="An image (file or base64) must be provided.")
    try:
        if image is not None:
            contents = image.file.read()
            img = Image.open(io.BytesIO(contents))
        else:
            img_bytes = base64.b64decode(image_base64)
            img = Image.open(io.BytesIO(img_bytes))
        fmt = (img.format or "").lower()
        size = img.size
        valid = (fmt in ALLOWED_FORMATS) and (size[0] <= MAX_IMAGE_SIZE[0] and size[1] <= MAX_IMAGE_SIZE[1])
        return ValidateImageResponse(valid=valid, format=fmt, size=size)
    except Exception as e:
        return ValidateImageResponse(valid=False, format=None, size=None)

# --- WEBSOCKET FOR REAL-TIME FACE DETECTION/RECOGNITION ---
@app.websocket("/ws/face")
async def websocket_face_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time face detection/recognition.
    Accepts JSON messages specifying an "action" and required data (e.g., 'detect', 'compare', 'recognize').
    Accepts images in Base64. Responds with detection/recognition results.
    Example JSON request:
    {
        "action": "detect",
        "image_base64": "...",
        "score_threshold": 0.6,
        "return_landmarks": false
    }
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                req = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON input", "ok": False})
                continue

            response = {}
            try:
                # Handle different actions: 'detect', 'compare', 'recognize'
                action = req.get("action")
                if action == "detect":
                    # Face detection on single image
                    image_b64 = req.get("image_base64")
                    score_threshold = float(req.get("score_threshold", DEFAULT_SCORE_THRESHOLD))
                    return_landmarks = bool(req.get("return_landmarks", False))
                    if not image_b64:
                        response = {"error": "image_base64 required", "ok": False}
                    else:
                        try:
                            img_np = decode_base64_image(image_b64)
                            validate_image_size((img_np.shape[1], img_np.shape[0]))
                            dets = inference.detect_faces(img_np, score_threshold=score_threshold, return_landmarks=return_landmarks)
                            faces_list = []
                            if dets:
                                for r in dets:
                                    if return_landmarks:
                                        faces_list.append(r.tolist() if hasattr(r, 'tolist') else list(r))
                                    else:
                                        faces_list.append([float(r[0]), float(r[1]), float(r[2]), float(r[3])])
                            response = {
                                "faces": faces_list,
                                "count": len(faces_list),
                                "ok": True
                            }
                        except Exception as ex:
                            response = {"error": str(ex), "ok": False}
                elif action == "compare":
                    # Face comparison between two images
                    image1_b64 = req.get("image1_base64")
                    image2_b64 = req.get("image2_base64")
                    threshold = float(req.get("threshold", DEFAULT_COMPARE_THRESHOLD))
                    if not image1_b64 or not image2_b64:
                        response = {"error": "image1_base64 and image2_base64 required", "ok": False}
                    else:
                        try:
                            img1 = decode_base64_image(image1_b64)
                            img2 = decode_base64_image(image2_b64)
                            validate_image_size((img1.shape[1], img1.shape[0]))
                            validate_image_size((img2.shape[1], img2.shape[0]))
                            faces1 = inference.detect_faces(img1, return_landmarks=True)
                            faces2 = inference.detect_faces(img2, return_landmarks=True)
                            if not faces1 or not faces2:
                                response = {"error": "No face detected in one or both images", "ok": False}
                            else:
                                feature1 = inference.extract_face_features(img1, faces1[0])
                                feature2 = inference.extract_face_features(img2, faces2[0])
                                if feature1 is None or feature2 is None:
                                    response = {"error": "Failed to extract features", "ok": False}
                                else:
                                    score, is_match = inference.compare_face_features(feature1, feature2, threshold=threshold)
                                    response = {
                                        "similarity_score": float(score),
                                        "is_match": bool(is_match),
                                        "ok": True
                                    }
                        except Exception as ex:
                            response = {"error": str(ex), "ok": False}
                elif action == "recognize":
                    # Recognize visitor among gallery
                    image_b64 = req.get("image_base64")
                    threshold = float(req.get("threshold", DEFAULT_COMPARE_THRESHOLD))
                    if not image_b64:
                        response = {"error": "image_base64 required", "ok": False}
                    else:
                        try:
                            img_np = decode_base64_image(image_b64)
                            validate_image_size((img_np.shape[1], img_np.shape[0]))
                            faces = inference.detect_faces(img_np, return_landmarks=True)
                            if not faces:
                                response = {"visitor": None, "match_score": None, "matches": [], "ok": True}
                            else:
                                query_feature = inference.extract_face_features(img_np, faces[0])
                                if query_feature is None:
                                    response = {"visitor": None, "match_score": None, "matches": [], "ok": True}
                                else:
                                    results = []
                                    for visitor_name, visitor in VISITOR_FEATURES.items():
                                        db_feature = visitor.get("feature")
                                        score, is_match = inference.compare_face_features(query_feature, db_feature, threshold=threshold)
                                        results.append({
                                            "visitor": visitor_name,
                                            "match_score": float(score),
                                            "is_match": bool(is_match),
                                            "filename": os.path.basename(visitor.get("path", ""))
                                        })
                                    results.sort(key=lambda x: x["match_score"], reverse=True)
                                    best_match = results[0] if results else None
                                    recognized_name = best_match["visitor"] if best_match and best_match["is_match"] else None
                                    match_score = best_match["match_score"] if best_match and best_match["is_match"] else None
                                    response = {
                                        "visitor": recognized_name,
                                        "match_score": match_score,
                                        "matches": results,
                                        "ok": True
                                    }
                        except Exception as ex:
                            response = {"error": str(ex), "ok": False}
                else:
                    response = {"error": "Unknown or missing action", "ok": False}
            except Exception as ex:
                response = {"error": str(ex), "ok": False}

            await websocket.send_json(response)
    except WebSocketDisconnect:
        # Client disconnected; cleanup if needed
        pass
    except Exception as e:
        # Catch all, try to tell client if possible
        try:
            await websocket.send_json({"error": str(e), "ok": False})
        except Exception:
            pass

# --- MAIN ENTRYPOINT (for direct run, optional) ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("face_recog_api:app", host="0.0.0.0", port=8000, reload=True)
