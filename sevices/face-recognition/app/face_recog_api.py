"""
Face Recognition API
REST API for face detection and recognition using YuNet and Sface models.

Implements endpoints for:
- POST /detect           : Detect faces in an image.
- POST /extract-features : Extract face features/vectors for detected faces.
- POST /compare          : Compare faces between two images.
- GET  /health           : Health check.
- GET  /models/status    : Model loading status.
- GET  /models/info      : Model metadata.
- POST /validate-image   : Validate image before processing.

"""
import os
import io
import json
import base64
import datetime
from typing import List, Tuple, Optional, Any

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from PIL import Image

import inference  # This should be your inference.py containing detection/recognition

# --- CONFIGURATION ---
DEFAULT_SCORE_THRESHOLD = 0.6
DEFAULT_COMPARE_THRESHOLD = 0.363
MAX_IMAGE_SIZE = 1920, 1920
ALLOWED_FORMATS = {"jpg", "jpeg", "png"}
MODELS_PATH = os.environ.get("MODELS_PATH", "models")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",") if os.environ.get("CORS_ORIGINS") else ["*"]


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
    global face_detector, face_recognizer
    try:
        face_detector = inference.get_face_detector(MODELS_PATH)
        face_recognizer = inference.get_face_recognizer(MODELS_PATH)
    except Exception as e:
        print("Error loading models:", e)
        face_detector = None
        face_recognizer = None


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
    feats, num_faces = inference.extract_face_features(img_np, detector=face_detector, recognizer=face_recognizer)
    return FeatureExtractionResponse(
        features=feats,
        num_faces=num_faces
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
                img_np, detector=face_detector, score_threshold=score_threshold, return_landmarks=False
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
    info = inference.get_model_info(MODELS_PATH)
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

# --- MAIN ENTRYPOINT (for direct run, optional) ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("face_recog_api:app", host="0.0.0.0", port=8000, reload=True)
