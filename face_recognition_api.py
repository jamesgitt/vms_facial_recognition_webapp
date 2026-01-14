"""
Face Detection & Recognition API
YuNet + Sface Models Only
Integrates with existing Visitor Management System
"""
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import cv2
import numpy as np
import base64
import io
from PIL import Image
import json
import os
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
import asyncio

# Import face detector
from detection import FaceDetector

app = FastAPI(title="Face Detection & Recognition API", version="1.0.0")

# CORS middleware
# TODO: Update allow_origins for production - use environment variable
# Example: allow_origins=os.getenv("ALLOWED_ORIGINS", "").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ SECURITY: Configure for production - restrict to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize face detector (loads YuNet + Sface models)
detector = FaceDetector()

# PostgreSQL connection for visitor database (assumes existing system)
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://admin:password@postgres:5432/visitors")

connection_pool = None

def get_db_connection():
    """Get database connection from pool"""
    global connection_pool
    if connection_pool is None:
        connection_pool = SimpleConnectionPool(1, 20, DATABASE_URL)
    return connection_pool.getconn()

def return_db_connection(conn):
    """Return connection to pool"""
    connection_pool.putconn(conn)

# Request/Response Models
class DetectRequest(BaseModel):
    image: str  # base64 encoded image

class DetectResponse(BaseModel):
    faces: List[dict]
    count: int

class RecognizeRequest(BaseModel):
    image: str  # base64 encoded image
    threshold: Optional[float] = 0.363

class RecognizeResponse(BaseModel):
    visitor_id: Optional[str]
    confidence: Optional[float]
    matched: bool

class ExtractFeaturesRequest(BaseModel):
    image: str  # base64 encoded image

class ExtractFeaturesResponse(BaseModel):
    feature_vector: Optional[List[float]]  # 512-dim feature vector
    face_detected: bool

def base64_to_image(base64_string: str) -> np.ndarray:
    """Convert base64 string to OpenCV image"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        image = np.array(image)
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

@app.get("/")
def root():
    return {
        "service": "Face Detection & Recognition API",
        "version": "1.0.0",
        "models": ["YuNet", "Sface"],
        "endpoints": {
            "detect": "/api/v1/detect",
            "recognize": "/api/v1/recognize",
            "extract_features": "/api/v1/extract-features",
            "realtime": "/ws/realtime"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": True}

@app.post("/api/v1/detect", response_model=DetectResponse)
async def detect_faces(request: DetectRequest):
    """
    Detect faces in an image using YuNet.
    Returns bounding boxes and landmarks.
    """
    try:
        image = base64_to_image(request.image)
        
        faces = detector.detect_faces(
            image,
            resize_factor=1.0,
            score_threshold=0.6,
            return_landmarks=False
        )
        
        faces_data = []
        for x, y, w, h in faces:
            faces_data.append({
                "bbox": [int(x), int(y), int(w), int(h)],
                "confidence": 1.0
            })
        
        return DetectResponse(faces=faces_data, count=len(faces_data))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/extract-features", response_model=ExtractFeaturesResponse)
async def extract_features(request: ExtractFeaturesRequest):
    """
    Extract face features from an image using Sface.
    Returns 512-dimensional feature vector.
    Use this to store features in your existing visitor management system.
    """
    try:
        image = base64_to_image(request.image)
        
        # Detect faces with landmarks
        faces_data = detector.detect_faces(
            image,
            resize_factor=1.0,
            score_threshold=0.6,
            return_landmarks=True
        )
        
        if not faces_data:
            return ExtractFeaturesResponse(
                feature_vector=None,
                face_detected=False
            )
        
        # Extract features from first face
        face_data = faces_data[0]
        feature = detector.extract_face_features(image, face_data)
        
        if feature is None:
            return ExtractFeaturesResponse(
                feature_vector=None,
                face_detected=True
            )
        
        return ExtractFeaturesResponse(
            feature_vector=feature.tolist(),
            face_detected=True
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/recognize", response_model=RecognizeResponse)
async def recognize_face(request: RecognizeRequest):
    """
    Recognize a face by comparing against visitor database.
    Extracts features from stored visitor images on-the-fly.
    
    Expected database schema:
    - visitors table with: visitor_id (UUID), base64Image (TEXT)
    - OR visitor_images table with: visitor_id (UUID), base64Image (TEXT)
    """
    try:
        image = base64_to_image(request.image)
        
        # Detect faces with landmarks
        faces_data = detector.detect_faces(
            image,
            resize_factor=1.0,
            score_threshold=0.6,
            return_landmarks=True
        )
        
        if not faces_data:
            return RecognizeResponse(
                visitor_id=None,
                confidence=None,
                matched=False
            )
        
        # Extract features from first face
        face_data = faces_data[0]
        feature = detector.extract_face_features(image, face_data)
        
        if feature is None:
            return RecognizeResponse(
                visitor_id=None,
                confidence=None,
                matched=False
            )
        
        # Get visitor images from database and extract features on-the-fly
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Try to get images from visitor_images table first, fallback to visitors table
            cursor.execute("""
                SELECT visitor_id, base64Image 
                FROM visitor_images
                WHERE base64Image IS NOT NULL AND base64Image != ''
            """)
            visitor_images = cursor.fetchall()
            
            # If no visitor_images table or empty, try visitors table
            if not visitor_images:
                cursor.execute("""
                    SELECT visitor_id, base64Image 
                    FROM visitors
                    WHERE base64Image IS NOT NULL AND base64Image != ''
                """)
                visitor_images = cursor.fetchall()
            
            best_score = 0.0
            best_visitor_id = None
            
            # Extract features from each visitor image and compare
            for visitor_id, stored_image_base64 in visitor_images:
                try:
                    # Decode stored image
                    stored_image = base64_to_image(stored_image_base64)
                    
                    # Detect and extract features from stored image
                    stored_faces = detector.detect_faces(
                        stored_image,
                        resize_factor=1.0,
                        score_threshold=0.6,
                        return_landmarks=True
                    )
                    
                    if not stored_faces:
                        continue
                    
                    # Extract features from first face in stored image
                    stored_face_data = stored_faces[0]
                    stored_feature = detector.extract_face_features(stored_image, stored_face_data)
                    
                    if stored_feature is None:
                        continue
                    
                    # Compare features
                    score, is_match = detector.compare_faces(
                        feature,
                        stored_feature,
                        threshold=request.threshold
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_visitor_id = visitor_id
                        if is_match and score >= request.threshold:
                            # Found good match, can stop searching
                            break
                
                except Exception as e:
                    # Skip this visitor image if there's an error
                    print(f"Error processing visitor {visitor_id}: {e}")
                    continue
            
            matched = best_score >= request.threshold
            
        finally:
            cursor.close()
            return_db_connection(conn)
        
        return RecognizeResponse(
            visitor_id=str(best_visitor_id) if matched else None,
            confidence=float(best_score) if matched else None,
            matched=matched
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/realtime")
async def websocket_realtime(websocket: WebSocket):
    """
    WebSocket endpoint for real-time face detection and recognition.
    Sends frames from client, receives detection/recognition results.
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive frame data
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "frame":
                image_base64 = message.get("image")
                if not image_base64:
                    continue
                
                try:
                    image = base64_to_image(image_base64)
                    
                    # Detect faces
                    faces = detector.detect_faces(
                        image,
                        resize_factor=0.5,  # Faster for real-time
                        score_threshold=0.6,
                        return_landmarks=False
                    )
                    
                    # Process recognition if faces found
                    results = []
                    if faces:
                        # Get faces with landmarks for recognition
                        faces_data = detector.detect_faces(
                            image,
                            resize_factor=0.5,
                            score_threshold=0.6,
                            return_landmarks=True
                        )
                        
                        for i, face_data in enumerate(faces_data[:5]):  # Limit to 5 faces
                            x, y, w, h = int(face_data[0]), int(face_data[1]), int(face_data[2]), int(face_data[3])
                            
                            # Extract features
                            feature = detector.extract_face_features(image, face_data)
                            
                            recognition_result = {
                                "bbox": [x, y, w, h],
                                "visitor_id": None,
                                "confidence": None,
                                "matched": False
                            }
                            
                            if feature is not None:
                                # Get visitor images from database and extract features on-the-fly
                                conn = get_db_connection()
                                cursor = conn.cursor()
                                
                                try:
                                    # Try visitor_images table first, fallback to visitors table
                                    cursor.execute("""
                                        SELECT visitor_id, base64Image 
                                        FROM visitor_images
                                        WHERE base64Image IS NOT NULL AND base64Image != ''
                                    """)
                                    visitor_images = cursor.fetchall()
                                    
                                    if not visitor_images:
                                        cursor.execute("""
                                            SELECT visitor_id, base64Image 
                                            FROM visitors
                                            WHERE base64Image IS NOT NULL AND base64Image != ''
                                        """)
                                        visitor_images = cursor.fetchall()
                                    
                                    best_score = 0.0
                                    best_visitor_id = None
                                    
                                    # Extract features from each visitor image and compare
                                    for visitor_id, stored_image_base64 in visitor_images:
                                        try:
                                            stored_image = base64_to_image(stored_image_base64)
                                            stored_faces = detector.detect_faces(
                                                stored_image,
                                                resize_factor=1.0,
                                                score_threshold=0.6,
                                                return_landmarks=True
                                            )
                                            
                                            if not stored_faces:
                                                continue
                                            
                                            stored_face_data = stored_faces[0]
                                            stored_feature = detector.extract_face_features(stored_image, stored_face_data)
                                            
                                            if stored_feature is None:
                                                continue
                                            
                                            score, is_match = detector.compare_faces(
                                                feature,
                                                stored_feature,
                                                threshold=0.363
                                            )
                                            
                                            if score > best_score:
                                                best_score = score
                                                best_visitor_id = visitor_id
                                                if is_match and score >= 0.363:
                                                    break
                                        
                                        except Exception as e:
                                            continue
                                    
                                    if best_score >= 0.363:
                                        recognition_result["visitor_id"] = str(best_visitor_id)
                                        recognition_result["confidence"] = float(best_score)
                                        recognition_result["matched"] = True
                                
                                finally:
                                    cursor.close()
                                    return_db_connection(conn)
                            
                            results.append(recognition_result)
                    
                    # Send results back
                    response = {
                        "type": "results",
                        "faces": results,
                        "count": len(results),
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    await websocket.send_text(json.dumps(response))
                
                except Exception as e:
                    error_response = {
                        "type": "error",
                        "message": str(e)
                    }
                    await websocket.send_text(json.dumps(error_response))
            
            elif message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("Face Detection & Recognition API started")
    print("Models loaded: YuNet + Sface")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global connection_pool
    if connection_pool:
        connection_pool.closeall()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
