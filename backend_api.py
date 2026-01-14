"""
Streamlined Backend API for Face Detection & Recognition
YuNet + Sface Model Services
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
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
import uuid
import os

# Import our face detector
from detection import FaceDetector

app = FastAPI(title="Face Detection & Recognition API", version="1.0.0")

# CORS middleware for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize face detector (loads YuNet + Sface models)
detector = FaceDetector()

# PostgreSQL database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://admin:password@postgres:5432/visitors")

# Connection pool for PostgreSQL
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

def init_database():
    """Initialize database tables"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS visitors (
                visitor_id UUID PRIMARY KEY,
                name VARCHAR(255),
                email VARCHAR(255),
                phone VARCHAR(50),
                registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_features (
                feature_id UUID PRIMARY KEY,
                visitor_id UUID REFERENCES visitors(visitor_id) ON DELETE CASCADE,
                feature_vector BYTEA,
                image_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS visits (
                visit_id UUID PRIMARY KEY,
                visitor_id UUID REFERENCES visitors(visitor_id) ON DELETE CASCADE,
                recognized_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confidence REAL,
                image_path TEXT
            )
        """)
        
        # Create indexes for performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_face_features_visitor 
            ON face_features(visitor_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_visits_visitor 
            ON visits(visitor_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_visits_date 
            ON visits(recognized_at)
        """)
        
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Database initialization error: {e}")
    finally:
        cursor.close()
        return_db_connection(conn)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database on application startup"""
    # Wait a bit for PostgreSQL to be ready
    import time
    max_retries = 10
    for i in range(max_retries):
        try:
            init_database()
            print("Database initialized successfully!")
            break
        except Exception as e:
            if i < max_retries - 1:
                print(f"Waiting for database... ({i+1}/{max_retries})")
                time.sleep(2)
            else:
                print(f"Failed to connect to database: {e}")
                raise

@app.on_event("shutdown")
async def shutdown_event():
    """Close connection pool on shutdown"""
    global connection_pool
    if connection_pool:
        connection_pool.closeall()

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
    visitor_info: Optional[dict]
    matched: bool

class RegisterRequest(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    images: List[str]  # base64 encoded images

class RegisterResponse(BaseModel):
    visitor_id: str
    message: str

def base64_to_image(base64_string: str) -> np.ndarray:
    """Convert base64 string to OpenCV image"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        image = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

@app.get("/")
def root():
    return {"message": "Face Detection & Recognition API", "version": "1.0.0"}

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
        # Convert base64 to image
        image = base64_to_image(request.image)
        
        # Detect faces
        faces = detector.detect_faces(
            image,
            resize_factor=1.0,
            score_threshold=0.6,
            return_landmarks=False
        )
        
        # Format response
        faces_data = []
        for x, y, w, h in faces:
            faces_data.append({
                "bbox": [int(x), int(y), int(w), int(h)],
                "confidence": 1.0  # YuNet confidence not returned in bbox mode
            })
        
        return DetectResponse(faces=faces_data, count=len(faces_data))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/recognize", response_model=RecognizeResponse)
async def recognize_face(request: RecognizeRequest):
    """
    Recognize a face in an image using Sface.
    Compares against registered visitors in database.
    """
    try:
        # Convert base64 to image
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
                visitor_info=None,
                matched=False
            )
        
        # Extract features from first face
        face_data = faces_data[0]
        feature = detector.extract_face_features(image, face_data)
        
        if feature is None:
            return RecognizeResponse(
                visitor_id=None,
                confidence=None,
                visitor_info=None,
                matched=False
            )
        
        # Compare with database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT visitor_id, feature_vector FROM face_features")
            all_features = cursor.fetchall()
            
            best_match = None
            best_score = 0.0
            best_visitor_id = None
            
            for visitor_id, stored_feature_blob in all_features:
                stored_feature = np.frombuffer(stored_feature_blob, dtype=np.float32)
                score, is_match = detector.compare_faces(
                    feature,
                    stored_feature,
                    threshold=request.threshold
                )
                
                if score > best_score:
                    best_score = score
                    best_visitor_id = visitor_id
                    best_match = is_match
            
            # Get visitor info if matched
            visitor_info = None
            if best_match and best_visitor_id:
                cursor.execute(
                    "SELECT name, email, phone, registered_at FROM visitors WHERE visitor_id = %s",
                    (best_visitor_id,)
                )
                visitor_data = cursor.fetchone()
                
                if visitor_data:
                    visitor_info = {
                        "name": visitor_data[0],
                        "email": visitor_data[1],
                        "phone": visitor_data[2],
                        "registered_at": visitor_data[3].isoformat() if visitor_data[3] else None
                    }
                    
                    # Log visit
                    log_visit(best_visitor_id, best_score)
        finally:
            cursor.close()
            return_db_connection(conn)
        
        return RecognizeResponse(
            visitor_id=best_visitor_id if best_match else None,
            confidence=float(best_score) if best_match else None,
            visitor_info=visitor_info,
            matched=best_match
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/register", response_model=RegisterResponse)
async def register_visitor(request: RegisterRequest):
    """
    Register a new visitor with face images.
    Extracts features from multiple images and stores in database.
    """
    try:
        visitor_id = str(uuid.uuid4())
        
        # Save visitor info
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO visitors (visitor_id, name, email, phone) VALUES (%s, %s, %s, %s)",
                (visitor_id, request.name, request.email, request.phone)
            )
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        finally:
            cursor.close()
            return_db_connection(conn)
        
        # Process images and extract features
        features = []
        os.makedirs(f"visitor_images/{visitor_id}", exist_ok=True)
        
        for idx, image_base64 in enumerate(request.images):
            image = base64_to_image(image_base64)
            
            # Detect faces
            faces_data = detector.detect_faces(
                image,
                resize_factor=1.0,
                score_threshold=0.6,
                return_landmarks=True
            )
            
            if not faces_data:
                continue
            
            # Extract features
            face_data = faces_data[0]
            feature = detector.extract_face_features(image, face_data)
            
            if feature is not None:
                features.append(feature)
                
                # Save image
                image_path = f"visitor_images/{visitor_id}/image_{idx}.jpg"
                cv2.imwrite(image_path, image)
        
        # Average features for better accuracy
        if features:
            avg_feature = np.mean(features, axis=0)
            
            # Store feature
            conn = get_db_connection()
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "INSERT INTO face_features (feature_id, visitor_id, feature_vector, image_path) VALUES (%s, %s, %s, %s)",
                    (str(uuid.uuid4()), visitor_id, avg_feature.tobytes(), f"visitor_images/{visitor_id}/")
                )
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
            finally:
                cursor.close()
                return_db_connection(conn)
        
        return RegisterResponse(
            visitor_id=visitor_id,
            message=f"Visitor registered with {len(features)} face images"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/visitor/{visitor_id}")
async def get_visitor(visitor_id: str):
    """Get visitor information"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "SELECT name, email, phone, registered_at FROM visitors WHERE visitor_id = %s",
            (visitor_id,)
        )
        visitor = cursor.fetchone()
        
        if not visitor:
            raise HTTPException(status_code=404, detail="Visitor not found")
        
        # Get visit count
        cursor.execute(
            "SELECT COUNT(*) FROM visits WHERE visitor_id = %s",
            (visitor_id,)
        )
        visit_count = cursor.fetchone()[0]
        
        return {
            "visitor_id": visitor_id,
            "name": visitor[0],
            "email": visitor[1],
            "phone": visitor[2],
            "registered_at": visitor[3].isoformat() if visitor[3] else None,
            "visit_count": visit_count
        }
    finally:
        cursor.close()
        return_db_connection(conn)

def log_visit(visitor_id: str, confidence: float):
    """Log a visit to the database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO visits (visit_id, visitor_id, confidence) VALUES (%s, %s, %s)",
            (str(uuid.uuid4()), visitor_id, confidence)
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error logging visit: {e}")
    finally:
        cursor.close()
        return_db_connection(conn)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
