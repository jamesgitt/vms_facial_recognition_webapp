"""
Manual HNSW Index Rebuild Script

Rebuilds the HNSW index from the database without starting the full API server.
Run this script whenever you need to rebuild the index after adding/updating visitors.

Usage:
    python rebuild_index.py

Environment Variables:
    USE_DATABASE: Set to "true" to use database (default: auto-detect)
    DB_TABLE_NAME: Database table name (default: public."Visitor")
    DB_VISITOR_ID_COLUMN: Visitor ID column name (default: id)
    DB_IMAGE_COLUMN: Image column name (default: base64Image)
    DB_FEATURES_COLUMN: Features column name (default: facefeatures)
    MODELS_PATH: Path to models directory (default: models)
"""

import os
import sys
import base64
from pathlib import Path
from typing import Optional

import numpy as np

# Add the app directory to Python path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# Load environment variables
try:
    from dotenv import load_dotenv
    for env_path in [SCRIPT_DIR.parent / ".env.test", SCRIPT_DIR / ".env.test"]:
        if env_path.exists():
            load_dotenv(env_path)
            break
except ImportError:
    pass

# Import modules
try:
    import database
except ImportError:
    print("[ERROR] database module not available")
    sys.exit(1)

try:
    from hnsw_index import HNSWIndexManager
except ImportError:
    print("[ERROR] HNSW index manager not available")
    sys.exit(1)

import inference
import image_loader

# Configuration
DB_TABLE_NAME = os.environ.get("DB_TABLE_NAME", 'public."Visitor"')
DB_VISITOR_ID_COLUMN = os.environ.get("DB_VISITOR_ID_COLUMN", "id")
DB_IMAGE_COLUMN = os.environ.get("DB_IMAGE_COLUMN", "base64Image")
DB_FEATURES_COLUMN = os.environ.get("DB_FEATURES_COLUMN", "facefeatures")
MODELS_PATH = os.environ.get("MODELS_PATH", "models")
EXPECTED_FEATURE_DIM = 128

# Lazy-loaded models
_face_detector = None
_face_recognizer = None


def _get_models():
    """Get or initialize face detection/recognition models."""
    global _face_detector, _face_recognizer
    if _face_detector is None or _face_recognizer is None:
        print("   Loading face detection and recognition models...")
        _face_detector = inference.get_face_detector(MODELS_PATH)
        _face_recognizer = inference.get_face_recognizer(MODELS_PATH)
    return _face_detector, _face_recognizer


def _decode_stored_features(features_data) -> Optional[np.ndarray]:
    """Decode stored features from database."""
    if isinstance(features_data, str):
        try:
            features_bytes = base64.b64decode(features_data)
            features = np.frombuffer(features_bytes, dtype=np.float32)
            if len(features) == EXPECTED_FEATURE_DIM:
                return features
        except Exception:
            pass
    elif isinstance(features_data, (list, tuple)):
        features = np.array(features_data, dtype=np.float32)
        if len(features) == EXPECTED_FEATURE_DIM:
            return features.flatten()
    return None


def _extract_from_image(visitor_data: dict) -> Optional[np.ndarray]:
    """Extract features from visitor's base64 image."""
    base64_image = visitor_data.get(DB_IMAGE_COLUMN)
    if not base64_image:
        return None
    
    try:
        image = image_loader.load_from_base64(base64_image)
        faces = inference.detect_faces(image, return_landmarks=True)
        
        if not faces:
            return None
        
        feature = inference.extract_face_features(image, faces[0])
        if feature is None:
            return None
        
        feature = np.asarray(feature).flatten().astype(np.float32)
        if len(feature) != EXPECTED_FEATURE_DIM:
            return None
        
        # Save features back to database
        visitor_id = visitor_data.get(DB_VISITOR_ID_COLUMN)
        if visitor_id:
            database.update_visitor_features(
                visitor_id=str(visitor_id),
                features=feature,
                table_name=DB_TABLE_NAME,
                visitor_id_column=DB_VISITOR_ID_COLUMN,
                features_column=DB_FEATURES_COLUMN
            )
        
        return feature
    except Exception as e:
        print(f"   [WARNING] Failed to extract features from image: {e}")
        return None


def extract_feature_from_visitor_data(visitor_data: dict) -> Optional[np.ndarray]:
    """
    Extract 128-dim feature from visitor data.
    
    Priority:
    1. Use faceFeatures column if available (stored features)
    2. Extract from base64Image and save to faceFeatures column
    """
    # Try stored features first
    if DB_FEATURES_COLUMN in visitor_data and visitor_data[DB_FEATURES_COLUMN]:
        feature = _decode_stored_features(visitor_data[DB_FEATURES_COLUMN])
        if feature is not None:
            return feature
    
    # Fallback: extract from image
    return _extract_from_image(visitor_data)


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("=" * 60)
    print(title)
    print("=" * 60)


def check_database_connection() -> bool:
    """Check and initialize database connection."""
    print("\n1. Checking database connection...")
    
    if not database.test_connection():
        print("   [ERROR] Database connection failed")
        return False
    
    print("   [OK] Database connection successful")
    database.init_connection_pool(min_conn=1, max_conn=5)
    return True


def load_visitors() -> Optional[list]:
    """Load visitors from database."""
    print("\n2. Loading visitors from database...")
    
    try:
        visitors = database.get_visitor_images_from_db(
            table_name=DB_TABLE_NAME,
            visitor_id_column=DB_VISITOR_ID_COLUMN,
            image_column=DB_IMAGE_COLUMN,
            features_column=DB_FEATURES_COLUMN
        )
        print(f"   [OK] Found {len(visitors)} visitors in database")
        
        if not visitors:
            print("   [WARNING] No visitors found. Index will be empty.")
            return None
        
        return visitors
    except Exception as e:
        print(f"   [ERROR] Error loading visitors: {e}")
        return None


def initialize_index() -> Optional[HNSWIndexManager]:
    """Initialize HNSW index manager."""
    print("\n3. Initializing HNSW index manager...")
    
    try:
        manager = HNSWIndexManager(
            dimension=EXPECTED_FEATURE_DIM,
            index_dir=MODELS_PATH
        )
        print("   [OK] HNSW index manager initialized")
        return manager
    except Exception as e:
        print(f"   [ERROR] Error initializing HNSW index manager: {e}")
        return None


def rebuild_index(manager: HNSWIndexManager, visitors: list) -> bool:
    """Rebuild HNSW index from visitors."""
    print(f"\n4. Rebuilding HNSW index from {len(visitors)} visitors...")
    print("   This may take a while...")
    
    try:
        count = manager.rebuild_from_database(
            get_visitors_func=lambda: visitors,
            extract_feature_func=extract_feature_from_visitor_data
        )
        
        if count == 0:
            print("\n[ERROR] Index rebuild failed - no visitors were indexed")
            return False
        
        print(f"\n[OK] HNSW index rebuilt with {count} visitors")
        print(f"   Index files saved to: {Path(MODELS_PATH).absolute()}")
        print("   - hnsw_visitor_index.bin")
        print("   - hnsw_visitor_metadata.pkl")
        
        # Display stats
        stats = manager.get_stats()
        print("\nIndex Statistics:")
        print(f"   Visitors indexed: {stats['visitors_indexed']}")
        print(f"   Total vectors: {stats['total_vectors']}")
        print(f"   HNSW Parameters: M={stats['m']}, ef_construction={stats['ef_construction']}, ef_search={stats['ef_search']}")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Error rebuilding index: {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> int:
    """Main function to rebuild the HNSW index."""
    print_header("HNSW Index Rebuild Script")
    
    # Step 1: Check database
    if not check_database_connection():
        return 1
    
    # Step 2: Load visitors
    visitors = load_visitors()
    if not visitors:
        return 1
    
    # Step 3: Initialize index manager
    manager = initialize_index()
    if not manager:
        return 1
    
    # Step 4: Rebuild index
    if not rebuild_index(manager, visitors):
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
