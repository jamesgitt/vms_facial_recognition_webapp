"""
Extract Face Features and Store in Database

This script extracts 128-dim face features from images in the database
and stores them in the faceFeatures column. Useful for:
- Initial population of faceFeatures column
- Re-extracting features for existing visitors
- Batch processing large databases

Usage:
    python extract_features_to_db.py

Environment Variables:
    USE_DATABASE=true
    DATABASE_URL=postgresql://user:password@host:5432/database
    DB_TABLE_NAME=public."Visitor"
    DB_IMAGE_COLUMN=base64Image
    DB_FEATURES_COLUMN=faceFeatures
    DB_VISITOR_ID_COLUMN=id
    MODELS_PATH=models
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
from psycopg2.extras import RealDictCursor

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
try:
    from dotenv import load_dotenv
    _SCRIPT_DIR = Path(__file__).parent
    for env_path in [
        _SCRIPT_DIR.parent / ".env.test",
        _SCRIPT_DIR.parent.parent.parent / ".env.test",
    ]:
        if env_path.exists():
            print(f"Loading environment from: {env_path}")
            load_dotenv(env_path)
            break
    else:
        print("Warning: No .env.test file found")
except ImportError:
    pass

# Import required modules
try:
    import database
    import inference
    import image_loader
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the app directory with all dependencies installed.")
    sys.exit(1)

# Constants
EXPECTED_FEATURE_DIM = 128
LOG_INTERVAL = 10

# Configuration from environment
USE_DATABASE = os.environ.get("USE_DATABASE", "false").lower() == "true"
DB_TABLE_NAME = os.environ.get("DB_TABLE_NAME", 'public."Visitor"')
DB_IMAGE_COLUMN = os.environ.get("DB_IMAGE_COLUMN", "base64Image")
DB_FEATURES_COLUMN = os.environ.get("DB_FEATURES_COLUMN", "faceFeatures")
DB_VISITOR_ID_COLUMN = os.environ.get("DB_VISITOR_ID_COLUMN", "id")
MODELS_PATH = os.environ.get("MODELS_PATH", "models")
SKIP_EXISTING = os.environ.get("SKIP_EXISTING_FEATURES", "true").lower() == "true"


def extract_feature_from_image(base64_image: str) -> Optional[np.ndarray]:
    """
    Extract 128-dim feature from base64-encoded image.
    
    Args:
        base64_image: Base64-encoded image string
        
    Returns:
        128-dim feature vector or None if extraction fails
    """
    try:
        img_cv = image_loader.load_from_base64(base64_image)
        
        faces = inference.detect_faces(img_cv, return_landmarks=True)
        if not faces:
            return None
        
        feature = inference.extract_face_features(img_cv, faces[0])
        if feature is None:
            return None
        
        feature = np.asarray(feature).flatten().astype(np.float32)
        if feature.shape[0] != EXPECTED_FEATURE_DIM:
            print(f"[WARNING] Feature dimension is {feature.shape[0]}, expected {EXPECTED_FEATURE_DIM}")
            return None
        
        return feature
    except Exception as e:
        print(f"Error extracting feature: {e}")
        return None


def get_visitors_needing_features(skip_existing: bool = True) -> List[Dict]:
    """
    Get visitors that need feature extraction.
    
    Args:
        skip_existing: If True, skip visitors that already have features
        
    Returns:
        List of visitor dictionaries
    """
    try:
        id_col = database._quote_column(DB_VISITOR_ID_COLUMN)
        img_col = database._quote_column(DB_IMAGE_COLUMN)
        feat_col = database._quote_column(DB_FEATURES_COLUMN)
        
        query = f"""
            SELECT {id_col}, {img_col}, {feat_col}, "firstName", "lastName"
            FROM {DB_TABLE_NAME}
            WHERE {img_col} IS NOT NULL
        """
        
        if skip_existing:
            query += f" AND ({feat_col} IS NULL OR {feat_col} = '')"
        
        query += f" ORDER BY {id_col}"
        
        with database.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query)
            return [dict(row) for row in cursor.fetchall()]
            
    except Exception as e:
        print(f"Error querying database: {e}")
        return []


def save_feature_to_db(visitor_id: str, feature: np.ndarray) -> bool:
    """
    Save feature to database faceFeatures column.
    
    Args:
        visitor_id: Visitor ID
        feature: 128-dim feature vector
        
    Returns:
        True if successful, False otherwise
    """
    try:
        return database.update_visitor_features(
            visitor_id=str(visitor_id),
            features=feature,
            table_name=DB_TABLE_NAME,
            visitor_id_column=DB_VISITOR_ID_COLUMN,
            features_column=DB_FEATURES_COLUMN
        )
    except Exception as e:
        print(f"Error saving feature to database: {e}")
        return False


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("=" * 60)
    print(title)
    print("=" * 60)


def validate_environment() -> bool:
    """Validate that required environment and modules are available."""
    if not USE_DATABASE:
        print("[ERROR] USE_DATABASE is not set to 'true'")
        print("   Set USE_DATABASE=true in your .env file")
        return False
    
    if not hasattr(database, 'test_connection'):
        print("[ERROR] Database module not properly loaded")
        return False
    
    print("Testing database connection...")
    if not database.test_connection():
        print("[ERROR] Database connection failed")
        print("   Check your DATABASE_URL and database credentials")
        return False
    print("[OK] Database connection successful\n")
    
    print("Loading face detection and recognition models...")
    try:
        inference.get_face_detector(MODELS_PATH)
        inference.get_face_recognizer(MODELS_PATH)
        print("[OK] Models loaded successfully\n")
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        print(f"   Make sure models are in: {MODELS_PATH}")
        return False
    
    return True


def process_visitors(visitors: List[Dict]) -> tuple:
    """
    Process visitors and extract features.
    
    Args:
        visitors: List of visitor dictionaries
        
    Returns:
        Tuple of (successful, failed, skipped) counts
    """
    successful = failed = skipped = 0
    total = len(visitors)
    start_time = time.time()
    
    print("-" * 60)
    
    for i, visitor_data in enumerate(visitors, 1):
        visitor_id = visitor_data.get(DB_VISITOR_ID_COLUMN, 'unknown')
        base64_image = visitor_data.get(DB_IMAGE_COLUMN)
        
        if not base64_image:
            skipped += 1
            continue
        
        feature = extract_feature_from_image(base64_image)
        
        if feature is None:
            failed += 1
            print(f"[{i}/{total}] FAILED - visitor {visitor_id}")
            continue
        
        if save_feature_to_db(visitor_id, feature):
            successful += 1
            if i % LOG_INTERVAL == 0 or i == total:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total - i) / rate if rate > 0 else 0
                print(f"[{i}/{total}] [OK] visitor {visitor_id} "
                      f"(Success: {successful}, Failed: {failed}, Rate: {rate:.1f}/s, ETA: {eta:.0f}s)")
        else:
            failed += 1
            print(f"[{i}/{total}] FAILED to save - visitor {visitor_id}")
    
    return successful, failed, skipped


def print_summary(total: int, successful: int, failed: int, skipped: int, elapsed: float) -> None:
    """Print extraction summary."""
    print()
    print_header("Extraction Summary")
    print(f"Total visitors processed: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Time elapsed: {elapsed:.1f} seconds")
    if successful > 0 and elapsed > 0:
        print(f"Average rate: {successful / elapsed:.2f} features/second")
    print()
    
    if successful > 0:
        print(f"[OK] Successfully extracted and stored features for {successful} visitors")
    if failed > 0:
        print(f"[WARNING] Failed to extract features for {failed} visitors")
        print("   Check logs above for error details")


def main() -> int:
    """
    Main function to extract and store features.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    print_header("Face Features Extraction Script")
    print()
    
    if not validate_environment():
        return 1
    
    print(f"Querying visitors {'without features' if SKIP_EXISTING else 'with images'}...")
    visitors = get_visitors_needing_features(skip_existing=SKIP_EXISTING)
    
    if not visitors:
        print("[OK] No visitors need feature extraction")
        if SKIP_EXISTING:
            print("   All visitors already have features stored")
        else:
            print("   No visitors found in database")
        return 0
    
    total = len(visitors)
    print(f"Found {total} visitors needing feature extraction\n")
    
    start_time = time.time()
    successful, failed, skipped = process_visitors(visitors)
    elapsed = time.time() - start_time
    
    print_summary(total, successful, failed, skipped, elapsed)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n[WARNING] Extraction interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
