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
import numpy as np
from pathlib import Path
from typing import Optional
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
try:
    from dotenv import load_dotenv
    _SCRIPT_DIR = Path(__file__).parent
    # Check services/face-recognition/.env.test first (where this script lives)
    env_file = _SCRIPT_DIR.parent / ".env.test"
    if env_file.exists():
        print(f"Loading environment from: {env_file}")
        load_dotenv(env_file)
    else:
        # Fallback to project root
        env_file = _SCRIPT_DIR.parent.parent.parent / ".env.test"
        if env_file.exists():
            print(f"Loading environment from: {env_file}")
            load_dotenv(env_file)
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
    print("Make sure you're running this from the app directory and all dependencies are installed.")
    sys.exit(1)

# Configuration from environment
USE_DATABASE = os.environ.get("USE_DATABASE", "false").lower() == "true"
DB_TABLE_NAME = os.environ.get("DB_TABLE_NAME", 'public."Visitor"')
DB_IMAGE_COLUMN = os.environ.get("DB_IMAGE_COLUMN", "base64Image")
DB_FEATURES_COLUMN = os.environ.get("DB_FEATURES_COLUMN", "faceFeatures")
DB_VISITOR_ID_COLUMN = os.environ.get("DB_VISITOR_ID_COLUMN", "id")
MODELS_PATH = os.environ.get("MODELS_PATH", "models")
BATCH_SIZE = int(os.environ.get("EXTRACT_BATCH_SIZE", "1000"))  # Process in batches
SKIP_EXISTING = os.environ.get("SKIP_EXISTING_FEATURES", "true").lower() == "true"  # Skip if features already exist


def extract_feature_from_image(base64_image: str) -> Optional[np.ndarray]:
    """
    Extract 128-dim feature from base64-encoded image.
    
    Args:
        base64_image: Base64-encoded image string
        
    Returns:
        128-dim feature vector or None if extraction fails
    """
    try:
        # Load image from base64
        img_cv = image_loader.load_from_base64(base64_image)
        
        # Detect faces
        faces = inference.detect_faces(img_cv, return_landmarks=True)
        if faces is None or len(faces) == 0:
            return None
        
        # Extract 128-dim feature using Sface
        feature = inference.extract_face_features(img_cv, faces[0])
        if feature is None:
            return None
        
        # Validate dimension
        feature = np.asarray(feature).flatten().astype(np.float32)
        if feature.shape[0] != 128:
            print(f"[WARNING] Feature dimension is {feature.shape[0]}, expected 128")
            return None
        
        return feature
    except Exception as e:
        print(f"Error extracting feature: {e}")
        return None


def get_visitors_needing_features(skip_existing: bool = True) -> list:
    """
    Get visitors that need feature extraction.
    
    Args:
        skip_existing: If True, skip visitors that already have features
        
    Returns:
        List of visitor dictionaries
    """
    try:
        # Build query to get visitors
        case_sensitive_cols = ['id', 'base64Image', 'imageUrl', 'firstName', 'lastName', 'fullName', 'createdAt', 'updatedAt', 'faceFeatures']
        visitor_id_col = f'"{DB_VISITOR_ID_COLUMN}"' if DB_VISITOR_ID_COLUMN in case_sensitive_cols else DB_VISITOR_ID_COLUMN
        image_col = f'"{DB_IMAGE_COLUMN}"' if DB_IMAGE_COLUMN in case_sensitive_cols else DB_IMAGE_COLUMN
        features_col = f'"{DB_FEATURES_COLUMN}"' if DB_FEATURES_COLUMN in case_sensitive_cols else DB_FEATURES_COLUMN
        
        # Get database connection
        from psycopg2.extras import RealDictCursor
        conn = database.get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Build query - include firstName and lastName
        firstName_col = '"firstName"'
        lastName_col = '"lastName"'
        query = f"""
            SELECT {visitor_id_col}, {image_col}, {features_col}, {firstName_col}, {lastName_col}
            FROM {DB_TABLE_NAME}
            WHERE {image_col} IS NOT NULL
        """
        
        if skip_existing:
            query += f" AND ({features_col} IS NULL OR {features_col} = '')"
        
        query += f" ORDER BY {visitor_id_col}"
        
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in results]
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


def main():
    """Main function to extract and store features."""
    print("=" * 60)
    print("Face Features Extraction Script")
    print("=" * 60)
    print()
    
    # Check database availability
    if not USE_DATABASE:
        print("[ERROR] USE_DATABASE is not set to 'true'")
        print("   Set USE_DATABASE=true in your .env file")
        sys.exit(1)
    
    # Check if database module is available
    try:
        if not hasattr(database, 'test_connection'):
            print("[ERROR] Database module not properly loaded")
            sys.exit(1)
    except AttributeError:
        print("[ERROR] Database module not available")
        print("   Install psycopg2: pip install psycopg2-binary")
        sys.exit(1)
    
    # Test database connection
    print("Testing database connection...")
    if not database.test_connection():
        print("[ERROR] Database connection failed")
        print("   Check your DATABASE_URL and database credentials")
        sys.exit(1)
    print("[OK] Database connection successful")
    print()
    
    # Load models
    print("Loading face detection and recognition models...")
    try:
        detector = inference.get_face_detector(MODELS_PATH)
        recognizer = inference.get_face_recognizer(MODELS_PATH)
        print("[OK] Models loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        print(f"   Make sure models are in: {MODELS_PATH}")
        sys.exit(1)
    print()
    
    # Get visitors needing features
    print(f"Querying visitors {'without features' if SKIP_EXISTING else 'with images'}...")
    visitors = get_visitors_needing_features(skip_existing=SKIP_EXISTING)
    
    if not visitors:
        print("[OK] No visitors need feature extraction")
        if SKIP_EXISTING:
            print("   All visitors already have features stored")
        else:
            print("   No visitors found in database")
        return
    
    total_visitors = len(visitors)
    print(f"Found {total_visitors} visitors needing feature extraction")
    print()
    
    # Process visitors
    successful = 0
    failed = 0
    skipped = 0
    start_time = time.time()
    
    print(f"Processing visitors (batch size: {BATCH_SIZE})...")
    print("-" * 60)
    
    for i, visitor_data in enumerate(visitors, 1):
        visitor_id = visitor_data.get(DB_VISITOR_ID_COLUMN, 'unknown')
        base64_image = visitor_data.get(DB_IMAGE_COLUMN)
        
        # Skip if no image
        if not base64_image:
            skipped += 1
            continue
        
        # Extract feature
        feature = extract_feature_from_image(base64_image)
        
        if feature is None:
            failed += 1
            print(f"[{i}/{total_visitors}] ✗ Failed to extract feature for visitor {visitor_id}")
            continue
        
        # Save to database
        if save_feature_to_db(visitor_id, feature):
            successful += 1
            if i % 10 == 0 or i == total_visitors:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (total_visitors - i) / rate if rate > 0 else 0
                print(f"[{i}/{total_visitors}] [OK] Extracted and saved feature for visitor {visitor_id} "
                      f"(Success: {successful}, Failed: {failed}, Rate: {rate:.1f}/s, ETA: {remaining:.0f}s)")
        else:
            failed += 1
            print(f"[{i}/{total_visitors}] ✗ Failed to save feature for visitor {visitor_id}")
    
    # Summary
    elapsed_time = time.time() - start_time
    print()
    print("=" * 60)
    print("Extraction Summary")
    print("=" * 60)
    print(f"Total visitors processed: {total_visitors}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Time elapsed: {elapsed_time:.1f} seconds")
    if successful > 0:
        print(f"Average rate: {successful/elapsed_time:.2f} features/second")
    print()
    
    if successful > 0:
        print(f"[OK] Successfully extracted and stored features for {successful} visitors")
    if failed > 0:
        print(f"[WARNING] Failed to extract features for {failed} visitors")
        print("   Check logs above for error details")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[WARNING] Extraction interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
