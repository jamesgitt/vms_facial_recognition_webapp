"""
Manual HNSW Index Rebuild Script

This script rebuilds the HNSW index from the database without starting the full API server.
Run this script whenever you need to rebuild the index after adding/updating visitors.

Usage:
    python rebuild_index.py

Environment Variables:
    - USE_DATABASE: Set to "true" to use database (default: auto-detect)
    - DB_TABLE_NAME: Database table name (default: public."Visitor")
    - DB_VISITOR_ID_COLUMN: Visitor ID column name (default: id)
    - DB_IMAGE_COLUMN: Image column name (default: base64Image)
    - DB_FEATURES_COLUMN: Features column name (default: facefeatures)
    - MODELS_PATH: Path to models directory (default: models)
    - HNSW_INDEX_FILE: Index filename (default: hnsw_visitor_index.bin)
    - HNSW_METADATA_FILE: Metadata filename (default: hnsw_visitor_metadata.pkl)
"""

import os
import sys
from pathlib import Path

# Add the app directory to Python path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# Load environment variables
try:
    from dotenv import load_dotenv
    env_file = SCRIPT_DIR.parent / ".env.test"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv(SCRIPT_DIR / ".env.test")
except ImportError:
    pass

# Import modules
try:
    import database
    DB_AVAILABLE = True
except ImportError:
    print("âŒ Error: database module not available")
    sys.exit(1)

try:
    from hnsw_index import HNSWIndexManager
    HNSW_AVAILABLE = True
except ImportError:
    print("âŒ Error: HNSW index manager not available")
    sys.exit(1)

import inference

# Configuration
USE_DATABASE = os.environ.get("USE_DATABASE", "false").lower() == "true"
DB_TABLE_NAME = os.environ.get("DB_TABLE_NAME", 'public."Visitor"')
DB_VISITOR_ID_COLUMN = os.environ.get("DB_VISITOR_ID_COLUMN", "id")
DB_IMAGE_COLUMN = os.environ.get("DB_IMAGE_COLUMN", "base64Image")
DB_FEATURES_COLUMN = os.environ.get("DB_FEATURES_COLUMN", "facefeatures")
MODELS_PATH = os.environ.get("MODELS_PATH", "models")

def extract_feature_from_visitor_data(visitor_data):
    """
    Extract 128-dim feature from visitor data.
    Priority:
    1. Use faceFeatures column if available (stored features)
    2. Extract from base64Image and save to faceFeatures column
    """
    import base64
    import numpy as np
    
    # Try to use stored features first
    if DB_FEATURES_COLUMN in visitor_data and visitor_data[DB_FEATURES_COLUMN]:
        features_data = visitor_data[DB_FEATURES_COLUMN]
        
        # Handle base64-encoded features
        if isinstance(features_data, str):
            try:
                features_bytes = base64.b64decode(features_data)
                features = np.frombuffer(features_bytes, dtype=np.float32)
                # Ensure it's 128-dim
                if len(features) == 128:
                    return features
            except Exception as e:
                print(f"âš ï¸  Warning: Failed to decode stored features: {e}")
        
        # Handle numpy array or list
        elif isinstance(features_data, (list, tuple)):
            features = np.array(features_data, dtype=np.float32)
            if len(features) == 128:
                return features.flatten()
    
    # Fallback: extract from image
    if DB_IMAGE_COLUMN in visitor_data and visitor_data[DB_IMAGE_COLUMN]:
        try:
            import image_loader
            image = image_loader.load_image_from_base64(visitor_data[DB_IMAGE_COLUMN])
            
            # Load models if not already loaded
            global face_detector, face_recognizer
            if face_detector is None or face_recognizer is None:
                print("Loading face detection and recognition models...")
                face_detector = inference.get_face_detector(MODELS_PATH)
                face_recognizer = inference.get_face_recognizer(MODELS_PATH)
            
            # Detect and extract features
            faces = inference.detect_faces(face_detector, image)
            if faces and len(faces) > 0:
                feature = inference.extract_features(face_recognizer, image, faces[0])
                if feature is not None:
                    # Save features back to database
                    visitor_id = visitor_data.get(DB_VISITOR_ID_COLUMN)
                    if visitor_id:
                        feature_base64 = base64.b64encode(feature.tobytes()).decode('utf-8')
                        database.update_visitor_features(
                            visitor_id=str(visitor_id),
                            features=feature_base64
                        )
                    return feature
        except Exception as e:
            print(f"âš ï¸  Warning: Failed to extract features from image: {e}")
    
    return None

def main():
    """Main function to rebuild the HNSW index."""
    global face_detector, face_recognizer
    face_detector = None
    face_recognizer = None
    
    print("=" * 60)
    print("HNSW Index Rebuild Script")
    print("=" * 60)
    
    # Check database connection
    print("\n1. Checking database connection...")
    if not DB_AVAILABLE:
        print("âŒ Database module not available")
        return False
    
    # Auto-detect database if not explicitly enabled
    if not USE_DATABASE:
        print("   Auto-detecting database connection...")
        if database.test_connection():
            print("   âœ“ Database connection successful")
            database.init_connection_pool(min_conn=1, max_conn=5)
        else:
            print("   âŒ Database connection failed")
            return False
    else:
        if not database.test_connection():
            print("   âŒ Database connection failed")
            return False
        database.init_connection_pool(min_conn=1, max_conn=5)
    
    # Load visitors from database
    print("\n2. Loading visitors from database...")
    try:
        visitors = database.get_visitor_images_from_db(
            table_name=DB_TABLE_NAME,
            visitor_id_column=DB_VISITOR_ID_COLUMN,
            image_column=DB_IMAGE_COLUMN,
            features_column=DB_FEATURES_COLUMN
        )
        print(f"   âœ“ Found {len(visitors)} visitors in database")
    except Exception as e:
        print(f"   âŒ Error loading visitors: {e}")
        return False
    
    if len(visitors) == 0:
        print("   âš ï¸  No visitors found in database. Index will be empty.")
        return False
    
    # Initialize HNSW index manager
    print("\n3. Initializing HNSW index manager...")
    try:
        hnsw_index_manager = HNSWIndexManager(
            dimension=128,  # Sface model output dimension
            index_dir=MODELS_PATH
        )
        print("   âœ“ HNSW index manager initialized")
    except Exception as e:
        print(f"   âŒ Error initializing HNSW index manager: {e}")
        return False
    
    # Rebuild index from database
    print("\n4. Rebuilding HNSW index from database...")
    print(f"   This may take a while for {len(visitors)} visitors...")
    try:
        def get_visitors():
            return visitors
        
        count = hnsw_index_manager.rebuild_from_database(
            get_visitors_func=get_visitors,
            extract_feature_func=extract_feature_from_visitor_data
        )
        
        if count > 0:
            print(f"\nâœ… Success! HNSW index rebuilt with {count} visitors")
            print(f"   Index files saved to: {Path(MODELS_PATH).absolute()}")
            print(f"   - hnsw_visitor_index.bin")
            print(f"   - hnsw_visitor_metadata.pkl")
            
            # Display stats
            stats = hnsw_index_manager.get_stats()
            print(f"\nðŸ“Š Index Statistics:")
            print(f"   - Visitors indexed: {stats['visitors_indexed']}")
            print(f"   - Total vectors: {stats['total_vectors']}")
            print(f"   - HNSW Parameters: M={stats.get('m', 'N/A')}, ef_construction={stats.get('ef_construction', 'N/A')}, ef_search={stats.get('ef_search', 'N/A')}")
            return True
        else:
            print("\nâŒ Index rebuild failed - no visitors were indexed")
            return False
    except Exception as e:
        print(f"\nâŒ Error rebuilding index: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
