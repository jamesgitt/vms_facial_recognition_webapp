"""
Visitor Loader Pipeline

Loading visitors from database or test images into HNSW index.
"""

import os
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np

from core.logger import get_logger
from core.config import settings
from core.state import app_state

logger = get_logger(__name__)

# Import ML modules
from ml import inference
from utils import image_loader
from db import database

# Import feature extraction
from .feature_extraction import extract_feature_from_visitor_data


@dataclass
class VisitorData:
    """Visitor data for indexing."""
    visitor_id: str
    feature: np.ndarray
    firstName: Optional[str] = None
    lastName: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LoadResult:
    """Result of visitor loading operation."""
    success: bool
    count: int = 0
    source: str = "unknown"
    error: Optional[str] = None


def init_database_connection() -> bool:
    """
    Initialize database connection.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Connecting to database...")
        if database.test_connection():
            logger.info("Database connection successful")
            db_config = settings.database
            database.init_connection_pool(
                min_conn=db_config.pool_min_conn,
                max_conn=db_config.pool_max_conn
            )
            app_state.use_database = True
            return True
        else:
            logger.warning("Database connection failed")
            app_state.use_database = False
            return False
    except Exception as e:
        logger.error(f"Database error: {e}")
        app_state.use_database = False
        return False


def load_visitors_from_database(
    hnsw_manager: Optional[Any] = None,
) -> LoadResult:
    """
    Load visitors from database and build HNSW index.
    
    Args:
        hnsw_manager: HNSW index manager (uses app_state if None)
    
    Returns:
        LoadResult with count of loaded visitors
    """
    manager = hnsw_manager or app_state.hnsw_manager
    db_config = settings.database
    
    try:
        visitors = database.get_visitor_images_from_db(
            table_name=db_config.table_name,
            visitor_id_column=db_config.visitor_id_column,
            image_column=db_config.image_column,
            features_column=db_config.features_column,
            limit=db_config.visitor_limit,
        )
        
        logger.info(f"Found {len(visitors)} visitors in database")
        
        if manager and visitors:
            logger.info(f"Building HNSW index from {len(visitors)} visitors...")
            count = manager.rebuild_from_database(
                get_visitors_func=lambda: visitors,
                extract_feature_func=extract_feature_from_visitor_data
            )
            if count > 0:
                logger.info(f"HNSW index built with {count} visitors")
            return LoadResult(
                success=True,
                count=count,
                source="database"
            )
        
        return LoadResult(
            success=True,
            count=len(visitors),
            source="database"
        )
        
    except Exception as e:
        logger.error(f"Loading visitors from database: {e}")
        return LoadResult(
            success=False,
            error=str(e),
            source="database"
        )


def load_visitors_from_test_images(
    hnsw_manager: Optional[Any] = None,
    images_dir: Optional[str] = None,
) -> LoadResult:
    """
    Load visitors from test_images directory.
    
    Args:
        hnsw_manager: HNSW index manager (uses app_state if None)
        images_dir: Path to images directory (uses default if None)
    
    Returns:
        LoadResult with count of loaded visitors
    """
    manager = hnsw_manager or app_state.hnsw_manager
    
    # Determine images directory
    if images_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        images_dir = os.path.join(script_dir, "..", "..", "test_images")
    
    if not os.path.isdir(images_dir):
        return LoadResult(
            success=False,
            error=f"Directory not found: {images_dir}",
            source="test_images"
        )
    
    logger.info(f"Loading visitors from {images_dir}")
    
    allowed_formats = settings.image.allowed_formats
    batch_data: List[Tuple[str, np.ndarray, Dict[str, Any]]] = []
    
    # Clear existing cache
    app_state.visitor_features.clear()
    
    for fname in os.listdir(images_dir):
        if not fname.lower().endswith(tuple(allowed_formats)):
            continue
        
        fpath = os.path.join(images_dir, fname)
        
        try:
            img_cv = image_loader.load_from_path(fpath)
            faces = inference.detect_faces(img_cv, return_landmarks=True)
            
            if faces is None or len(faces) == 0:
                continue
            
            feature = inference.extract_face_features(img_cv, faces[0])
            if feature is None:
                continue
            
            feature = np.asarray(feature).flatten().astype(np.float32)
            if feature.shape[0] != 128:
                continue
            
            visitor_name = os.path.splitext(fname)[0]
            
            # Store in cache
            app_state.visitor_features[visitor_name] = {
                "feature": feature,
                "path": fpath
            }
            
            # Prepare for batch add to HNSW
            if manager:
                batch_data.append((visitor_name, feature, {"path": fpath}))
                
        except Exception as e:
            logger.warning(f"Failed to process {fname}: {e}")
    
    count = len(app_state.visitor_features)
    logger.info(f"Loaded {count} visitors from test_images")
    
    # Add to HNSW index
    if manager and batch_data:
        hnsw_count = manager.add_visitors_batch(batch_data)
        if hnsw_count > 0:
            manager.save()
            logger.info(f"HNSW index built with {hnsw_count} test_images visitors")
    
    return LoadResult(
        success=True,
        count=count,
        source="test_images"
    )


def init_hnsw_index() -> Optional[Any]:
    """
    Initialize HNSW index manager.
    
    Returns:
        HNSWIndexManager instance or None
    """
    try:
        from ml.hnsw_index import HNSWIndexManager
    except ImportError:
        logger.warning("HNSW index not available")
        return None
    
    hnsw_config = settings.hnsw
    
    try:
        index_dir = hnsw_config.index_dir or settings.models.models_path
        manager = HNSWIndexManager(
            index_dir=index_dir,
            max_elements=hnsw_config.max_elements,
            m=hnsw_config.m,
            ef_construction=hnsw_config.ef_construction,
            ef_search=hnsw_config.ef_search,
        )
        logger.info(f"HNSW index initialized (max_elements={hnsw_config.max_elements})")
        return manager
    except Exception as e:
        logger.warning(f"HNSW init error: {e}")
        return None


def initialize_all() -> None:
    """
    Initialize all components: database, HNSW index, and load visitors.
    
    This is the main initialization function called on startup.
    """
    # Initialize HNSW index
    app_state.hnsw_manager = init_hnsw_index()
    
    # Initialize database connection
    if settings.database.use_database:
        init_database_connection()
    
    # Check if HNSW index already has data
    index_has_data = False
    if app_state.hnsw_manager:
        try:
            count = app_state.hnsw_manager.ntotal
            if count > 0:
                logger.info(f"HNSW index already has {count} vectors - skipping reload")
                index_has_data = True
        except Exception as e:
            logger.debug(f"Could not check index count: {e}")
    
    # Load visitors from database or test images
    if not index_has_data:
        if app_state.use_database:
            logger.info("Using database for visitor recognition")
            result = load_visitors_from_database()
            if result.count == 0:
                logger.warning("No visitors loaded from database - falling back to test_images")
                app_state.use_database = False
        
        if not app_state.use_database:
            load_visitors_from_test_images()
    
    app_state.initialized = True


__all__ = [
    "VisitorData",
    "LoadResult",
    "init_database_connection",
    "load_visitors_from_database",
    "load_visitors_from_test_images",
    "init_hnsw_index",
    "initialize_all",
]
