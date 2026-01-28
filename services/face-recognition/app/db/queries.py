"""
Database Query Functions

Contains all SQL query functions for visitor data operations.
Separated from connection management for better organization.
"""

import base64
from typing import List, Dict, Optional

import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np

from .connection import get_connection
from core.logger import get_logger

logger = get_logger(__name__)

# Case-sensitive column names that require quoting in PostgreSQL
CASE_SENSITIVE_COLUMNS = frozenset([
    'id', 'base64Image', 'imageUrl', 'firstName', 'lastName',
    'fullName', 'createdAt', 'updatedAt', 'faceFeatures'
])

# Default table configuration
DEFAULT_TABLE_NAME = 'public."Visitor"'
DEFAULT_ID_COLUMN = "id"
DEFAULT_IMAGE_COLUMN = "base64Image"
DEFAULT_FEATURES_COLUMN = "faceFeatures"


def _quote_column(column: str) -> str:
    """Quote column name if it's case-sensitive."""
    return f'"{column}"' if column in CASE_SENSITIVE_COLUMNS else column


def get_visitor_images_from_db(
    table_name: str = DEFAULT_TABLE_NAME,
    visitor_id_column: str = DEFAULT_ID_COLUMN,
    image_column: str = DEFAULT_IMAGE_COLUMN,
    features_column: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Dict]:
    """
    Query visitor images and features from the database.
    
    Args:
        table_name: Name of the visitors table
        visitor_id_column: Column name for visitor ID
        image_column: Column name for base64 image
        features_column: Column name for face features (optional)
        limit: Maximum number of visitors to retrieve (None = all)
    
    Returns:
        List of dictionaries with visitor data including id, base64Image,
        firstName, lastName, and optionally faceFeatures
    """
    with get_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Build SELECT columns
        id_col = _quote_column(visitor_id_column)
        img_col = _quote_column(image_column)
        
        select_cols = [id_col, img_col, '"firstName"', '"lastName"']
        if features_column:
            select_cols.append(_quote_column(features_column))
        
        # Build query
        query = f"""
            SELECT {', '.join(select_cols)}
            FROM {table_name}
            WHERE {img_col} IS NOT NULL
            ORDER BY {id_col}
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        try:
            cursor.execute(query)
            results = cursor.fetchall()
            visitors = [dict(row) for row in results]
            logger.info(f"Retrieved {len(visitors)} visitors from database")
            return visitors
        except psycopg2.Error as e:
            logger.error(f"Database error: {e}")
            raise


def get_visitors_with_features_only(
    table_name: str = DEFAULT_TABLE_NAME,
    visitor_id_column: str = DEFAULT_ID_COLUMN,
    features_column: str = "facefeatures",
    batch_size: int = 5000
) -> List[Dict]:
    """
    Memory-efficient query that only loads visitors with pre-computed features.
    Does NOT load base64 images to save memory.
    
    Args:
        table_name: Name of the visitors table
        visitor_id_column: Column name for visitor ID
        features_column: Column name for face features
        batch_size: Number of records to fetch per batch
    
    Returns:
        List of dictionaries with id, firstName, lastName, and facefeatures
    """
    visitors = []
    offset = 0
    
    with get_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        id_col = _quote_column(visitor_id_column)
        feat_col = _quote_column(features_column)
        
        while True:
            query = f"""
                SELECT {id_col}, "firstName", "lastName", {feat_col}
                FROM {table_name}
                WHERE {feat_col} IS NOT NULL
                ORDER BY {id_col}
                LIMIT {batch_size} OFFSET {offset}
            """
            
            try:
                cursor.execute(query)
                batch = cursor.fetchall()
                
                if not batch:
                    break
                
                visitors.extend([dict(row) for row in batch])
                offset += batch_size
                
                if len(batch) < batch_size:
                    break
                    
            except psycopg2.Error as e:
                logger.error(f"Database error: {e}")
                raise
    
    logger.info(f"Retrieved {len(visitors)} visitors with features from database")
    return visitors


def get_visitor_details(
    visitor_id: str,
    table_name: str = DEFAULT_TABLE_NAME,
    visitor_id_column: str = DEFAULT_ID_COLUMN
) -> Optional[Dict]:
    """
    Get full visitor details by visitor ID.
    
    Args:
        visitor_id: The visitor ID to look up
        table_name: Name of the visitors table
        visitor_id_column: Column name for visitor ID
    
    Returns:
        Dictionary with visitor details, or None if not found
    """
    with get_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        id_col = _quote_column(visitor_id_column)
        
        query = f'SELECT * FROM {table_name} WHERE {id_col} = %s'
        
        try:
            cursor.execute(query, (visitor_id,))
            result = cursor.fetchone()
            return dict(result) if result else None
        except psycopg2.Error as e:
            logger.error(f"Database error: {e}")
            raise


def update_visitor_features(
    visitor_id: str,
    features: np.ndarray,
    table_name: str = DEFAULT_TABLE_NAME,
    visitor_id_column: str = DEFAULT_ID_COLUMN,
    features_column: str = DEFAULT_FEATURES_COLUMN
) -> bool:
    """
    Update face features for a visitor in the database.
    
    Args:
        visitor_id: The visitor ID to update
        features: 128-dim feature vector as numpy array
        table_name: Name of the visitors table
        visitor_id_column: Column name for visitor ID
        features_column: Column name for face features
    
    Returns:
        True if successful, False otherwise
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        id_col = _quote_column(visitor_id_column)
        feat_col = _quote_column(features_column)
        
        # Convert numpy array to base64-encoded bytes
        features_bytes = features.astype(np.float32).tobytes()
        features_base64 = base64.b64encode(features_bytes).decode('utf-8')
        
        query = f"""
            UPDATE {table_name}
            SET {feat_col} = %s
            WHERE {id_col} = %s
        """
        
        try:
            cursor.execute(query, (features_base64, visitor_id))
            conn.commit()
            logger.info(f"Updated face features for visitor {visitor_id}")
            return True
        except psycopg2.Error as e:
            logger.error(f"Database error updating features: {e}")
            conn.rollback()
            return False
