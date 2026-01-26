"""
Database connection and query utilities for visitor recognition.
Handles PostgreSQL database connections and visitor image queries.
"""

import os
import base64
from typing import List, Dict, Optional
from pathlib import Path
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
import logging
import numpy as np

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    _SCRIPT_DIR = Path(__file__).parent
    env_file = _SCRIPT_DIR.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv(_SCRIPT_DIR / ".env")
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Database configuration from environment variables
DATABASE_URL = os.environ.get("DATABASE_URL")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "visitors_db")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")

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

# Connection pool (optional, for better performance)
_connection_pool: Optional[SimpleConnectionPool] = None


def _quote_column(column: str) -> str:
    """Quote column name if it's case-sensitive."""
    return f'"{column}"' if column in CASE_SENSITIVE_COLUMNS else column


def get_db_connection():
    """
    Get a database connection.
    Uses DATABASE_URL if provided, otherwise uses individual parameters.
    
    Returns:
        psycopg2.connection: Database connection
    """
    if DATABASE_URL:
        return psycopg2.connect(DATABASE_URL)
    
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )


@contextmanager
def get_connection():
    """
    Context manager for database connections.
    Uses connection pool if available, otherwise creates a new connection.
    
    Yields:
        psycopg2.connection: Database connection
    """
    conn = None
    try:
        if _connection_pool:
            conn = _connection_pool.getconn()
        else:
            conn = get_db_connection()
        yield conn
    finally:
        if conn:
            if _connection_pool:
                _connection_pool.putconn(conn)
            else:
                conn.close()


def init_connection_pool(min_conn: int = 1, max_conn: int = 10) -> None:
    """
    Initialize a connection pool for better performance.
    
    Args:
        min_conn: Minimum number of connections
        max_conn: Maximum number of connections
    """
    global _connection_pool
    
    if DATABASE_URL:
        _connection_pool = SimpleConnectionPool(min_conn, max_conn, dsn=DATABASE_URL)
    else:
        _connection_pool = SimpleConnectionPool(
            min_conn, max_conn,
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
    logger.info(f"Database connection pool initialized ({min_conn}-{max_conn} connections)")


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


def test_connection() -> bool:
    """
    Test database connection.
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
        logger.info("Database connection test successful")
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


def close_connection_pool() -> None:
    """Close all connections in the pool."""
    global _connection_pool
    if _connection_pool:
        _connection_pool.closeall()
        _connection_pool = None
        logger.info("Database connection pool closed")
