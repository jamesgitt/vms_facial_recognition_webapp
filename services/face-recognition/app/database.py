"""
Database connection and query utilities for visitor recognition.
Handles PostgreSQL database connections and visitor image queries.
"""

import os
import base64
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
import logging
import numpy as np

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    # Try to load .env from parent directory (services/face-recognition/.env)
    _SCRIPT_DIR = Path(__file__).parent
    env_file = _SCRIPT_DIR.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        # Also try current directory
        load_dotenv(_SCRIPT_DIR / ".env")
except ImportError:
    # python-dotenv not installed, skip .env loading
    pass

logger = logging.getLogger(__name__)

# Database configuration from environment variables
DATABASE_URL = os.environ.get("DATABASE_URL")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "visitors_db")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")

# Connection pool (optional, for better performance)
_connection_pool: Optional[SimpleConnectionPool] = None


def get_db_connection():
    """
    Get a database connection.
    Uses connection pool if available, otherwise creates a new connection.
    
    Returns:
        psycopg2.connection: Database connection
    """
    global _connection_pool
    
    # If DATABASE_URL is provided, use it directly
    if DATABASE_URL:
        return psycopg2.connect(DATABASE_URL)
    
    # Otherwise, use individual connection parameters
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )


def init_connection_pool(min_conn=1, max_conn=10):
    """
    Initialize a connection pool for better performance.
    
    Args:
        min_conn: Minimum number of connections
        max_conn: Maximum number of connections
    """
    global _connection_pool
    
    if DATABASE_URL:
        _connection_pool = SimpleConnectionPool(
            min_conn, max_conn, dsn=DATABASE_URL
        )
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
    table_name: str = 'public."Visitor"',
    visitor_id_column: str = "id",
    image_column: str = "base64Image",
    features_column: Optional[str] = None,
    limit: Optional[int] = None,
    active_only: bool = False
) -> List[Dict]:
    """
    Query visitor images and features from the database.
    
    Args:
        table_name: Name of the visitors table (default: 'public."Visitor"')
        visitor_id_column: Column name for visitor ID (default: "visitor_id")
        image_column: Column name for base64 image (default: "base64Image")
        features_column: Column name for face features (default: None, will not fetch if not provided)
        limit: Maximum number of visitors to retrieve (None = all)
        active_only: If True, only get active visitors (requires 'active' or 'status' column)
    
    Returns:
        List of dictionaries with visitor_id, base64Image, firstName, lastName, and optionally faceFeatures
        Example: [{"id": "123", "base64Image": "base64_string", "firstName": "John", "lastName": "Doe", "faceFeatures": "base64_feature", ...}, ...]
    """
    conn = None
    try:
        if _connection_pool:
            conn = _connection_pool.getconn()
        else:
            conn = get_db_connection()
        
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Build query - use quoted column names for case-sensitive columns
        # PostgreSQL requires quotes for mixed-case column names
        case_sensitive_cols = ['id', 'base64Image', 'imageUrl', 'firstName', 'lastName', 'fullName', 'createdAt', 'updatedAt', 'faceFeatures']
        visitor_id_col = f'"{visitor_id_column}"' if visitor_id_column in case_sensitive_cols else visitor_id_column
        image_col = f'"{image_column}"' if image_column in case_sensitive_cols else image_column
        
        # Build SELECT clause
        select_cols = [f"{visitor_id_col}", f"{image_col}"]
        if features_column:
            features_col = f'"{features_column}"' if features_column in case_sensitive_cols else features_column
            select_cols.append(f"{features_col}")
        
        # Always include firstName and lastName if they exist in the table
        firstName_col = '"firstName"' if 'firstName' in case_sensitive_cols else 'firstName'
        lastName_col = '"lastName"' if 'lastName' in case_sensitive_cols else 'lastName'
        select_cols.extend([firstName_col, lastName_col])
        
        query = f"""
            SELECT {', '.join(select_cols)}
            FROM {table_name}
            WHERE {image_col} IS NOT NULL
        """
        
        # Note: active_only removed as it's not in the minimal schema
        # Add back if you add active/status columns later
        
        query += f" ORDER BY {visitor_id_col}"
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        # Convert to list of dicts
        visitors = [dict(row) for row in results]
        
        logger.info(f"Retrieved {len(visitors)} visitors from database")
        return visitors
        
    except psycopg2.Error as e:
        logger.error(f"Database error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error querying database: {e}")
        raise
    finally:
        if conn:
            if _connection_pool:
                _connection_pool.putconn(conn)
            else:
                conn.close()


def get_visitor_details(
    visitor_id: str,
    table_name: str = 'public."Visitor"',
    visitor_id_column: str = "id"
) -> Optional[Dict]:
    """
    Get full visitor details by visitor_id.
    
    Args:
        visitor_id: The visitor ID to look up
        table_name: Name of the visitors table
        visitor_id_column: Column name for visitor ID
    
    Returns:
        Dictionary with visitor details, or None if not found
    """
    conn = None
    try:
        if _connection_pool:
            conn = _connection_pool.getconn()
        else:
            conn = get_db_connection()
        
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Use quoted column name if it's case-sensitive
        visitor_id_col = f'"{visitor_id_column}"' if visitor_id_column in ['id', 'base64Image', 'imageUrl', 'firstName', 'lastName', 'fullName', 'createdAt', 'updatedAt'] else visitor_id_column
        query = f'SELECT * FROM {table_name} WHERE {visitor_id_col} = %s'
        cursor.execute(query, (visitor_id,))
        result = cursor.fetchone()
        
        if result:
            return dict(result)
        return None
        
    except psycopg2.Error as e:
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            if _connection_pool:
                _connection_pool.putconn(conn)
            else:
                conn.close()


def update_visitor_features(
    visitor_id: str,
    features: np.ndarray,
    table_name: str = 'public."Visitor"',
    visitor_id_column: str = "id",
    features_column: str = "faceFeatures"
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
    conn = None
    try:
        if _connection_pool:
            conn = _connection_pool.getconn()
        else:
            conn = get_db_connection()
        
        cursor = conn.cursor()
        
        # Use quoted column names if case-sensitive
        case_sensitive_cols = ['id', 'base64Image', 'imageUrl', 'firstName', 'lastName', 'fullName', 'createdAt', 'updatedAt', 'faceFeatures']
        visitor_id_col = f'"{visitor_id_column}"' if visitor_id_column in case_sensitive_cols else visitor_id_column
        features_col = f'"{features_column}"' if features_column in case_sensitive_cols else features_column
        
        # Convert numpy array to base64-encoded bytes
        features_bytes = features.astype(np.float32).tobytes()
        features_base64 = base64.b64encode(features_bytes).decode('utf-8')
        
        # Update query
        query = f"""
            UPDATE {table_name}
            SET {features_col} = %s
            WHERE {visitor_id_col} = %s
        """
        
        cursor.execute(query, (features_base64, visitor_id))
        conn.commit()
        
        logger.info(f"Updated face features for visitor {visitor_id}")
        return True
        
    except psycopg2.Error as e:
        logger.error(f"Database error updating features: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            if _connection_pool:
                _connection_pool.putconn(conn)
            else:
                conn.close()

def test_connection() -> bool:
    """
    Test database connection.
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        logger.info("Database connection test successful")
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


def close_connection_pool():
    """Close all connections in the pool."""
    global _connection_pool
    if _connection_pool:
        _connection_pool.closeall()
        _connection_pool = None
        logger.info("Database connection pool closed")
