"""
Database Connection Management

Handles PostgreSQL connection pooling and connection lifecycle.
Provides context managers for safe connection handling.
"""

import os
from typing import Optional
from pathlib import Path
from contextlib import contextmanager

import psycopg2
from psycopg2.pool import SimpleConnectionPool

from core.logger import get_logger

# Load environment variables
try:
    from dotenv import load_dotenv
    _SCRIPT_DIR = Path(__file__).parent.parent.parent
    env_file = _SCRIPT_DIR / ".env.test"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv(_SCRIPT_DIR / ".env")
except ImportError:
    pass

logger = get_logger(__name__)

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
