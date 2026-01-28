"""
Database Module (Backward Compatibility Layer)

This module maintains backward compatibility with existing code that imports
`database` directly. New code should use explicit imports from:
- db.connection for connection management
- db.queries for query functions
- db.models for type definitions

Usage (legacy):
    import database
    database.get_connection()
    database.get_visitor_images_from_db()

Usage (recommended):
    from db.connection import get_connection
    from db.queries import get_visitor_images_from_db
"""

# Re-export everything from the new modular structure
from .connection import (
    get_db_connection,
    get_connection,
    init_connection_pool,
    test_connection,
    close_connection_pool,
)

from .queries import (
    get_visitor_images_from_db,
    get_visitors_with_features_only,
    get_visitor_details,
    update_visitor_features,
    _quote_column,
    DEFAULT_TABLE_NAME,
    DEFAULT_ID_COLUMN,
    DEFAULT_IMAGE_COLUMN,
    DEFAULT_FEATURES_COLUMN,
    CASE_SENSITIVE_COLUMNS,
)

# Export constants for backward compatibility
__all__ = [
    'get_db_connection',
    'get_connection',
    'init_connection_pool',
    'test_connection',
    'close_connection_pool',
    'get_visitor_images_from_db',
    'get_visitors_with_features_only',
    'get_visitor_details',
    'update_visitor_features',
    '_quote_column',
    'DEFAULT_TABLE_NAME',
    'DEFAULT_ID_COLUMN',
    'DEFAULT_IMAGE_COLUMN',
    'DEFAULT_FEATURES_COLUMN',
    'CASE_SENSITIVE_COLUMNS',
]
