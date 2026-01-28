"""
Database Module

Provides database connection management and query functions.

Usage:
    from db import get_connection, get_visitor_images_from_db
    from db.connection import init_connection_pool, test_connection
    from db.queries import get_visitor_details, update_visitor_features
"""

# Connection management
from .connection import (
    get_db_connection,
    get_connection,
    init_connection_pool,
    test_connection,
    close_connection_pool,
)

# Query functions
from .queries import (
    get_visitor_images_from_db,
    get_visitors_with_features_only,
    get_visitor_details,
    update_visitor_features,
    _quote_column,  # Exposed for advanced use cases
)

# Models and types
from .models import (
    VisitorBase,
    VisitorWithImage,
    VisitorWithFeatures,
    VisitorFull,
    VisitorDict,
    VisitorList,
    validate_visitor_id,
    validate_table_name,
)

__all__ = [
    # Connection
    'get_db_connection',
    'get_connection',
    'init_connection_pool',
    'test_connection',
    'close_connection_pool',
    # Queries
    'get_visitor_images_from_db',
    'get_visitors_with_features_only',
    'get_visitor_details',
    'update_visitor_features',
    # Models
    'VisitorBase',
    'VisitorWithImage',
    'VisitorWithFeatures',
    'VisitorFull',
    'VisitorDict',
    'VisitorList',
    'validate_visitor_id',
    'validate_table_name',
]
