"""
Database Models and Type Definitions

Type definitions for database records. These are not ORM models but rather
TypedDict classes for type hints and documentation purposes.

If migrating to an ORM (e.g., SQLAlchemy) in the future, these can be
replaced with actual ORM model classes.
"""

from typing import TypedDict, Optional, List
from datetime import datetime


class VisitorBase(TypedDict, total=False):
    """Base visitor record structure - all fields optional."""
    id: str
    firstName: Optional[str]
    lastName: Optional[str]
    fullName: Optional[str]
    base64Image: Optional[str]
    imageUrl: Optional[str]
    faceFeatures: Optional[str]  # Base64-encoded feature vector
    createdAt: Optional[datetime]
    updatedAt: Optional[datetime]


class VisitorWithImage(TypedDict):
    """Visitor record with required image data."""
    id: str
    firstName: Optional[str]
    lastName: Optional[str]
    base64Image: str  # Required


class VisitorWithFeatures(TypedDict):
    """Visitor record with required face features."""
    id: str
    firstName: Optional[str]
    lastName: Optional[str]
    faceFeatures: str  # Required


class VisitorFull(TypedDict, total=False):
    """Complete visitor record with all fields."""
    id: str
    firstName: Optional[str]
    lastName: Optional[str]
    fullName: Optional[str]
    base64Image: str
    imageUrl: Optional[str]
    faceFeatures: Optional[str]
    createdAt: Optional[datetime]
    updatedAt: Optional[datetime]


# Type aliases for common use cases
VisitorDict = dict  # Generic visitor dictionary
VisitorList = List[VisitorDict]  # List of visitor dictionaries


def validate_visitor_id(visitor_id: str) -> bool:
    """
    Validate visitor ID format.
    
    Args:
        visitor_id: Visitor ID to validate
    
    Returns:
        True if valid, False otherwise
    """
    if not visitor_id or not isinstance(visitor_id, str):
        return False
    # Add any specific validation rules here
    return len(visitor_id.strip()) > 0


def validate_table_name(table_name: str) -> bool:
    """
    Validate table name format to prevent SQL injection.
    
    Args:
        table_name: Table name to validate
    
    Returns:
        True if valid, False otherwise
    """
    import re
    # Allow schema.table format with quoted identifiers
    pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*(\."[a-zA-Z_][a-zA-Z0-9_]*")?$'
    return bool(re.match(pattern, table_name))
