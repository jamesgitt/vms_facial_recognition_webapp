"""
Image Loader Module
Centralized image loading and processing for face recognition API endpoints.

This module provides a unified interface for loading images from various sources:
- File uploads (UploadFile)
- Base64 encoded strings
- Data URLs (data:image/...)
- Reference capture points (stored image IDs, file paths, database references)
- URL references (HTTP/HTTPS)

All endpoints should use this module for consistent image handling.
"""
from typing import Union, Optional, Tuple, List, Dict
import numpy as np
from fastapi import UploadFile
from PIL import Image
import cv2
import base64
import io
import os
from pathlib import Path

# Optional imports for advanced features
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import database
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

# 1. CORE IMAGE LOADING FUNCTIONS
#    [ ] load_from_upload(upload_file: UploadFile) -> np.ndarray
#        - Convert FastAPI UploadFile to OpenCV BGR numpy array
#        - Handle file reading, PIL conversion, RGB->BGR
#        - Error handling for invalid files
def load_from_upload(upload_file: UploadFile) -> np.ndarray:
    try:
        contents = upload_file.file.read()
        img = Image.open(io.BytesIO(contents))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_np = np.array(img)
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    except Exception:
        raise ValueError("Invalid uploaded image file.")
        
#    [ ] load_from_base64(image_b64: str) -> np.ndarray
#        - Decode base64 string to image
#        - Handle data URL format (data:image/jpeg;base64,...)
#        - Strip data URL prefix if present
#        - Convert to OpenCV BGR format
def load_from_base64(image_b64: str) -> np.ndarray:
    try:
        # Handle data URL format (data:image/jpeg;base64,...)
        if image_b64.startswith('data:'):
            # Extract base64 part after comma
            image_b64 = image_b64.split(',', 1)[1] if ',' in image_b64 else image_b64
        
        img_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_np = np.array(img)
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    except Exception:
        raise ValueError("Invalid base64 image.")


#    [ ] load_from_path(file_path: str) -> np.ndarray
#        - Load image from local filesystem path
#        - Support absolute and relative paths
#        - Handle path validation and file existence
def load_from_path(file_path: str) -> np.ndarray:
    try:
        img = Image.open(file_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_np = np.array(img)
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    except Exception:
        raise ValueError("Invalid image file path.")

#    [ ] load_from_reference(ref_id: str, ref_type: str) -> np.ndarray
#        - Load image from reference capture point
#        - Support reference types:
#          * 'database' - Load from database by visitor_id
#          * 'file' - Load from file path reference
#          * 'cache' - Load from in-memory cache
#          * 'url' - Load from HTTP/HTTPS URL
#        - Return cached image if available
def load_from_reference(ref_id: str, ref_type: str) -> np.ndarray:
    try:
        if ref_type == "database":
            return load_from_database(ref_id)
        elif ref_type == "file":
            return load_from_path(ref_id)
        elif ref_type == "cache":
            cached = get_cached_image(ref_id)
            if cached is None:
                raise ValueError(f"Image not found in cache: {ref_id}")
            return cached
        elif ref_type == "url":
            return load_from_url(ref_id)
        else:
            raise ValueError(f"Invalid reference type: {ref_type}")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Invalid reference image: {str(e)}")
        
#    [ ] load_from_url(url: str) -> np.ndarray
#        - Download image from HTTP/HTTPS URL
#        - Handle redirects and timeouts
#        - Validate content type
#        - Convert to OpenCV BGR format
def load_from_url(url: str) -> np.ndarray:
    if not REQUESTS_AVAILABLE:
        raise ValueError("URL loading requires 'requests' library. Install with: pip install requests")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_np = np.array(img)
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    except Exception:
        raise ValueError("Invalid image URL.")

# 6. CACHING SYSTEM - Simple in-memory cache
_image_cache: Dict[str, np.ndarray] = {}

def cache_image(key: str, image_array: np.ndarray) -> None:
    """Store image in cache with key."""
    _image_cache[key] = image_array

def get_cached_image(key: str) -> Optional[np.ndarray]:
    """Retrieve from cache if available."""
    return _image_cache.get(key)

# 7. DATABASE INTEGRATION
def load_from_database(visitor_id: str, table_name: str = "visitors", 
                      image_column: str = "base64Image") -> np.ndarray:
    """
    Load image from database by visitor_id.
    Query database for visitor image, decode base64, convert to OpenCV BGR format.
    """
    if not DB_AVAILABLE:
        raise ValueError("Database module not available. Cannot load from database.")
    
    try:
        # Get visitor image from database
        visitors = database.get_visitor_images_from_db(
            table_name=table_name,
            visitor_id_column="id",
            image_column=image_column,
            limit=1
        )
        
        # Find matching visitor
        for visitor_data in visitors:
            if str(visitor_data.get("id")) == str(visitor_id):
                base64_image = visitor_data.get(image_column)
                if base64_image:
                    # Decode base64 image
                    return load_from_base64(base64_image)
        
        raise ValueError(f"Visitor {visitor_id} not found in database or has no image.")
    except Exception as e:
        raise ValueError(f"Error loading image from database: {str(e)}")

# 2. UNIFIED LOADER FUNCTION
#    [ ] load_image(source: Union[UploadFile, str], source_type: Optional[str] = None) -> np.ndarray
#        - Main entry point for all image loading
#        - Auto-detect source type if not provided
#        - Support multiple input formats:
#          * UploadFile object
#          * Base64 string
#          * Data URL string
#          * File path string
#          * Reference ID string (with source_type)
#          * URL string
#        - Return standardized OpenCV BGR numpy array
#        - Raise ImageLoadError with descriptive messages
def load_image(source: Union[UploadFile, str], source_type: Optional[str] = None) -> np.ndarray:
    """
    Unified image loader - main entry point for all image loading.
    
    Args:
        source: Image source (UploadFile, base64 string, file path, URL, or reference ID)
        source_type: Optional type hint ('base64', 'file', 'url', 'database', 'cache', 'reference')
    
    Returns:
        np.ndarray: OpenCV BGR format image array
    """
    try:
        if isinstance(source, UploadFile):
            return load_from_upload(source)
        elif isinstance(source, str):
            # If source_type is provided, use it
            if source_type:
                if source_type == "base64":
                    return load_from_base64(source)
                elif source_type == "file":
                    return load_from_path(source)
                elif source_type == "url":
                    return load_from_url(source)
                elif source_type == "database":
                    return load_from_database(source)
                elif source_type == "cache":
                    cached = get_cached_image(source)
                    if cached is not None:
                        return cached
                    raise ValueError(f"Image not found in cache: {source}")
                elif source_type == "reference":
                    # Reference format: "ref_id:ref_type" or just ref_id (defaults to cache)
                    if ":" in source:
                        ref_id, ref_type = source.split(":", 1)
                        return load_from_reference(ref_id, ref_type)
                    else:
                        return load_from_reference(source, "cache")
                else:
                    raise ValueError(f"Unknown source_type: {source_type}")
            
            # Auto-detect source type
            if source.startswith("data:"):
                return load_from_base64(source)
            elif source.startswith("http://") or source.startswith("https://"):
                return load_from_url(source)
            elif os.path.exists(source) or Path(source).exists():
                return load_from_path(source)
            else:
                # Try as base64 string
                try:
                    return load_from_base64(source)
                except ValueError:
                    raise ValueError(f"Could not determine image source type for: {source[:50]}...")
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Error loading image: {str(e)}")
                
# 3. IMAGE VALIDATION
#    [ ] validate_image_format(img: Image.Image) -> bool
#        - Check if format is in ALLOWED_FORMATS
#        - Raise ValueError with allowed formats list
#    
#    [ ] validate_image_size(size: Tuple[int, int], max_size: Tuple[int, int]) -> bool
#        - Check dimensions against MAX_IMAGE_SIZE
#        - Raise ValueError if too large
#    
#    [ ] validate_image_data(img_array: np.ndarray) -> bool
#        - Check if array is valid (not empty, correct shape, valid dtype)
#        - Validate color channels (BGR format expected)
#
# 4. IMAGE PROCESSING UTILITIES
#    [ ] normalize_image(img_array: np.ndarray) -> np.ndarray
#        - Ensure RGB->BGR conversion
#        - Handle grayscale conversion if needed
#        - Normalize pixel values if required
#    
#    [ ] resize_if_needed(img_array: np.ndarray, max_size: Tuple[int, int]) -> np.ndarray
#        - Resize image if exceeds max dimensions
#        - Maintain aspect ratio
#        - Use high-quality interpolation
#
# 5. REFERENCE CAPTURE POINT SYSTEM
#    [ ] register_capture_point(ref_id: str, image_data: Union[np.ndarray, str, bytes], 
#                               ref_type: str = 'cache') -> bool
#        - Register an image with a reference ID for reuse
#        - Support storage types: 'cache' (memory), 'file' (filesystem), 'database'
#        - Return success status
#    
#    [ ] get_capture_point(ref_id: str) -> Optional[np.ndarray]
#        - Retrieve image by reference ID
#        - Check cache first, then filesystem, then database
#        - Return None if not found
#    
#    [ ] clear_capture_point(ref_id: str) -> bool
#        - Remove reference from cache/filesystem
#        - Return success status
#    
#    [ ] list_capture_points() -> List[str]
#        - Return list of all registered reference IDs
#        - Useful for debugging and management
#
# 6. CACHING SYSTEM
#    [ ] ImageCache class
#        - In-memory LRU cache for frequently accessed images
#        - Configurable cache size (max items, max memory)
#        - Cache key generation (hash of image data or reference ID)
#        - Cache invalidation methods
#    
#    [ ] cache_image(key: str, image_array: np.ndarray) -> None
#        - Store image in cache with key
#    
#    [ ] get_cached_image(key: str) -> Optional[np.ndarray]
#        - Retrieve from cache if available
#
# 7. DATABASE INTEGRATION
#    [ ] load_from_database(visitor_id: str, table_name: str, 
#                          image_column: str = 'base64Image') -> np.ndarray
#        - Query database for visitor image
#        - Decode base64 image from database
#        - Convert to OpenCV BGR format
#        - Cache result for future use
#        - Handle database errors gracefully
#
# 8. ERROR HANDLING
#    [ ] ImageLoadError(Exception)
#        - Custom exception for image loading errors
#        - Include error type, source, and message
#    
#    [ ] ImageValidationError(Exception)
#        - Custom exception for validation failures
#        - Include validation rule and actual value
#
# 9. CONFIGURATION
#    [ ] Load configuration from environment variables
#        - MAX_IMAGE_SIZE (default: 1920x1920)
#        - ALLOWED_FORMATS (default: jpg, jpeg, png)
#        - CACHE_SIZE (default: 100 items)
#        - CACHE_TTL (default: 3600 seconds)
#        - ENABLE_URL_LOADING (default: False for security)
#        - MAX_URL_SIZE (default: 10MB)
#
# 10. TYPE HINTS AND DOCUMENTATION
#     [ ] Add comprehensive type hints for all functions
#     [ ] Add docstrings with examples for each function
#     [ ] Document error conditions and exceptions
#     [ ] Add usage examples in module docstring
#
# 11. TESTING CONSIDERATIONS
#     [ ] Unit tests for each loading method
#     [ ] Test error handling paths
#     [ ] Test caching behavior
#     [ ] Test reference capture point system
#     [ ] Test database integration
#     [ ] Test URL loading (with mock server)
#
# 12. MIGRATION FROM CURRENT CODE
#     [ ] Replace decode_base64_image() calls with load_image()
#     [ ] Replace uploadfile_to_np() calls with load_image()
#     [ ] Update all endpoints to use unified loader
#     [ ] Remove duplicate image loading code from face_recog_api.py
#     [ ] Update WebSocket handler to use new loader
#
# 13. PERFORMANCE OPTIMIZATIONS
#     [ ] Add image preprocessing caching
#     [ ] Optimize base64 decoding
#     [ ] Add async support for URL loading
#     [ ] Implement lazy loading for large images
#     [ ] Add image compression for storage
#
# 14. SECURITY ENHANCEMENTS
#     [ ] Validate file extensions and MIME types
#     [ ] Scan for malicious image content
#     [ ] Limit URL loading to whitelisted domains (if enabled)
#     [ ] Sanitize file paths to prevent directory traversal
#     [ ] Rate limiting for image loading operations
#
# ===================================
# IMPLEMENTATION PRIORITY:
# 1. Core loading functions (load_from_upload, load_from_base64)
# 2. Unified loader function (load_image)
# 3. Image validation functions
# 4. Reference capture point system
# 5. Caching system
# 6. Database integration
# 7. Error handling and configuration
# 8. Migration from existing code
# 9. Performance and security enhancements
#
# ===================================


