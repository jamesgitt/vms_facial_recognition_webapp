"""
Image Loader Module

Centralized image loading and processing for face recognition API endpoints.
Provides a unified interface for loading images from various sources:
- File uploads (UploadFile)
- Base64 encoded strings
- Data URLs (data:image/...)
- File paths
- URLs (HTTP/HTTPS)
- Database references
- In-memory cache
"""

import os
import io
import base64
from pathlib import Path
from typing import Union, Optional, Tuple, List, Dict

import cv2
import numpy as np
from PIL import Image
from fastapi import UploadFile

# Optional imports
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from db import database
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

# Configuration
ALLOWED_FORMATS = frozenset({"jpg", "jpeg", "png", "webp", "bmp"})
DEFAULT_MAX_IMAGE_SIZE = (1920, 1920)
DEFAULT_TABLE_NAME = 'public."Visitor"'

# In-memory image cache
_image_cache: Dict[str, np.ndarray] = {}


class ImageLoadError(Exception):
    """Custom exception for image loading errors."""
    def __init__(self, error_type: str, source: str, message: str):
        self.error_type = error_type
        self.source = source
        self.message = message
        super().__init__(f"Image loading error: {error_type} from {source}: {message}")


class ImageValidationError(Exception):
    """Custom exception for validation failures."""
    def __init__(self, validation_rule: str, actual_value: str):
        self.validation_rule = validation_rule
        self.actual_value = actual_value
        super().__init__(f"Validation failed: {validation_rule} - actual: {actual_value}")


def _pil_to_cv2(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV BGR numpy array."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _strip_data_url(image_b64: str) -> str:
    """Strip data URL prefix if present."""
    if image_b64.startswith('data:') and ',' in image_b64:
        return image_b64.split(',', 1)[1]
    return image_b64


# Core Loading Functions

def load_from_upload(upload_file: UploadFile) -> np.ndarray:
    """Load image from FastAPI UploadFile."""
    try:
        contents = upload_file.file.read()
        img = Image.open(io.BytesIO(contents))
        return _pil_to_cv2(img)
    except Exception:
        raise ValueError("Invalid uploaded image file.")


def load_from_base64(image_b64: str) -> np.ndarray:
    """Load image from base64 string (with or without data URL prefix)."""
    try:
        image_b64 = _strip_data_url(image_b64)
        img_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_bytes))
        return _pil_to_cv2(img)
    except Exception:
        raise ValueError("Invalid base64 image.")


def load_from_path(file_path: str) -> np.ndarray:
    """Load image from local filesystem path."""
    try:
        img = Image.open(file_path)
        return _pil_to_cv2(img)
    except Exception:
        raise ValueError("Invalid image file path.")


def load_from_url(url: str) -> np.ndarray:
    """Load image from HTTP/HTTPS URL."""
    if not REQUESTS_AVAILABLE:
        raise ValueError("URL loading requires 'requests' library.")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))
        return _pil_to_cv2(img)
    except Exception:
        raise ValueError("Invalid image URL.")


def load_from_database(
    visitor_id: str,
    table_name: str = DEFAULT_TABLE_NAME,
    image_column: str = "base64Image"
) -> np.ndarray:
    """Load image from database by visitor_id."""
    if not DB_AVAILABLE:
        raise ValueError("Database module not available.")
    
    try:
        visitors = database.get_visitor_images_from_db(
            table_name=table_name,
            visitor_id_column="id",
            image_column=image_column,
            limit=1
        )
        
        for visitor_data in visitors:
            if str(visitor_data.get("id")) == str(visitor_id):
                base64_image = visitor_data.get(image_column)
                if base64_image:
                    return load_from_base64(base64_image)
        
        raise ValueError(f"Visitor {visitor_id} not found in database.")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Error loading from database: {e}")


def load_from_reference(ref_id: str, ref_type: str) -> np.ndarray:
    """Load image from reference capture point."""
    loaders = {
        "database": lambda: load_from_database(ref_id),
        "file": lambda: load_from_path(ref_id),
        "cache": lambda: _load_from_cache(ref_id),
        "url": lambda: load_from_url(ref_id),
    }
    
    if ref_type not in loaders:
        raise ValueError(f"Invalid reference type: {ref_type}")
    
    return loaders[ref_type]()


def _load_from_cache(ref_id: str) -> np.ndarray:
    """Load image from cache."""
    cached = _image_cache.get(ref_id)
    if cached is None:
        raise ValueError(f"Image not found in cache: {ref_id}")
    return cached


# Unified Loader

def load_image(source: Union[UploadFile, str], source_type: Optional[str] = None) -> np.ndarray:
    """
    Unified image loader - main entry point for all image loading.
    
    Args:
        source: Image source (UploadFile, base64 string, file path, URL, or reference ID)
        source_type: Optional type hint ('base64', 'file', 'url', 'database', 'cache', 'reference')
    
    Returns:
        OpenCV BGR format image array
    """
    if isinstance(source, UploadFile):
        return load_from_upload(source)
    
    if not isinstance(source, str):
        raise ValueError(f"Unsupported source type: {type(source)}")
    
    # Use explicit source_type if provided
    if source_type:
        return _load_by_type(source, source_type)
    
    # Auto-detect source type
    return _auto_detect_and_load(source)


def _load_by_type(source: str, source_type: str) -> np.ndarray:
    """Load image by explicit source type."""
    if source_type == "base64":
        return load_from_base64(source)
    elif source_type == "file":
        return load_from_path(source)
    elif source_type == "url":
        return load_from_url(source)
    elif source_type == "database":
        return load_from_database(source)
    elif source_type == "cache":
        return _load_from_cache(source)
    elif source_type == "reference":
        if ":" in source:
            ref_id, ref_type = source.split(":", 1)
            return load_from_reference(ref_id, ref_type)
        return load_from_reference(source, "cache")
    else:
        raise ValueError(f"Unknown source_type: {source_type}")


def _auto_detect_and_load(source: str) -> np.ndarray:
    """Auto-detect source type and load image."""
    if source.startswith("data:"):
        return load_from_base64(source)
    
    if source.startswith(("http://", "https://")):
        return load_from_url(source)
    
    if os.path.exists(source):
        return load_from_path(source)
    
    # Try as base64 string
    try:
        return load_from_base64(source)
    except ValueError:
        raise ValueError(f"Could not determine image source type: {source[:50]}...")


# Validation Functions

def validate_image_format(img: Image.Image) -> bool:
    """Check if image format is allowed."""
    fmt = (img.format or "").lower()
    if fmt not in ALLOWED_FORMATS:
        raise ValueError(f"Format '{fmt}' not allowed. Allowed: {ALLOWED_FORMATS}")
    return True


def validate_image_size(size: Tuple[int, int], max_size: Tuple[int, int]) -> bool:
    """Check if image dimensions are within limits."""
    w, h = size
    max_w, max_h = max_size
    if w > max_w or h > max_h:
        raise ValueError(f"Dimensions {w}x{h} exceed max {max_w}x{max_h}")
    return True


def validate_image_data(img_array: np.ndarray) -> bool:
    """Validate image array structure."""
    if img_array.size == 0:
        raise ValueError("Image array is empty")
    if img_array.ndim != 3:
        raise ValueError("Image must have 3 dimensions (H, W, C)")
    if img_array.shape[2] != 3:
        raise ValueError("Image must have 3 color channels (BGR)")
    return True


# Image Processing Utilities

def normalize_image(img_array: np.ndarray) -> np.ndarray:
    """Convert BGR to RGB."""
    return cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)


def resize_if_needed(img_array: np.ndarray, max_size: Tuple[int, int]) -> np.ndarray:
    """Resize image if it exceeds max dimensions, maintaining aspect ratio."""
    h, w = img_array.shape[:2]
    max_w, max_h = max_size
    
    if w <= max_w and h <= max_h:
        return img_array
    
    scale = min(max_w / w, max_h / h)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(img_array, new_size, interpolation=cv2.INTER_AREA)


# Cache Functions

def cache_image(key: str, image_array: np.ndarray) -> None:
    """Store image in cache."""
    _image_cache[key] = image_array


def get_cached_image(key: str) -> Optional[np.ndarray]:
    """Retrieve image from cache."""
    return _image_cache.get(key)


def clear_cache(key: Optional[str] = None) -> bool:
    """Clear specific key or entire cache."""
    if key:
        if key in _image_cache:
            del _image_cache[key]
            return True
        return False
    _image_cache.clear()
    return True


def list_cached_keys() -> List[str]:
    """List all cached image keys."""
    return list(_image_cache.keys())


# Capture Point System

def _convert_to_array(image_data: Union[np.ndarray, str, bytes]) -> np.ndarray:
    """Convert various image data types to numpy array."""
    if isinstance(image_data, np.ndarray):
        return image_data
    if isinstance(image_data, str):
        return load_from_base64(image_data)
    if isinstance(image_data, bytes):
        img = Image.open(io.BytesIO(image_data))
        return _pil_to_cv2(img)
    raise ValueError(f"Unsupported image_data type: {type(image_data)}")


def save_image_to_file(file_path: str, image_data: Union[np.ndarray, str, bytes]) -> None:
    """Save image data to file."""
    try:
        img_array = _convert_to_array(image_data)
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        cv2.imwrite(file_path, img_array)
    except Exception as e:
        raise ValueError(f"Error saving image: {e}")


def register_capture_point(
    ref_id: str,
    image_data: Union[np.ndarray, str, bytes],
    ref_type: str = 'cache'
) -> bool:
    """Register an image with a reference ID for reuse."""
    if ref_type == 'cache':
        cache_image(ref_id, _convert_to_array(image_data))
        return True
    elif ref_type == 'file':
        save_image_to_file(ref_id, image_data)
        return True
    elif ref_type == 'database':
        raise NotImplementedError("Database storage not implemented")
    else:
        raise ValueError(f"Invalid reference type: {ref_type}")


def get_capture_point(ref_id: str) -> Optional[np.ndarray]:
    """Retrieve image by reference ID from cache."""
    return _image_cache.get(ref_id)


def clear_capture_point(ref_id: str) -> bool:
    """Remove reference from cache."""
    return clear_cache(ref_id)


def list_capture_points() -> List[str]:
    """List all registered reference IDs."""
    return list_cached_keys()


# Configuration

def get_config() -> dict:
    """Get image loader configuration."""
    return {
        "max_image_size": DEFAULT_MAX_IMAGE_SIZE,
        "allowed_formats": list(ALLOWED_FORMATS),
        "requests_available": REQUESTS_AVAILABLE,
        "database_available": DB_AVAILABLE,
        "cache_size": len(_image_cache),
    }
