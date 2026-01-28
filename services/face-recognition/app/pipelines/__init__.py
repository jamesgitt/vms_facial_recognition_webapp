"""
Pipelines Module

Business logic for face detection, recognition, comparison, and feature extraction.
Separates core ML logic from API route handling.
"""

from .detection import detect_faces_in_image, DetectionResult
from .feature_extraction import (
    extract_features_from_image,
    extract_feature_from_visitor_data,
    decode_feature_from_base64,
)
from .comparison import compare_two_faces, ComparisonResult
from .recognition import recognize_visitor, RecognitionResult
from .visitor_loader import (
    load_visitors_from_database,
    load_visitors_from_test_images,
    VisitorData,
)

__all__ = [
    # Detection
    "detect_faces_in_image",
    "DetectionResult",
    # Feature extraction
    "extract_features_from_image",
    "extract_feature_from_visitor_data",
    "decode_feature_from_base64",
    # Comparison
    "compare_two_faces",
    "ComparisonResult",
    # Recognition
    "recognize_visitor",
    "RecognitionResult",
    # Visitor loading
    "load_visitors_from_database",
    "load_visitors_from_test_images",
    "VisitorData",
]
