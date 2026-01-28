"""
Face Detection and Recognition Inference Utilities

Loads and configures YuNet (face detection) and SFace (recognition) ONNX models.
Designed as a reusable utility for API or web service integration.

Exports:
- detect_faces: Face detection with YuNet
- extract_face_features: Feature extraction with SFace (128-dim)
- compare_face_features: Cosine similarity comparison
- draw_face_rectangles: Visualization utility
- get_face_landmarks: Extract 5-point landmarks from detection
"""

import os
from typing import Optional, List, Tuple, Union

import cv2
import numpy as np

from core.logger import get_logger
logger = get_logger(__name__)


# Configuration
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

YUNET_FILENAME = 'face_detection_yunet_2023mar.onnx'
SFACE_FILENAME = 'face_recognition_sface_2021dec.onnx'

# Model parameters
YUNET_INPUT_SIZE = (640, 640)
YUNET_SCORE_THRESHOLD = 0.7
YUNET_NMS_THRESHOLD = 0.3
YUNET_TOP_K = 5000
SFACE_SIMILARITY_THRESHOLD = 0.55


def _find_models_dir() -> str:
    """Find the models directory from environment or common locations."""
    env_path = os.environ.get("MODELS_PATH")
    if env_path:
        return env_path
    
    search_paths = [
        os.path.join(_SCRIPT_DIR, 'models'),
        os.path.join(os.path.dirname(_SCRIPT_DIR), 'models'),
        os.path.join(os.path.dirname(os.path.dirname(_SCRIPT_DIR)), 'models'),
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            return path
    
    return os.path.join(_SCRIPT_DIR, 'models')


DEFAULT_MODELS_DIR = _find_models_dir()
YUNET_PATH = os.path.join(DEFAULT_MODELS_DIR, YUNET_FILENAME)
SFACE_PATH = os.path.join(DEFAULT_MODELS_DIR, SFACE_FILENAME)


def _verify_model_file(path: str, model_name: str) -> None:
    """Raise FileNotFoundError if model file is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{model_name} model not found at {path}")


def _load_models():
    """Load and return detector and recognizer models."""
    _verify_model_file(YUNET_PATH, "YuNet")
    _verify_model_file(SFACE_PATH, "SFace")
    
    try:
        det = cv2.FaceDetectorYN.create(
            model=YUNET_PATH,
            config='',
            input_size=YUNET_INPUT_SIZE,
            score_threshold=YUNET_SCORE_THRESHOLD,
            nms_threshold=YUNET_NMS_THRESHOLD,
            top_k=YUNET_TOP_K
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize YuNet: {e}")
    
    try:
        rec = cv2.FaceRecognizerSF.create(
            model=SFACE_PATH,
            config='',
            backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize SFace: {e}")
    
    return det, rec


# Initialize models at module load
detector, recognizer = _load_models()


def detect_faces(
    frame: np.ndarray,
    input_size: Tuple[int, int] = YUNET_INPUT_SIZE,
    score_threshold: float = YUNET_SCORE_THRESHOLD,
    nms_threshold: float = YUNET_NMS_THRESHOLD,
    return_landmarks: bool = False
) -> Optional[Union[np.ndarray, List[Tuple[int, int, int, int]]]]:
    """
    Detect faces in a BGR image using YuNet.

    Args:
        frame: Input BGR image
        input_size: Input size for detector
        score_threshold: Detection confidence threshold
        nms_threshold: Non-max suppression threshold
        return_landmarks: If True, return full face data with landmarks

    Returns:
        If return_landmarks=True: np.ndarray shape [num_faces, 15]
        If return_landmarks=False: List of (x, y, w, h) tuples
        None if no faces detected
    """
    if frame is None or not hasattr(frame, 'shape'):
        raise ValueError("Frame is None or invalid")
    
    resized = cv2.resize(frame, input_size)
    detector.setInputSize(input_size)

    try:
        _, faces = detector.detect(resized)
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return None

    if faces is None or len(faces) == 0:
        return None

    # Rescale to original frame size
    sx = frame.shape[1] / input_size[0]
    sy = frame.shape[0] / input_size[1]
    
    faces_rescaled = faces.astype(np.float32).copy()
    faces_rescaled[:, [0, 2, 5, 7, 9, 11, 13]] *= sx  # x coords
    faces_rescaled[:, [1, 3, 6, 8, 10, 12, 14]] *= sy  # y coords

    if return_landmarks:
        return faces_rescaled
    
    bboxes = faces_rescaled[:, :4].astype(int)
    return [(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])) for bbox in bboxes]


def extract_face_features(frame: np.ndarray, face_row: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract 128-dim SFace feature vector from a detected face.

    Args:
        frame: Full BGR image
        face_row: Face detection row from YuNet [x, y, w, h, score, ...landmarks]

    Returns:
        128-dim feature vector (float32) or None on failure
    """
    if frame is None or face_row is None:
        raise ValueError("Input frame or face_row is missing")
    
    try:
        aligned = recognizer.alignCrop(frame, face_row)
        return recognizer.feature(aligned)
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        return None


def compare_face_features(
    feature1: np.ndarray,
    feature2: np.ndarray,
    threshold: float = SFACE_SIMILARITY_THRESHOLD
) -> Tuple[float, bool]:
    """
    Compare two face features using cosine similarity.

    Args:
        feature1: First 128-dim feature vector
        feature2: Second 128-dim feature vector
        threshold: Similarity threshold for match

    Returns:
        Tuple of (similarity_score, is_match)
    """
    if feature1 is None or feature2 is None:
        raise ValueError("Both features must be provided")
    
    try:
        score = recognizer.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
        return float(score), score >= threshold
    except Exception as e:
        logger.error(f"Face comparison error: {e}")
        return 0.0, False


def draw_face_rectangles(
    frame: np.ndarray,
    faces: Union[np.ndarray, List],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    labels: Optional[List[str]] = None
) -> None:
    """
    Draw rectangles and optional labels on detected faces.

    Args:
        frame: Image to draw on (modified in place)
        faces: List of [x, y, w, h] or ndarray
        color: BGR color tuple
        thickness: Line thickness
        labels: Optional list of labels for each face
    """
    if faces is None:
        return
    
    faces_arr = np.array(faces)
    for i, face in enumerate(faces_arr):
        x, y, w, h = map(int, face[:4])
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        
        if labels and i < len(labels):
            label = str(labels[i])
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x, y - th - 10), (x + tw, y), color, -1)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def get_face_landmarks(face_row: np.ndarray) -> np.ndarray:
    """
    Extract 5-point landmarks from YuNet detection row.

    Args:
        face_row: 15-element array [x, y, w, h, score, l0x, l0y, ..., l4y]

    Returns:
        np.ndarray shape [5, 2] for landmark points:
        [left_eye, right_eye, nose, left_mouth, right_mouth]
    """
    if face_row is None or len(face_row) < 15:
        raise ValueError("Invalid face_row for landmarks extraction")
    
    return np.array([
        [face_row[5], face_row[6]],    # left eye
        [face_row[7], face_row[8]],    # right eye
        [face_row[9], face_row[10]],   # nose
        [face_row[11], face_row[12]],  # left mouth
        [face_row[13], face_row[14]],  # right mouth
    ], dtype=np.float32)


def get_face_detector():
    """Get the loaded YuNet face detector model."""
    return detector


def get_face_recognizer():
    """Get the loaded SFace face recognizer model."""
    return recognizer


__all__ = [
    'detect_faces',
    'extract_face_features',
    'compare_face_features',
    'draw_face_rectangles',
    'get_face_landmarks',
    'get_face_detector',
    'get_face_recognizer',
    'YUNET_PATH',
    'SFACE_PATH',
    'YUNET_INPUT_SIZE',
    'SFACE_SIMILARITY_THRESHOLD',
]
