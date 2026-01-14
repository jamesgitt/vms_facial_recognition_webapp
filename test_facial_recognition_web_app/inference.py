"""
Face Detection and Recognition Inference Utilities

Loads and configures YuNet (face detection) and Sface (recognition) ONNX models.
Designed as a reusable utility for later integration into an API or web service layer.
No CLI or app logic is included here.

Exports:
- detect_faces: Face detection with YuNet
- extract_face_features: Feature extraction with Sface
- compare_face_features: Cosine similarity and verdict between two feature vectors
- Utility (configuration, validation, alignment, rectangle drawing)
"""

import cv2
import numpy as np
import os

# ------------------- CONFIGURATION -------------------

DEFAULT_MODELS_DIR = 'test_facial_recognitin_web_app/models'

YUNET_FILENAME = 'face_detection_yunet_2023mar.onnx'
SFACE_FILENAME = 'face_recognition_sface_2021dec.onnx'

YUNET_PATH = os.path.join(DEFAULT_MODELS_DIR, YUNET_FILENAME)
SFACE_PATH = os.path.join(DEFAULT_MODELS_DIR, SFACE_FILENAME)

# Default model params; can be adjusted at import time for your API needs
YUNET_INPUT_SIZE = (320, 320)
YUNET_SCORE_THRESHOLD = 0.6
YUNET_NMS_THRESHOLD = 0.3
YUNET_TOP_K = 5000
SFACE_SIMILARITY_THRESHOLD = 0.363  # Empirical threshold for same/not-same under Cosine

# ---------------- ERROR HANDLING AND MODEL LOADING ----------------

def _verify_model_file(path, model_name="model"):
    """Raise error if model file missing."""
    if not os.path.exists(path):
        print(f"ERROR: {model_name} not found at {path}")
        raise FileNotFoundError(f"{model_name} model not found. Please ensure the file exists at {path}")

# Check for required model files up front for clearer error tracing
_verify_model_file(YUNET_PATH, "YuNet")
_verify_model_file(SFACE_PATH, "Sface")

try:
    detector = cv2.FaceDetectorYN.create(
        model=YUNET_PATH,
        config='',
        input_size=YUNET_INPUT_SIZE,
        score_threshold=YUNET_SCORE_THRESHOLD,
        nms_threshold=YUNET_NMS_THRESHOLD,
        top_k=YUNET_TOP_K
    )
except Exception as e:
    raise RuntimeError(f"Failed to initialize YuNet model: {str(e)}")

try:
    recognizer = cv2.FaceRecognizerSF.create(
        model=SFACE_PATH,
        config='',
        backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
        target_id=cv2.dnn.DNN_TARGET_CPU
    )
except Exception as e:
    raise RuntimeError(f"Failed to initialize Sface model: {str(e)}")

# ----------- UTILITY FUNCTIONS ----------------

def detect_faces(frame, input_size=YUNET_INPUT_SIZE, score_threshold=YUNET_SCORE_THRESHOLD,
                 nms_threshold=YUNET_NMS_THRESHOLD, return_landmarks=False):
    """
    Detect faces in a BGR image using YuNet.

    Args:
        frame (np.ndarray): Input BGR image.
        input_size (tuple): Input size for detector.
        score_threshold (float): Detection threshold.
        nms_threshold (float): NMS suppression threshold.
        return_landmarks (bool): If True, return landmark data also.

    Returns:
        faces: None if no faces, else np.ndarray shape [num_faces, 15] per YuNet docs.
    """
    if frame is None or not hasattr(frame, 'shape'):
        raise ValueError("Frame is None or invalid")
    
    # YuNet requires input image with size exactly input_size
    resized = cv2.resize(frame, input_size)
    detector.setInputSize(input_size)

    try:
        retval, faces = detector.detect(resized)
    except Exception as e:
        print(f"Detection error: {e}")
        return None

    if faces is None or len(faces) == 0:
        return None

    # Optionally re-scale detected boxes/landmarks to original frame size
    sx = frame.shape[1] / input_size[0]
    sy = frame.shape[0] / input_size[1]
    faces_rescaled = faces.astype(np.float32).copy()
    faces_rescaled[:, [0, 2, 5, 7, 9,11,13]] *= sx  # x, w, landmarks_x
    faces_rescaled[:, [1, 3, 6, 8,10,12,14]] *= sy  # y, h, landmarks_y

    if return_landmarks:
        # Return full face data (box, score, and 5 landmarks) as np.ndarray
        return faces_rescaled
    else:
        # Only return bounding boxes [(x, y, w, h), ...]
        bboxes = faces_rescaled[:, :4].astype(int)
        return [tuple(bbox) for bbox in bboxes]

def extract_face_features(frame, face_row):
    """
    Extract 512-dim Sface feature vector from a given face.

    Args:
        frame (np.ndarray): Full BGR image.
        face_row (array-like): Face detection output row from YuNet [x, y, w, h, score, ...landmarks]
    Returns:
        np.ndarray: Face feature vector (512-dim float32 shape).
    """
    if frame is None or face_row is None:
        raise ValueError("Input frame or face_row is missing")
    try:
        aligned = recognizer.alignCrop(frame, face_row)
        feature = recognizer.feature(aligned)
        return feature
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

def compare_face_features(feature1, feature2, threshold=SFACE_SIMILARITY_THRESHOLD):
    """
    Compare two face features and return similarity score + match verdict.

    Args:
        feature1, feature2: Output from extract_face_features (np.ndarray, 512-dim)
        threshold (float): Similarity threshold for "same" acceptance

    Returns:
        (score: float, is_same_person: bool)
    """
    if feature1 is None or feature2 is None:
        raise ValueError("Both features must be np.ndarray")
    try:
        # Uses cosine similarity as default
        score = recognizer.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
        is_match = score >= threshold
        return score, is_match
    except Exception as e:
        print(f"Face comparison error: {e}")
        return 0.0, False

def draw_face_rectangles(frame, faces, color=(0,255,0), thickness=2, labels=None):
    """
    Draw rectangles (and optional labels) on faces on the frame.

    Args:
        frame: image (np.ndarray)
        faces: list of [x, y, w, h] or [num_faces, 4] ndarray
        color: BGR tuple
        thickness: Rectangle line thickness
        labels: Optional [str] list of annotation for each face
    """
    if faces is None:
        return
    # Accept np.ndarray or list of tuples
    faces = np.array(faces)
    for i, face in enumerate(faces):
        x, y, w, h = map(int, face[:4])
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        if labels and i < len(labels):
            label = str(labels[i])
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x, y - th - 10), (x + tw, y), color, -1)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        else:
            cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def get_face_landmarks(face_row):
    """
    Extract landmarks (5 points) from YuNet face detection row.

    Args:
        face_row: 15 element np.ndarray from YuNet detection: 
        [x, y, w, h, score, l0x, l0y, l1x, l1y, ... l4y]
    Returns:
        np.ndarray shape [5, 2] for 5 landmark points
    """
    if face_row is None or len(face_row) < 15:
        raise ValueError("Invalid face_row for landmarks extraction")
    # landmarks: l0x, l0y, l1x, l1y, ..., l4y
    landmarks = np.array([
        [face_row[5],  face_row[6]],   # left eye
        [face_row[7],  face_row[8]],   # right eye
        [face_row[9], face_row[10]],   # nose
        [face_row[11],face_row[12]],   # left mouth
        [face_row[13],face_row[14]],   # right mouth
    ], dtype=np.float32)
    return landmarks

# --------------- MAIN API EXPORTS ---------------

__all__ = [
    'detect_faces',
    'extract_face_features',
    'compare_face_features',
    'draw_face_rectangles',
    'get_face_landmarks',
    'YUNET_PATH',
    'SFACE_PATH',
    'YUNET_INPUT_SIZE',
    'SFACE_SIMILARITY_THRESHOLD'
]