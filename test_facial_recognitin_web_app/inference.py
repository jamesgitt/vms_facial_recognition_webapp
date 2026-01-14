# Face Detection and Recognition Inference Module

# Uses downloaded YuNet and Sface ONNX models from test_facial_recognitin_web_app/models

# Load downloaded models
#    - Import cv2, numpy, os
import cv2
import numpy as np
import os   
#    - Define model directory path: 'test_facial_recognitin_web_app/models'
models_dir = 'test_facial_recognitin_web_app/models'
#    - Check if models exist:
yunet_model_path = os.path.join(models_dir, 'face_detection_yunet_2023mar.onnx')
sface_model_path = os.path.join(models_dir, 'face_recognition_sface_2021dec.onnx')
#    - Load YuNet model using cv2.FaceDetectorYN.create()
detector = cv2.FaceDetectorYN.create(
    model=yunet_model_path,
    config='',
    input_size=(320, 320),
    score_threshold=0.6,
    nms_threshold=0.3,
    top_k=5000
)
#    - Load Sface model using cv2.FaceRecognizerSF.create()
recognizer = cv2.FaceRecognizerSF.create(
    model=sface_model_path,
    config='',
    backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
    target_id=cv2.dnn.DNN_TARGET_CPU
)
# Handle missing model files gracefully
if not os.path.exists(yunet_model_path):
    print(f"ERROR: YuNet model not found!")
    print(f"Expected one of:")
    print(f"  - {yunet_model_path}")
    raise FileNotFoundError(f"YuNet model not found. Please ensure the model file exists in {models_dir}/")
if not os.path.exists(sface_model_path):
    print(f"ERROR: Sface model not found!")
    print(f"Expected one of:")
    print(f"  - {sface_model_path}")
    raise FileNotFoundError(f"Sface model not found. Please ensure the model file exists in {models_dir}/")

# Detect faces in an image/frame
#    - Create function to detect faces in an image/frame
#    - Input: image (numpy array, BGR format)
#    - Use YuNet detector to find faces
#    - Return: list of face bounding boxes or face data with landmarks
def detect_faces(frame):
    """
    Detect faces in an image/frame
    """
    faces = detector.detect(frame)
    return faces

# Extract face features
#    - Create function to extract face embeddings
#    - Input: image and detected face data
#    - Use Sface recognizer to align and crop face
#    - Extract 512-dimensional feature vector
#    - Return: feature vector (numpy array)
def extract_face_features(frame, face_data):
    """
    Extract face features from a detected face
    """
    face = recognizer.alignCrop(frame, face_data)
    return recognizer.feature(face)

# Compare faces
#    - Create function to compare two face features
#    - Input: two feature vectors
#    - Calculate similarity score (cosine similarity)
#    - Return: similarity score and match result (True/False)

# 5. SIMPLE USAGE INTERFACE
#    - Create main class or simple functions for easy use
#    - Example usage patterns:
#      * detect_faces(image) -> faces
#      * extract_features(image, face) -> features
#      * compare_features(features1, features2) -> match
#    - Add error handling and validation
#    - Return clear results for easy integration

