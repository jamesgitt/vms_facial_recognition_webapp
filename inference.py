import os
import cv2
import numpy as np
import time
import urllib.request

class FaceDetector:
    """
    Face detection using OpenCV's YuNet model.
    Face recognition using OpenCV's Sface model.
    """
    
    def __init__(self, model_dir='models'):
        """
        Initialize YuNet face detector and Sface recognizer.
        
        Args:
            model_dir: Directory to store model files
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Model file paths - check for both short and full names
        yunet_full = os.path.join(model_dir, 'face_detection_yunet_2023mar.onnx')
        yunet_model_path = yunet_full if os.path.exists(yunet_full) else None
        
        sface_full = os.path.join(model_dir, 'face_recognition_sface_2021dec.onnx')
        sface_model_path = sface_full if os.path.exists(sface_full) else None
        
        # Check if models exist, if not provide helpful error
        if not os.path.exists(yunet_model_path):
            print(f"ERROR: YuNet model not found!")
            print(f"Expected one of:")
            print(f"  - {yunet_full}")
            raise FileNotFoundError(f"YuNet model not found. Please ensure the model file exists in {model_dir}/")
        
        if not os.path.exists(sface_model_path):
            print(f"ERROR: Sface model not found!")
            print(f"Expected one of:")
            print(f"  - {sface_full}")
            raise FileNotFoundError(f"Sface model not found. Please ensure the model file exists in {model_dir}/")
        
        print(f"Using YuNet model: {os.path.basename(yunet_model_path)}")
        print(f"Using Sface model: {os.path.basename(sface_model_path)}")
        
        # Initialize YuNet face detector
        self.detector = cv2.FaceDetectorYN.create(
            model=yunet_model_path,
            config='',
            input_size=(320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000
        )
        
        # Initialize Sface face recognizer
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=sface_model_path,
            config='',
            backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )
        
        print("YuNet face detector and Sface recognizer initialized successfully!")
    

    def detect_faces(self, frame, resize_factor=1.0, score_threshold=0.6, return_landmarks=False):
        """
        Detect faces in a frame using YuNet.
        
        Args:
            frame: Input image/frame (BGR format)
            resize_factor: Resize frame before detection (0.0-1.0) for speed.
                         1.0 = full resolution, 0.5 = half resolution
            score_threshold: Confidence threshold for detection (0.0-1.0)
            return_landmarks: If True, returns full face data with landmarks for Sface
        
        Returns:
            If return_landmarks=False: List of face bounding boxes [(x, y, w, h), ...]
            If return_landmarks=True: List of face data arrays with landmarks
        """
        try:
            # Resize frame for faster processing if needed
            original_h, original_w = frame.shape[:2]
            if resize_factor < 1.0:
                new_w = int(original_w * resize_factor)
                new_h = int(original_h * resize_factor)
                resized_frame = cv2.resize(frame, (new_w, new_h))
            else:
                resized_frame = frame
                new_w, new_h = original_w, original_h
            
            # Set input size for detector
            self.detector.setInputSize((new_w, new_h))
            self.detector.setScoreThreshold(score_threshold)
            
            # Detect faces
            _, faces = self.detector.detect(resized_frame)
            
            if faces is None:
                return []
            
            # Scale factor to convert back to original resolution
            scale_x = original_w / new_w
            scale_y = original_h / new_h
            
            if return_landmarks:
                # Return full face data with landmarks (for Sface)
                scaled_faces = []
                for face in faces:
                    scaled_face = face.copy()
                    # YuNet format: [x, y, w, h, confidence?, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm]
                    # Could be 14 or 15 elements depending on version
                    
                    # Scale bounding box coordinates
                    scaled_face[0] *= scale_x  # x
                    scaled_face[1] *= scale_y  # y
                    scaled_face[2] *= scale_x  # w
                    scaled_face[3] *= scale_y  # h
                    
                    # Determine landmark start index (skip confidence if present)
                    # If 15 elements, confidence is at index 4, landmarks start at 5
                    # If 14 elements, landmarks start at 4
                    landmark_start = 5 if len(scaled_face) == 15 else 4
                    
                    # Scale landmarks (pairs of x, y coordinates)
                    for i in range(landmark_start, len(scaled_face) - 1, 2):
                        if i + 1 < len(scaled_face):
                            scaled_face[i] *= scale_x     # x coordinates
                            scaled_face[i+1] *= scale_y   # y coordinates
                    
                    scaled_faces.append(scaled_face.astype(np.float32))
                return scaled_faces
            else:
                # Extract bounding boxes only
                face_boxes = []
                for face in faces:
                    # YuNet returns: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm]
                    x, y, w, h = face[:4].astype(int)
                    x = int(x * scale_x)
                    y = int(y * scale_y)
                    w = int(w * scale_x)
                    h = int(h * scale_y)
                    if w > 0 and h > 0:
                        face_boxes.append((x, y, w, h))
                return face_boxes
            
        except Exception as e:
            print(f"Error detecting faces: {e}")
            return []
    
    def extract_face_features(self, frame, face_data):
        """
        Extract face features using Sface for recognition.
        
        Args:
            frame: Input image/frame (BGR format)
            face_data: Face data array from YuNet with landmarks [x, y, w, h, confidence?, landmarks...]
        
        Returns:
            Face feature vector (512-dim) or None
        """
        try:
            # face_data format: [x, y, w, h, confidence?, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm]
            # Could be 14 or 15 elements depending on YuNet version
            if len(face_data) < 14:
                return None
            
            # Sface alignCrop expects exactly 14 elements: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm]
            # If face_data has 15 elements (with confidence), remove confidence
            if len(face_data) == 15:
                # Remove confidence at index 4: [x, y, w, h, conf, landmarks...] -> [x, y, w, h, landmarks...]
                face_data_for_sface = np.concatenate([face_data[:4], face_data[5:]])
            else:
                face_data_for_sface = face_data
            
            # Ensure we have exactly 14 elements
            if len(face_data_for_sface) != 14:
                return None
            
            # Align and crop face using landmarks (Sface requires aligned face)
            aligned_face = self.recognizer.alignCrop(frame, face_data_for_sface)
            
            if aligned_face is None or aligned_face.size == 0:
                return None
            
            # Extract features
            face_feature = self.recognizer.feature(aligned_face)
            
            return face_feature
            
        except Exception as e:
            print(f"Error extracting face features: {e}")
            return None
    
    def compare_faces(self, feature1, feature2, threshold=0.363):
        """
        Compare two face features using cosine similarity.
        
        Args:
            feature1: First face feature vector
            feature2: Second face feature vector
            threshold: Similarity threshold (default 0.363 for Sface)
        
        Returns:
            (similarity_score, is_match)
        """
        if feature1 is None or feature2 is None:
            return 0.0, False
        
        try:
            # Calculate cosine similarity
            score = self.recognizer.match(
                feature1, 
                feature2, 
                cv2.FaceRecognizerSF_FR_COSINE
            )
            
            is_match = score >= threshold
            return score, is_match
            
        except Exception as e:
            print(f"Error comparing faces: {e}")
            return 0.0, False


def draw_rectangles(frame, faces, color=(0, 255, 0), thickness=2, labels=None):
    """
    Draw rectangles on detected faces with optional labels.
    
    Args:
        frame: Image frame to draw on
        faces: List of face bounding boxes [(x, y, w, h), ...]
        color: BGR color tuple for rectangles
        thickness: Line thickness
        labels: Optional list of labels to display above each face
    """
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        
        # Add label if provided
        if labels and i < len(labels):
            label = labels[i]
            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            # Draw background rectangle for text
            cv2.rectangle(
                frame, 
                (x, y - text_height - 10), 
                (x + text_width, y), 
                color, 
                -1
            )
            cv2.putText(
                frame, label, (x, y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )
        else:
            cv2.putText(
                frame, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
