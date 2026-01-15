"""
Model Download Script
Downloads YuNet and Sface ONNX models for face detection and recognition
"""

# - urllib.request for downloading files
# - os/pathlib for file path handling
# - Optional: hashlib for file verification

import urllib.request
import os
import hashlib
from pathlib import Path

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    # Try to load .env from parent directory (sevices/face-recognition/.env)
    _SCRIPT_DIR = Path(__file__).parent
    env_file = _SCRIPT_DIR.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        # Also try current directory
        load_dotenv(_SCRIPT_DIR / ".env")
except ImportError:
    # python-dotenv not installed, skip .env loading
    pass

# - YuNet model URL and filename
# - Sface model URL and filename

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

# - Define models directory path
# - Get or create models directory
# - Handle relative/absolute paths
# - In Docker: MODELS_PATH is /app/app/models, default to "models" relative to working directory

models_dir = os.environ.get("MODELS_PATH", "models")
os.makedirs(models_dir, exist_ok=True)
yunet_filepath = os.path.join(models_dir, "face_detection_yunet_2023mar.onnx")
sface_filepath = os.path.join(models_dir, "face_recognition_sface_2021dec.onnx")

def download_file(url, filepath, model_name):
    """
    Download a file from URL
    
    Args:
        url: URL to download from
        filepath: Path to save the file
        model_name: Name of model (for display)
    
    Returns:
        bool: True if successful
    """
    
    # Add User-Agent header to avoid 403 errors
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    # Show download progress
    print(f"Downloading {model_name} from {url}...")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
            file_size = int(response.headers.get('Content-Length', 0))
            if file_size > 0:
                print(f"File size: {file_size / (1024*1024):.2f} MB")
            with open(filepath, 'wb') as out_file:
                out_file.write(response.read())
        downloaded_size = os.path.getsize(filepath)
        print(f"Downloaded {model_name} to {filepath} ({downloaded_size / (1024*1024):.2f} MB)")
        return True
    except Exception as e:
        print(f"Error downloading {model_name}: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)  # Remove partial download
        return False

def verify_file(filepath, expected_hash=None):
    """
    Verify downloaded file exists and optionally check hash
    
    Args:
        filepath: Path to file
        expected_hash: Optional SHA256 hash for verification
    
    Returns:
        bool: True if file is valid
    """
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist")
        return False
    # If expected_hash provided, calculate file hash and compare
    if expected_hash:
        with open(filepath, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        if file_hash != expected_hash:
            print(f"File {filepath} hash mismatch")
            return False
    return True

def download_model(model_key, models_dir):
    """
    Download models from single URL
    
    Args:
        model_key: model name
        models_dir: Directory to save models
    
    Returns:
        bool: True if successful
    """
    # Get model info (URL, filename) from configuration
    model_info = {
        'yunet': {
            'url': YUNET_URL,
            'filename': 'face_detection_yunet_2023mar.onnx',
            'hash': None
        },
        'sface': {
            'url': SFACE_URL,
            'filename': 'face_recognition_sface_2021dec.onnx',
            'hash': None
        }
    }
    # Check if model already exists and is valid, skip if valid
    filepath = os.path.join(models_dir, model_info[model_key]['filename'])
    if os.path.exists(filepath):
        # Check if file is valid (not empty, reasonable size for ONNX model)
        file_size = os.path.getsize(filepath)
        if file_size > 1000000:  # ONNX models should be at least 1MB
            print(f"Model {model_key} already exists and is valid ({file_size / (1024*1024):.2f} MB)")
            return True
        else:
            print(f"Model {model_key} exists but appears invalid (size: {file_size} bytes). Re-downloading...")
            os.remove(filepath)  # Remove invalid file
    # Download from URL
    if download_file(model_info[model_key]['url'], filepath, model_key):
        # Verify downloaded file
        if verify_file(filepath, model_info[model_key]['hash']):
            return True
        else:
            print(f"Failed to verify model {model_key}")
            return False
    return False
            
def main():
    """Main function to download all models"""
    # Print header/start message
    print("Downloading models...")
    
    # Get/create models directory
    os.makedirs(models_dir, exist_ok=True)
    # Download each model (yunet, sface)
    if download_model('yunet', models_dir):
        print("YuNet model downloaded successfully")
    else:
        print("Failed to download YuNet model")
    if download_model('sface', models_dir):
        print("Sface model downloaded successfully")
    else:
        print("Failed to download Sface model")

if __name__ == "__main__":
    # Call main() and exit with appropriate code
    main()
    exit(0 if all(download_model(model, models_dir) for model in ['yunet', 'sface']) else 1)
