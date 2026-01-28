"""
Model Download Script
Downloads YuNet and SFace ONNX models for face detection and recognition.
"""

import os
import sys
import hashlib
import urllib.request
from pathlib import Path
from typing import Optional

from core.logger import get_logger
logger = get_logger(__name__)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    _SCRIPT_DIR = Path(__file__).parent
    env_file = _SCRIPT_DIR.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv(_SCRIPT_DIR / ".env")
except ImportError:
    pass

# Model configuration
MODELS_DIR = os.environ.get("MODELS_PATH", "models")
MIN_MODEL_SIZE_BYTES = 1_000_000  # ONNX models should be at least 1MB

MODEL_INFO = {
    'yunet': {
        'url': "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        'filename': 'face_detection_yunet_2023mar.onnx',
        'hash': None
    },
    'sface': {
        'url': "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        'filename': 'face_recognition_sface_2021dec.onnx',
        'hash': None
    }
}

# HTTP headers to avoid 403 errors
REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}


def _format_size(size_bytes: int) -> str:
    """Format byte size as MB string."""
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def download_file(url: str, filepath: str, model_name: str) -> bool:
    """
    Download a file from URL.
    
    Args:
        url: URL to download from
        filepath: Path to save the file
        model_name: Name of model (for display)
    
    Returns:
        True if download successful, False otherwise
    """
    logger.info(f"Downloading {model_name} from {url}...")
    
    try:
        request = urllib.request.Request(url, headers=REQUEST_HEADERS)
        with urllib.request.urlopen(request) as response:
            file_size = int(response.headers.get('Content-Length', 0))
            if file_size > 0:
                logger.info(f"File size: {_format_size(file_size)}")
            
            with open(filepath, 'wb') as out_file:
                out_file.write(response.read())
        
        downloaded_size = os.path.getsize(filepath)
        logger.info(f"Downloaded {model_name} to {filepath} ({_format_size(downloaded_size)})")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading {model_name}: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False


def verify_file(filepath: str, expected_hash: Optional[str] = None) -> bool:
    """
    Verify downloaded file exists and optionally check hash.
    
    Args:
        filepath: Path to file
        expected_hash: Optional SHA256 hash for verification
    
    Returns:
        True if file is valid, False otherwise
    """
    if not os.path.exists(filepath):
        logger.error(f"File {filepath} does not exist")
        return False
    
    if expected_hash:
        with open(filepath, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        if file_hash != expected_hash:
            logger.error(f"File {filepath} hash mismatch")
            return False
    
    return True


def download_model(model_key: str, models_dir: str = MODELS_DIR) -> bool:
    """
    Download a model if not already present.
    
    Args:
        model_key: Model identifier ('yunet' or 'sface')
        models_dir: Directory to save models
    
    Returns:
        True if model is available (downloaded or already exists), False otherwise
    """
    if model_key not in MODEL_INFO:
        logger.error(f"Unknown model: {model_key}")
        return False
    
    info = MODEL_INFO[model_key]
    filepath = os.path.join(models_dir, info['filename'])
    
    # Check if model already exists and is valid
    if os.path.exists(filepath):
        file_size = os.path.getsize(filepath)
        if file_size >= MIN_MODEL_SIZE_BYTES:
            logger.info(f"Model {model_key} already exists ({_format_size(file_size)})")
            return True
        logger.warning(f"Model {model_key} appears invalid ({file_size} bytes). Re-downloading...")
        os.remove(filepath)
    
    # Download and verify
    if not download_file(info['url'], filepath, model_key):
        return False
    
    if not verify_file(filepath, info['hash']):
        logger.error(f"Failed to verify model {model_key}")
        return False
    
    return True


def main() -> int:
    """
    Download all required models.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    logger.info("Downloading models...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    success = True
    for model_key in MODEL_INFO:
        if download_model(model_key):
            logger.info(f"{model_key.upper()} model ready")
        else:
            logger.error(f"Failed to download {model_key.upper()} model")
            success = False
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
