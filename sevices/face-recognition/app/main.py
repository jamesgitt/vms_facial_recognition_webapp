"""
Face Recognition ML Microservice - Main Entry Point

This is the main entry point for running the Face Recognition API service.
It can be used to run the service locally or as a production entry point.

Usage:
    python main.py                    # Run with default settings
    python main.py --host 0.0.0.0     # Run on specific host
    python main.py --port 8001        # Run on specific port
    python main.py --reload           # Run with auto-reload (development)
"""

import os
import sys
import argparse
from pathlib import Path

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    # Try to load .env from parent directory (sevices/face-recognition/.env)
    _SCRIPT_DIR = Path(__file__).parent
    env_file = _SCRIPT_DIR.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✓ Loaded environment variables from {env_file}")
    else:
        # Also try current directory
        load_dotenv(_SCRIPT_DIR / ".env")
except ImportError:
    # python-dotenv not installed, skip .env loading
    pass

# Ensure the script directory is in Python path for imports
_SCRIPT_DIR = Path(__file__).parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import uvicorn

# Import the FastAPI app (both files are in the same directory in Docker)
# In Docker: working directory is /app/app, so both files are directly accessible
try:
    from face_recog_api import app
except ImportError:
    # Fallback for local development if running from parent directory
    from app.face_recog_api import app

# Try to import download_models from download_models.py
try:
    from download_models import download_models
except ImportError:
    # Fallback for local development if running from parent directory
    try:
        from app.download_models import download_models
    except ImportError:
        download_models = None

def check_models_exist(models_path: str = None) -> bool:
    """
    Check if required ONNX model files exist.
    
    Args:
        models_path: Path to models directory. If None, uses default.
    
    Returns:
        True if all models exist, False otherwise.
    """
    if models_path is None:
        models_path = os.environ.get("MODELS_PATH", str(_SCRIPT_DIR / "models"))
    
    models_dir = Path(models_path)
    yunet_path = models_dir / "face_detection_yunet_2023mar.onnx"
    sface_path = models_dir / "face_recognition_sface_2021dec.onnx"
    
    if not yunet_path.exists():
        print(f"ERROR: YuNet model not found at {yunet_path}")
        return False
    
    if not sface_path.exists():
        print(f"ERROR: Sface model not found at {sface_path}")
        return False
    
    print(f"✓ Models found in {models_dir}")
    return True


def main():
    """Main entry point for the Face Recognition API service."""
    parser = argparse.ArgumentParser(
        description="Face Recognition ML Microservice",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run with defaults
  python main.py --host 0.0.0.0 --port 8000
  python main.py --reload                 # Development mode with auto-reload
  python main.py --workers 4              # Production mode with 4 workers
        """
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("API_HOST", "0.0.0.0"),
        help="Host to bind to (default: from API_HOST env var or 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("API_PORT", "8000")),
        help="Port to bind to (default: from API_PORT env var or 8000)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development (default: False)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1, use 1 for development)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Log level (default: info)"
    )
    
    parser.add_argument(
        "--skip-model-check",
        action="store_true",
        help="Skip checking if models exist before starting (default: False)"
    )
    
    args = parser.parse_args()
    
    # Check models exist (unless skipped)
    if not args.skip_model_check:
        # In Docker: MODELS_PATH is /app/app/models, default to models/ relative to script
        models_path = os.environ.get("MODELS_PATH", str(_SCRIPT_DIR / "models"))
        if not check_models_exist(models_path):
            print("\nModels not found. Attempting to download models...\n")
            if download_models is not None:
                try:
                    download_models(destination=models_path)
                except TypeError:
                    # If old signature, fallback to calling without arguments
                    download_models()
                # Check again after download
                if not check_models_exist(models_path):
                    print("ERROR: Models could not be downloaded or found!")
                    sys.exit(1)
            else:
                print("ERROR: Cannot find or import 'download_models'. Please ensure download_models.py is present.")
                print("\nTo download models, run:")
                print(f"  python {_SCRIPT_DIR / 'download_models.py'}")
                sys.exit(1)
    
    # Validate workers and reload combination
    if args.reload and args.workers > 1:
        print("WARNING: --reload is not compatible with multiple workers. Using 1 worker.")
        args.workers = 1
    
    # Print startup information
    print("\n" + "="*60)
    print("Face Recognition ML Microservice")
    print("="*60)
    print(f"Host:        {args.host}")
    print(f"Port:        {args.port}")
    print(f"Workers:     {args.workers}")
    print(f"Reload:      {args.reload}")
    print(f"Log Level:   {args.log_level}")
    print("="*60)
    print(f"\nAPI will be available at: http://{args.host}:{args.port}")
    print(f"API Documentation: http://{args.host}:{args.port}/docs")
    print(f"Health Check: http://{args.host}:{args.port}/api/v1/health")
    print("\nPress Ctrl+C to stop the server\n")
    
    # Run the server
    try:
        # Use app object directly for better compatibility
        config = uvicorn.Config(
            app=app,
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,
            log_level=args.log_level,
            access_log=True,
        )
        server = uvicorn.Server(config)
        server.run()
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
