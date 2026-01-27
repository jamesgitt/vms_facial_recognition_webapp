"""
Face Recognition ML Microservice - Main Entry Point

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
from typing import Optional

# Set up the script directory
_SCRIPT_DIR = Path(__file__).parent.resolve()

# Load environment variables in priority order: ../.env, ./.env
try:
    from dotenv import load_dotenv
    for env_path in [_SCRIPT_DIR.parent / ".env", _SCRIPT_DIR / ".env"]:
        if env_path.exists():
            load_dotenv(env_path)
            print(f"[OK] Loaded environment from {env_path}")
            break
except ImportError:
    pass

# Ensure script directory is in Python path
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import uvicorn

# Import FastAPI app
try:
    from face_recog_api import app
except ImportError:
    from app.face_recog_api import app

# Model file constants
YUNET_FILENAME = 'face_detection_yunet_2023mar.onnx'
SFACE_FILENAME = 'face_recognition_sface_2021dec.onnx'

def get_models_path() -> Path:
    """Return the path to the models directory."""
    return Path(os.environ.get("MODELS_PATH", _SCRIPT_DIR / "models"))

def check_models_exist(models_path: Optional[Path] = None) -> bool:
    """
    Return True if required ONNX model files exist, False otherwise.
    """
    if models_path is None:
        models_path = get_models_path()

    yunet_path = models_path / YUNET_FILENAME
    sface_path = models_path / SFACE_FILENAME

    if not yunet_path.exists():
        print(f"[ERROR] YuNet model not found at {yunet_path}")
        return False

    if not sface_path.exists():
        print(f"[ERROR] SFace model not found at {sface_path}")
        return False

    print(f"[OK] Models found in {models_path}")
    return True

def download_models_if_needed(models_path: Path) -> bool:
    """
    Attempt to download models if they don't exist.
    Return True if models are available; otherwise, False.
    """
    if check_models_exist(models_path):
        return True

    print("\nModels not found. Attempting to download...\n")
    downloader_variants = [
        ("download_models", "main"),
        ("app.download_models", "main"),
    ]
    for module_name, func_name in downloader_variants:
        try:
            mod = __import__(module_name, fromlist=[func_name])
            getattr(mod, func_name)()
            return check_models_exist(models_path)
        except ImportError:
            continue

    print("[ERROR] Cannot import download_models module.")
    print(f"\nTo download models manually, run:")
    print(f"  python {_SCRIPT_DIR / 'download_models.py'}")
    return False

def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Face Recognition ML Microservice",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run with defaults
  python main.py --host 0.0.0.0 --port 8000
  python main.py --reload                 # Development mode
  python main.py --workers 4              # Production mode
        """
    )

    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("API_HOST", "0.0.0.0"),
        help="Host to bind to (default: API_HOST env or 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("API_PORT", "8000")),
        help="Port to bind to (default: API_PORT env or 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Log level (default: info)",
    )
    parser.add_argument(
        "--skip-model-check",
        action="store_true",
        help="Skip checking if models exist before starting",
    )

    return parser

def print_startup_info(host: str, port: int, workers: int, reload: bool, log_level: str) -> None:
    """Print summary of server startup parameters."""
    print("\n" + "=" * 60)
    print("Face Recognition ML Microservice")
    print("=" * 60)
    print(f"Host:        {host}")
    print(f"Port:        {port}")
    print(f"Workers:     {workers}")
    print(f"Reload:      {reload}")
    print(f"Log Level:   {log_level}")
    print("=" * 60)
    print(f"\nAPI: http://{host}:{port}")
    print(f"Docs: http://{host}:{port}/docs")
    print(f"Health: http://{host}:{port}/api/v1/health")
    print("\nPress Ctrl+C to stop\n")

def run_server(host: str, port: int, workers: int, reload: bool, log_level: str) -> None:
    """Start the uvicorn server with the desired settings."""
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        reload=reload,
        workers=1 if reload else workers,
        log_level=log_level,
        access_log=True,
    )
    server = uvicorn.Server(config)
    server.run()

def main() -> int:
    """
    Main entry point for the Face Recognition API service.

    Returns:
        0 = success, 1 = failure
    """
    parser = create_parser()
    args = parser.parse_args()

    # Check models exist (unless skipped)
    if not args.skip_model_check:
        if not download_models_if_needed(get_models_path()):
            return 1

    # Check compatibility: reload mode with >1 worker is not supported
    if args.reload and args.workers > 1:
        print("[WARNING] --reload not compatible with multiple workers. Using 1 worker.")
        args.workers = 1

    print_startup_info(args.host, args.port, args.workers, args.reload, args.log_level)

    try:
        run_server(args.host, args.port, args.workers, args.reload, args.log_level)
        return 0
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        return 0
    except Exception as e:
        print(f"\n[ERROR] Failed to start server: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
