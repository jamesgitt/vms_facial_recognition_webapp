"""
Face Recognition ML Microservice - Main Entry Point

Usage:
    python main.py                    # Run with default settings
    python main.py --host 0.0.0.0     # Run on specific host
    python main.py --port 8001        # Run on specific port
    python main.py --reload           # Run with auto-reload (development)
"""

import sys
import argparse
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

# Set up the script directory first
_SCRIPT_DIR = Path(__file__).parent.resolve()

# Ensure script directory is in Python path
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Now import from core
from core.logger import get_logger
from core.config import settings
from core.state import app_state

logger = get_logger("main")

# Load environment variables in priority order
try:
    from dotenv import load_dotenv
    for env_path in [_SCRIPT_DIR.parent / ".env", _SCRIPT_DIR / ".env"]:
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"Loaded environment from {env_path}")
            break
except ImportError:
    pass

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import API routers
from api import api_router

# Import initialization functions
from pipelines.visitor_loader import initialize_all
from ml import inference


# =============================================================================
# LIFESPAN CONTEXT MANAGER
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # === STARTUP ===
    logger.info("Starting Face Recognition API...")
    
    try:
        # Load ML models
        logger.info("Loading ML models...")
        app_state.face_detector = inference.get_face_detector()
        app_state.face_recognizer = inference.get_face_recognizer()
        logger.info("ML models loaded successfully")
        
        # Initialize database, HNSW index, and load visitors
        initialize_all()
        
        logger.info("Face Recognition API started successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield  # Application runs here
    
    # === SHUTDOWN ===
    logger.info("Shutting down Face Recognition API...")
    
    # Reset app state
    app_state.reset()
    
    logger.info("Shutdown complete")


# =============================================================================
# CREATE FASTAPI APP
# =============================================================================

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI instance
    """
    app = FastAPI(
        title=settings.api.title,
        version=settings.api.version,
        description=settings.api.description,
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors.origins,
        allow_credentials=settings.cors.allow_credentials,
        allow_methods=settings.cors.allow_methods,
        allow_headers=settings.cors.allow_headers,
    )
    
    # Include API routes
    app.include_router(api_router)
    
    # Add exception handlers
    @app.exception_handler(ValueError)
    async def valueerror_handler(request, exc):
        return JSONResponse(
            status_code=400,
            content={"error": str(exc), "type": "ValueError"}
        )
    
    @app.exception_handler(FileNotFoundError)
    async def filenotfounderror_handler(request, exc):
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "type": "FileNotFoundError"}
        )
    
    return app


# Create the app instance
app = create_app()


# =============================================================================
# MODEL CHECKING UTILITIES
# =============================================================================

YUNET_FILENAME = 'face_detection_yunet_2023mar.onnx'
SFACE_FILENAME = 'face_recognition_sface_2021dec.onnx'


def get_models_path() -> Path:
    """Return the path to the models directory."""
    return Path(settings.models.models_path)


def check_models_exist(models_path: Optional[Path] = None) -> bool:
    """
    Return True if required ONNX model files exist, False otherwise.
    """
    if models_path is None:
        models_path = get_models_path()

    yunet_path = models_path / YUNET_FILENAME
    sface_path = models_path / SFACE_FILENAME

    if not yunet_path.exists():
        logger.error(f"YuNet model not found at {yunet_path}")
        return False

    if not sface_path.exists():
        logger.error(f"SFace model not found at {sface_path}")
        return False

    logger.info(f"Models found in {models_path}")
    return True


def download_models_if_needed(models_path: Path) -> bool:
    """
    Attempt to download models if they don't exist.
    Return True if models are available; otherwise, False.
    """
    if check_models_exist(models_path):
        return True

    logger.warning("Models not found. Attempting to download...")

    try:
        from ml.download_models import main as download_main
        download_main()
        return check_models_exist(models_path)
    except ImportError as e:
        logger.error(f"Cannot import download_models module: {e}")
    logger.error(f"To download models manually, run:\n  python {_SCRIPT_DIR / 'ml' / 'download_models.py'}")
    return False


# =============================================================================
# CLI
# =============================================================================

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
        default=settings.api.host,
        help=f"Host to bind to (default: {settings.api.host})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.api.port,
        help=f"Port to bind to (default: {settings.api.port})",
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
        default=settings.logging.log_level.lower(),
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help=f"Log level (default: {settings.logging.log_level.lower()})",
    )
    parser.add_argument(
        "--skip-model-check",
        action="store_true",
        help="Skip checking if models exist before starting",
    )

    return parser


def print_startup_info(host: str, port: int, workers: int, reload: bool, log_level: str) -> None:
    """Print summary of server startup parameters."""
    logger.info("=" * 60)
    logger.info("Face Recognition ML Microservice")
    logger.info("=" * 60)
    logger.info(f"Host:        {host}")
    logger.info(f"Port:        {port}")
    logger.info(f"Workers:     {workers}")
    logger.info(f"Reload:      {reload}")
    logger.info(f"Log Level:   {log_level}")
    logger.info("=" * 60)
    logger.info(f"API: http://{host}:{port}")
    logger.info(f"Docs: http://{host}:{port}/docs")
    logger.info(f"Health: http://{host}:{port}/api/v1/health")
    logger.info("Press Ctrl+C to stop")


def run_server(host: str, port: int, workers: int, reload: bool, log_level: str) -> None:
    """Start the uvicorn server with the desired settings."""
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        workers=1 if reload else workers,
        log_level=log_level,
        access_log=True,
    )


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
        logger.warning("--reload not compatible with multiple workers. Using 1 worker.")
        args.workers = 1

    print_startup_info(args.host, args.port, args.workers, args.reload, args.log_level)

    try:
        run_server(args.host, args.port, args.workers, args.reload, args.log_level)
        return 0
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        return 0
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
