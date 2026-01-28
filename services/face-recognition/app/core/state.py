"""
Application State Management

Provides a singleton AppState class to manage global application state
instead of using scattered global variables.

Usage:
    from core.state import app_state
    
    # During startup
    app_state.face_detector = inference.get_face_detector()
    app_state.hnsw_manager = HNSWIndexManager(...)
    
    # In routes/services
    if app_state.face_detector is None:
        raise ModelNotLoadedError("face_detector")
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import threading


@dataclass
class AppState:
    """
    Singleton class to hold application-wide state.
    
    Replaces scattered global variables with a centralized,
    type-hinted state container.
    
    Attributes:
        face_detector: Loaded YuNet face detector model
        face_recognizer: Loaded SFace face recognizer model
        hnsw_manager: HNSW index manager instance
        visitor_features: In-memory cache of visitor features (for test_images fallback)
        use_database: Whether database integration is active
        db_pool: Database connection pool
        initialized: Whether startup initialization is complete
    """
    
    # ML Models
    face_detector: Optional[Any] = None
    face_recognizer: Optional[Any] = None
    
    # HNSW Index
    hnsw_manager: Optional[Any] = None
    
    # Visitor data cache (for test_images fallback)
    visitor_features: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Database state
    use_database: bool = False
    db_pool: Optional[Any] = None
    
    # Initialization flag
    initialized: bool = False
    
    # Thread lock for thread-safe modifications
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    
    def reset(self) -> None:
        """Reset all state to initial values."""
        with self._lock:
            self.face_detector = None
            self.face_recognizer = None
            self.hnsw_manager = None
            self.visitor_features.clear()
            self.use_database = False
            self.db_pool = None
            self.initialized = False
    
    @property
    def models_loaded(self) -> bool:
        """Check if both ML models are loaded."""
        return self.face_detector is not None and self.face_recognizer is not None
    
    @property
    def index_available(self) -> bool:
        """Check if HNSW index is available and has data."""
        if self.hnsw_manager is None:
            return False
        try:
            return self.hnsw_manager.ntotal > 0
        except Exception:
            return False
    
    @property
    def visitor_count(self) -> int:
        """Get total number of indexed visitors."""
        if self.hnsw_manager is not None:
            try:
                return self.hnsw_manager.ntotal
            except Exception:
                pass
        return len(self.visitor_features)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current state status for health checks."""
        return {
            "initialized": self.initialized,
            "models_loaded": self.models_loaded,
            "face_detector": self.face_detector is not None,
            "face_recognizer": self.face_recognizer is not None,
            "hnsw_available": self.hnsw_manager is not None,
            "hnsw_indexed": self.index_available,
            "visitor_count": self.visitor_count,
            "use_database": self.use_database,
            "db_pool_active": self.db_pool is not None,
        }


# Singleton instance
app_state = AppState()


def get_app_state() -> AppState:
    """
    Get the application state singleton.
    
    Can be used as a FastAPI dependency:
        def get_state() -> AppState:
            return get_app_state()
        
        @app.get("/status")
        def status(state: AppState = Depends(get_state)):
            return state.get_status()
    """
    return app_state


__all__ = [
    "AppState",
    "app_state",
    "get_app_state",
]
