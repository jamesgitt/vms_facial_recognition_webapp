"""
Centralized Configuration for Face Recognition API

All configuration is loaded from environment variables with sensible defaults.
Uses Pydantic Settings for validation and type coercion.

Usage:
    from core.config import settings
    
    print(settings.models_path)
    print(settings.db_table_name)
"""

import os
from typing import Optional, List, Tuple
from pathlib import Path
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# SETTINGS CLASSES
# =============================================================================

class APISettings(BaseSettings):
    """API server configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="API_",
        extra="ignore",
    )
    
    host: str = Field(default="0.0.0.0", description="API server host")
    port: int = Field(default=8000, description="API server port")
    title: str = Field(default="Face Recognition API", description="API title")
    description: str = Field(
        default="REST API for face detection and recognition using YuNet and SFace models.",
        description="API description for OpenAPI docs"
    )
    version: str = Field(default="1.0.0", description="API version")
    debug: bool = Field(default=False, description="Enable debug mode")


class CORSSettings(BaseSettings):
    """CORS configuration."""
    
    model_config = SettingsConfigDict(extra="ignore")
    
    cors_origins: str = Field(
        default="*",
        description="Comma-separated list of allowed origins"
    )
    allow_credentials: bool = Field(
        default=True,
        description="Allow credentials in CORS requests"
    )
    allow_methods: List[str] = Field(
        default=["*"],
        description="Allowed HTTP methods"
    )
    allow_headers: List[str] = Field(
        default=["*"],
        description="Allowed HTTP headers"
    )
    
    @property
    def origins(self) -> List[str]:
        """Parse CORS origins into a list."""
        if not self.cors_origins or self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    @property
    def origins_list(self) -> List[str]:
        """Alias for origins property."""
        return self.origins


class ModelSettings(BaseSettings):
    """ML model configuration."""
    
    model_config = SettingsConfigDict(extra="ignore")
    
    # Paths
    models_path: str = Field(
        default="models",
        description="Path to ONNX model files"
    )
    
    # YuNet (face detection) settings
    yunet_filename: str = Field(
        default="face_detection_yunet_2023mar.onnx",
        description="YuNet model filename"
    )
    yunet_input_size: Tuple[int, int] = Field(
        default=(640, 640),
        description="YuNet input size (width, height)"
    )
    yunet_score_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="YuNet detection confidence threshold"
    )
    yunet_nms_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="YuNet NMS threshold"
    )
    yunet_top_k: int = Field(
        default=5000,
        description="YuNet maximum detections"
    )
    
    # SFace (face recognition) settings
    sface_filename: str = Field(
        default="face_recognition_sface_2021dec.onnx",
        description="SFace model filename"
    )
    sface_similarity_threshold: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="SFace similarity threshold for matching"
    )
    sface_feature_dim: int = Field(
        default=128,
        description="SFace feature vector dimension"
    )
    
    @property
    def yunet_path(self) -> str:
        """Full path to YuNet model."""
        return os.path.join(self.models_path, self.yunet_filename)
    
    @property
    def sface_path(self) -> str:
        """Full path to SFace model."""
        return os.path.join(self.models_path, self.sface_filename)


class DatabaseSettings(BaseSettings):
    """Database configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="DB_",
        extra="ignore",
    )
    
    # Connection settings
    use_database: bool = Field(
        default=False,
        alias="USE_DATABASE",
        description="Enable database integration"
    )
    database_url: Optional[str] = Field(
        default=None,
        alias="DATABASE_URL",
        description="Full database connection URL"
    )
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    name: str = Field(default="visitors_db", description="Database name")
    user: str = Field(default="postgres", description="Database user")
    password: str = Field(default="", description="Database password")
    
    # Table/column configuration
    table_name: str = Field(
        default='public."Visitor"',
        description="Visitor table name"
    )
    visitor_id_column: str = Field(
        default="id",
        description="Visitor ID column name"
    )
    image_column: str = Field(
        default="base64Image",
        description="Base64 image column name"
    )
    features_column: str = Field(
        default="facefeatures",
        description="Face features column name"
    )
    visitor_limit: Optional[int] = Field(
        default=None,
        description="Max visitors to load (None = all)"
    )
    
    # Connection pool
    pool_min_conn: int = Field(default=1, description="Min pool connections")
    pool_max_conn: int = Field(default=10, description="Max pool connections")
    
    @field_validator("visitor_limit", mode="before")
    @classmethod
    def parse_visitor_limit(cls, v):
        """Convert 0 or empty string to None."""
        if v in (0, "0", "", None):
            return None
        return int(v) if v else None


class HNSWSettings(BaseSettings):
    """HNSW index configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="HNSW_",
        extra="ignore",
    )
    
    # Index parameters
    dimension: int = Field(
        default=128,
        description="Feature vector dimension"
    )
    m: int = Field(
        default=32,
        description="Number of bi-directional links (higher = better recall)"
    )
    ef_construction: int = Field(
        default=400,
        description="Size of dynamic candidate list during construction"
    )
    ef_search: int = Field(
        default=400,
        description="Number of nearest neighbors to explore during search"
    )
    max_elements: int = Field(
        default=100000,
        description="Maximum number of vectors in index"
    )
    
    # File paths
    index_dir: Optional[str] = Field(
        default=None,
        description="Directory to store index files (defaults to models_path)"
    )
    index_file: str = Field(
        default="hnsw_visitor_index.bin",
        description="HNSW index filename"
    )
    metadata_file: str = Field(
        default="hnsw_visitor_metadata.pkl",
        description="HNSW metadata filename"
    )


class ImageSettings(BaseSettings):
    """Image processing configuration."""
    
    model_config = SettingsConfigDict(extra="ignore")
    
    max_width: int = Field(
        default=1920,
        description="Maximum image width"
    )
    max_height: int = Field(
        default=1920,
        description="Maximum image height"
    )
    allowed_formats: Tuple[str, ...] = Field(
        default=("jpg", "jpeg", "png", "webp", "bmp"),
        description="Allowed image formats"
    )
    
    @property
    def max_size(self) -> Tuple[int, int]:
        """Return max size as (width, height) tuple."""
        return (self.max_width, self.max_height)
    
    @property
    def allowed_formats_set(self) -> frozenset:
        """Return allowed formats as frozenset for fast lookup."""
        return frozenset(self.allowed_formats)


class LoggingSettings(BaseSettings):
    """Logging configuration."""
    
    model_config = SettingsConfigDict(extra="ignore")
    
    log_level: str = Field(
        default="INFO",
        description="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    log_file: Optional[str] = Field(
        default=None,
        description="Optional log file path"
    )
    log_format: str = Field(
        default="[%(levelname)s] %(message)s",
        description="Log message format"
    )


class Settings(BaseSettings):
    """
    Main settings class that combines all configuration sections.
    
    Usage:
        from core.config import settings
        
        # Access nested settings
        print(settings.api.port)
        print(settings.database.table_name)
        print(settings.models.sface_similarity_threshold)
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Sub-settings
    api: APISettings = Field(default_factory=APISettings)
    cors: CORSSettings = Field(default_factory=CORSSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    hnsw: HNSWSettings = Field(default_factory=HNSWSettings)
    image: ImageSettings = Field(default_factory=ImageSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    
    # Convenience properties for backward compatibility
    @property
    def models_path(self) -> str:
        return self.models.models_path
    
    @property
    def use_database(self) -> bool:
        return self.database.use_database
    
    @property
    def db_table_name(self) -> str:
        return self.database.table_name
    
    @property
    def score_threshold(self) -> float:
        return self.models.yunet_score_threshold
    
    @property
    def compare_threshold(self) -> float:
        return self.models.sface_similarity_threshold
    
    @property
    def max_image_size(self) -> Tuple[int, int]:
        return self.image.max_size
    
    @property
    def cors_origins(self) -> List[str]:
        return self.cors.origins_list


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()


# Default settings instance
settings = get_settings()


# =============================================================================
# ENVIRONMENT VARIABLE REFERENCE
# =============================================================================
"""
Environment Variables Reference:

API Settings:
    API_HOST            - API server host (default: 0.0.0.0)
    API_PORT            - API server port (default: 8000)
    API_DEBUG           - Enable debug mode (default: false)

CORS Settings:
    CORS_ORIGINS        - Comma-separated allowed origins (default: *)

Model Settings:
    MODELS_PATH         - Path to ONNX models (default: models)
    YUNET_SCORE_THRESHOLD - Detection threshold (default: 0.7)
    SFACE_SIMILARITY_THRESHOLD - Match threshold (default: 0.55)

Database Settings:
    USE_DATABASE        - Enable database (default: false)
    DATABASE_URL        - Full connection URL (overrides individual settings)
    DB_HOST             - Database host (default: localhost)
    DB_PORT             - Database port (default: 5432)
    DB_NAME             - Database name (default: visitors_db)
    DB_USER             - Database user (default: postgres)
    DB_PASSWORD         - Database password (default: "")
    DB_TABLE_NAME       - Visitor table name (default: public."Visitor")
    DB_VISITOR_ID_COLUMN - ID column (default: id)
    DB_IMAGE_COLUMN     - Image column (default: base64Image)
    DB_FEATURES_COLUMN  - Features column (default: facefeatures)
    DB_VISITOR_LIMIT    - Max visitors to load (default: 0 = all)

HNSW Settings:
    HNSW_MAX_ELEMENTS   - Max vectors in index (default: 100000)
    HNSW_M              - Bi-directional links (default: 32)
    HNSW_EF_CONSTRUCTION - Construction candidate list size (default: 400)
    HNSW_EF_SEARCH      - Search candidate list size (default: 400)
    HNSW_INDEX_DIR      - Index directory (default: same as MODELS_PATH)
    HNSW_INDEX_FILE     - Index filename (default: hnsw_visitor_index.bin)
    HNSW_METADATA_FILE  - Metadata filename (default: hnsw_visitor_metadata.pkl)

Logging Settings:
    LOG_LEVEL           - Log level (default: INFO)
    LOG_FILE            - Optional log file path
"""


__all__ = [
    "settings",
    "get_settings",
    "Settings",
    "APISettings",
    "CORSSettings",
    "ModelSettings",
    "DatabaseSettings",
    "HNSWSettings",
    "ImageSettings",
    "LoggingSettings",
]
