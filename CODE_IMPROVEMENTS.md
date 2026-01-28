# Code Improvement Checklist

> **Overall Rating: 7.5/10** (Good)  
> The codebase demonstrates solid engineering practices with room for improvement in certain areas.

---

## Summary Ratings

| Category | Score | Notes |
|----------|-------|-------|
| **Code Organization** | 8/10 | Good module separation, clear file responsibilities |
| **Documentation** | 7/10 | Docstrings present, but inconsistent detail level |
| **Error Handling** | 7/10 | Basic try/except, could use more specific exceptions |
| **Type Hints** | 6/10 | Partial coverage, missing in some functions |
| **Testing** | 4/10 | No unit tests visible |
| **Configuration** | 8/10 | Environment-based, flexible |
| **Code Duplication** | 6/10 | Some repeated patterns |
| **Naming Conventions** | 8/10 | Clear, consistent naming |
| **Separation of Concerns** | 7/10 | Some mixing of responsibilities |

---

## 1. Architecture & Structure

### `face_recog_api.py` (836 lines) - Split into modules

| Priority | Task | Current State | Improvement |
|----------|------|---------------|-------------|
| HIGH | Split file into services | All logic in one 836-line file | **Avoid creating an unnecessary nested `services/services` folder. Since your `app/` folder is already *inside* a `services/` folder, just use `app/` for all submodules, rather than creating `app/services/`.** |
| HIGH | Extract Pydantic schemas | Lines 369-427 mixed with routes | Move to `schemas.py` |
| HIGH | Extract config | Lines 67-93 scattered constants | Create `config.py` with Pydantic `BaseSettings` |
| MED | Extract startup logic | `load_models()` does too much (lines 319-362) | Create `startup.py` |

**Proposed structure (if `app/` is already inside `services//`):**
```
services/
└── app/
    ├── face_recog_api.py      # Just routes (slim)
    ├── config.py              # All configuration
    ├── schemas.py             # Pydantic models
    ├── startup.py             # Initialization logic
    ├── detection.py           # detect_faces_api logic
    ├── recognition.py         # recognize_visitor_api logic
    └── comparison.py          # compare_faces_api logic
```
> ⚠️ *Don't add an additional `services/` folder inside `app/`, or you'll end up with `services/app/services/` which is redundant.*

---

## 2. Global State Management

### `face_recog_api.py` lines 99-102

| Priority | Issue | Location | Fix |
|----------|-------|----------|-----|
| HIGH | Mutable global state | Lines 99-102 | Use dependency injection or singleton class |

**Current (problematic):**
```python
face_detector = None
face_recognizer = None
hnsw_index_manager = None
VISITOR_FEATURES = {}
```

**Proposed fix - create `state.py` inside `app/`:**
```python
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class AppState:
    face_detector: Optional[Any] = None
    face_recognizer: Optional[Any] = None
    hnsw_manager: Optional[Any] = None
    visitor_features: Dict[str, Dict] = field(default_factory=dict)
    use_database: bool = False

app_state = AppState()
```

---

## 3. Configuration Management

### Create `config.py` inside `app/`

| Priority | Issue | Location | Fix |
|----------|-------|----------|-----|
| HIGH | Config scattered across files | `face_recog_api.py:67-93`, `database.py:32-49`, `hnsw_index.py:25-31` | Centralize in one Pydantic settings class |
| MED | No validation | Raw `os.environ.get()` | Use Pydantic validators |

**Proposed `config.py`:**
```python
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Paths
    models_path: str = "models"
    visitor_images_dir: str = "../test_images"
    
    # Thresholds
    score_threshold: float = 0.7
    compare_threshold: float = 0.55
    
    # Image limits
    max_image_width: int = 1920
    max_image_height: int = 1920
    
    # Database
    use_database: bool = False
    database_url: Optional[str] = None
    db_table_name: str = 'public."Visitor"'
    db_visitor_id_column: str = "id"
    db_image_column: str = "base64Image"
    db_features_column: str = "facefeatures"
    db_visitor_limit: Optional[int] = None
    
    # HNSW
    hnsw_max_elements: int = 100000
    hnsw_m: int = 32
    hnsw_ef_construction: int = 400
    hnsw_ef_search: int = 400
    
    # CORS
    cors_origins: str = "*"
    
    class Config:
        env_file = ".env"

settings = Settings()
```

---

## 4. Error Handling

### Multiple files

| Priority | Issue | Location | Fix |
|----------|-------|----------|-----|
| HIGH | Generic exceptions | `face_recog_api.py:175-176` silent `except Exception: pass` | Log and handle properly |
| HIGH | No custom exceptions | All files use built-in exceptions | Create `exceptions.py` |
| MED | Inconsistent error responses | Various endpoints return different formats | Standardize error response schema |

**Create `exceptions.py` inside `app/`:**
```python
class FaceRecognitionError(Exception):
    """Base exception for face recognition errors."""
    pass

class NoFaceDetectedError(FaceRecognitionError):
    """Raised when no face is detected in image."""
    pass

class FeatureExtractionError(FaceRecognitionError):
    """Raised when feature extraction fails."""
    pass

class InvalidImageError(FaceRecognitionError):
    """Raised when image is invalid or unsupported."""
    pass

class DatabaseConnectionError(FaceRecognitionError):
    """Raised when database connection fails."""
    pass

class IndexNotInitializedError(FaceRecognitionError):
    """Raised when HNSW index is not available."""
    pass
```

**Fix silent exception at `face_recog_api.py:175-176`:**
```python
# Current (bad)
except Exception:
    pass

# Better
except Exception as e:
    logger.warning(f"Failed to update features for {visitor_id}: {e}")
```

---

## 5. Logging

### All files

| Priority | Issue | Location | Fix |
|----------|-------|----------|-----|
| HIGH | Using print statements | All files | Replace with `logging` module |
| MED | No structured logging | N/A | Add request IDs, timing |

Affected locations for logging improvements (all print statements replaced with logger, no "etc"):

face_recog_api.py:
- Line 55: Replace print(f"[WARNING] Error decoding features for {visitor_id}: {e}")  
  with logger.warning(f"Error decoding features for {visitor_id}: {e}")
- Line 64: Replace print(f"[ERROR] Feature extraction failed for {visitor_data.get(DB_VISITOR_ID_COLUMN, 'unknown')}: {e}")  
  with logger.error(f"Feature extraction failed for {visitor_data.get(DB_VISITOR_ID_COLUMN, 'unknown')}: {e}")
- Line 145: Replace print(f"Downloading {model_name} from {url}...")  
  with logger.info(f"Downloading {model_name} from {url}...")
- Line 180: Replace print(f"Downloaded {model_name} to {filepath} ({_format_size(downloaded_size)})")  
  with logger.info(f"Downloaded {model_name} to {filepath} ({_format_size(downloaded_size)})")
- Line 212: Replace print(f"Loading environment from: {env_path}")  
  with logger.info(f"Loading environment from: {env_path}")
- Lines 217-228: 
  - Replace print("Warning: No .env.test file found")  
    with logger.warning("No .env.test file found")
  - Replace print(f"Error importing modules: {e}")  
    with logger.error(f"Error importing modules: {e}")
  - Replace print("Make sure you're running from the app directory with all dependencies installed.")  
    with logger.error("Make sure you're running from the app directory with all dependencies installed.")
- Line 242: Replace any print used for logging with an appropriate logger call
- Line 259: Replace any print used for logging with an appropriate logger call
- Line 268: Replace any print used for logging with an appropriate logger call
- Add `from .logger import logger` to the file's imports

hnsw_index.py:
- Line 22: Replace print("HNSW index initialized with dimension ...")  
  with logger.info("HNSW index initialized with dimension ...")
- Line 114: Replace print(f"Visitor {visitor_id} already in index, skipping")  
  with logger.info(f"Visitor {visitor_id} already in index, skipping")
- Line 118: Replace print(f"Error adding visitor to HNSW index: {e}")  
  with logger.error(f"Error adding visitor to HNSW index: {e}")
- Line 140: Replace print(f"[WARNING] Skipping visitor {visitor_id}: dimension {feature.shape[0]} != {self.dimension}")  
  with logger.warning(f"Skipping visitor {visitor_id}: dimension {feature.shape[0]} != {self.dimension}")
- Line 163: Replace print(f"[WARNING] No valid features to add (processed {len(visitors)} visitors)")  
  with logger.warning(f"No valid features to add (processed {len(visitors)} visitors)")
- Line 176: Replace print(f"Adding {len(features_list)} features to HNSW index...")  
  with logger.info(f"Adding {len(features_list)} features to HNSW index...")
- After successful batch add: Replace print(f"[OK] Added {len(features_list)} visitors to HNSW index")  
  with logger.info(f"Added {len(features_list)} visitors to HNSW index")
- On batch add error: Replace print(f"[WARNING] Error batch adding to HNSW index: {e}")  
  with logger.warning(f"Error batch adding to HNSW index: {e}")
- Add `from .logger import logger` to the file's imports

database.py:
- Line 125: Replace print(f"Connected to database at {DB_HOST}:{DB_PORT} ...")  
  with logger.info(f"Connected to database at {DB_HOST}:{DB_PORT} ...")
- Line 175: On error, replace silent exception with  
  except Exception as e: logger.error(f"Database operation failed: {e}")
- Line 178: Replace any manual print errors with logger.error or logger.warning as appropriate
- Line 235: Replace print(f"Query returned {len(results)} visitors")  
  with logger.info(f"Query returned {len(results)} visitors")
- Line 308: Replace print("Closing connection pool...")  
  with logger.info("Closing connection pool...")
- Line 311: Replace print("Connection pool closed.")  
  with logger.info("Connection pool closed.")
- Line 328: On failure to close pool, replace print("Failed to close connection pool: ...")  
  with logger.warning("Failed to close connection pool: ...")
- Add `from .logger import logger` to the file's imports

In all listed files, import the shared logger by adding:
from .logger import logger

Do not create new logger instances in these files. Use the shared logger for all logging.



**Summary of fixes:**
- All print statements → appropriate logger method
- Silently-swallowed exceptions → logger with error/warning
- Add `from .logger import logger` to all affected modules
- Use structured log messages, referencing context (visitor_id, feature shape, etc.) 

**Create `logger.py` in `app/`:**
```python
import logging
import sys

def setup_logger(name: str = "face_recognition") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(levelname)s] %(asctime)s - %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

logger = setup_logger()
```

---

## 6. Type Hints

### Multiple files - incomplete type hints

| Priority | File | Line(s) | Issue |
|----------|------|---------|-------|
| MED | `face_recog_api.py` | 320 | `load_models()` missing return type |
| MED | `face_recog_api.py` | 448 | `health()` missing return type |
| MED | `hnsw_index.py` | 68 | `self.index: Optional[Any]` - use proper type |
| MED | `database.py` | 60 | `get_db_connection()` missing return type annotation |

**Examples to fix:**

```python
# face_recog_api.py:320
def load_models() -> None:

# face_recog_api.py:448  
def health() -> dict:

# database.py:60
def get_db_connection() -> psycopg2.extensions.connection:
```

---

## 7. Code Duplication

### `face_recog_api.py`

| Priority | Issue | Locations | Fix |
|----------|-------|-----------|-----|
| MED | Repeated face detection check | Lines 154, 292, 495, 520, 522, 569, 625 | Create helper function |
| MED | Repeated image loading + validation | Lines 456-457, 491-492, 511-515, 557-562 | Create decorator or helper |

**Create helper functions:**

```python
# Add to face_recog_api.py or new helpers.py

def load_and_validate_image(image_data: str, source_type: str = "base64") -> np.ndarray:
    """Load image and validate size."""
    img_np = image_loader.load_image(image_data, source_type=source_type)
    image_loader.validate_image_size((img_np.shape[1], img_np.shape[0]), MAX_IMAGE_SIZE)
    return img_np

def require_faces(img: np.ndarray, error_message: str = "No face detected") -> np.ndarray:
    """Detect faces or raise HTTPException."""
    faces = inference.detect_faces(img, return_landmarks=True)
    if faces is None or len(faces) == 0:
        raise HTTPException(status_code=400, detail=error_message)
    return faces

def extract_single_feature(img: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Extract feature from first detected face or raise."""
    feature = inference.extract_face_features(img, faces[0])
    if feature is None:
        raise HTTPException(status_code=400, detail="Failed to extract features")
    return feature
```

---

## 8. API Design

### `face_recog_api.py`

| Priority | Issue | Location | Fix |
|----------|-------|----------|-----|
| LOW | Deprecated `on_event` | Line 319 | Use lifespan context manager |
| LOW | Inconsistent endpoint paths | `/validate-image` vs `/api/v1/detect` | Standardize all under `/api/v1/` |
| LOW | Legacy response fields | Lines 399-400 `visitor`, `match_score` | Mark as deprecated in schema |

**Fix deprecated startup event:**
```python
# Current (deprecated in FastAPI)
@app.on_event("startup")
def load_models():

# Better (modern approach)
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_models()
    yield
    # Shutdown (cleanup if needed)

app = FastAPI(lifespan=lifespan)
```

**Fix inconsistent paths:**
```python
# Current
@app.post("/validate-image", ...)

# Better
@app.post("/api/v1/validate-image", ...)
```

---

## 9. HNSW Index

### `hnsw_index.py`

| Priority | Issue | Location | Fix |
|----------|-------|----------|-----|
| MED | No zero-vector check | `_normalize_feature()` line 143-145 | Add divide-by-zero protection |
| LOW | `remove_visitor` doesn't remove from index | Lines 280-292 | Document limitation or implement proper removal |

**Fix normalization:**
```python
# Current
def _normalize_feature(self, feature: np.ndarray) -> np.ndarray:
    return (feature / np.linalg.norm(feature)).astype('float32')

# Better
def _normalize_feature(self, feature: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(feature)
    if norm < 1e-10:
        raise ValueError("Cannot normalize zero or near-zero vector")
    return (feature / norm).astype('float32')
```

---

## 10. Database

### `database.py`

| Priority | Issue | Location | Fix |
|----------|-------|----------|-----|
| MED | Sync operations in async API | All functions | Consider `asyncpg` or thread pool |
| LOW | SQL injection risk in table name | Lines 161-169 | Validate table name format |
| LOW | Connection pool not closed on shutdown | Line 336-341 | Add FastAPI shutdown hook |

**Add table name validation:**
```python
import re

def _validate_table_name(table_name: str) -> str:
    """Validate table name to prevent SQL injection."""
    # Allow: schema."Table" or just table_name
    pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*(\."[a-zA-Z_][a-zA-Z0-9_]*")?$'
    if not re.match(pattern, table_name):
        raise ValueError(f"Invalid table name format: {table_name}")
    return table_name
```

---

## 11. Script Improvements

### `extract_features_to_db.py`

| Priority | Issue | Location | Fix |
|----------|-------|----------|-----|
| MED | Calls old function signature | Lines 195-196 | `get_face_detector()` takes no args now |
| LOW | No batch commit | Individual updates | Add batch commit for performance |

**Fix at lines 195-196:**
```python
# Current (wrong - functions take no args)
inference.get_face_detector(MODELS_PATH)
inference.get_face_recognizer(MODELS_PATH)

# Correct
inference.get_face_detector()
inference.get_face_recognizer()
```

---

## 12. Testing

### Create `tests/` directory

| Priority | Task | Details |
|----------|------|---------|
| HIGH | Unit tests for inference | Test `detect_faces`, `extract_face_features`, `compare_face_features` |
| HIGH | Unit tests for HNSW | Test `add_visitor`, `search`, `rebuild_from_database` |
| MED | Integration tests | Test full `/api/v1/recognize` flow |
| MED | Mock database tests | Test database fallback behavior |

**Proposed test structure:**
```
tests/
├── __init__.py
├── conftest.py              # Pytest fixtures
├── test_inference.py
├── test_hnsw_index.py
├── test_database.py
├── test_image_loader.py
├── test_api_detect.py
├── test_api_recognize.py
└── fixtures/
    └── test_face.jpg
```

---

## Quick Reference - Files to Modify

| File | Changes Needed |
|------|----------------|
| `face_recog_api.py` | Split, fix globals, add helpers, modernize lifespan |
| `hnsw_index.py` | Add zero-vector check, improve logging |
| `database.py` | Add table validation, proper shutdown |
| `inference.py` | Already clean - minor type hint additions |
| `image_loader.py` | Already clean - no major changes |
| `extract_features_to_db.py` | Fix function call signature (line 195-196) |

**New files to create (place in `app/`):**
- `config.py`
- `schemas.py`
- `exceptions.py`
- `logger.py`
- `state.py` (optional)
- `tests/` directory at project root

---

## Strengths (Keep These)

1. **Clear module responsibilities** - Each file has a focused purpose
2. **Good fallback handling** - Graceful degradation (DB → test_images)
3. **Environment-based configuration** - Flexible deployment
4. **HNSW implementation** - Efficient ANN search with L2 normalization
5. **Consistent error prefixes** - `[OK]`, `[WARNING]`, `[ERROR]`
6. **Context managers** - `get_connection()` in database.py

---

## Quick Wins (Low Effort, High Impact)

- [ ] Add `py.typed` marker file for type checking
- [ ] Add `.flake8` / `pyproject.toml` for linting rules
- [ ] Add `requirements-dev.txt` with pytest, black, mypy
- [ ] Fix `extract_features_to_db.py` function signatures
- [ ] Add zero-vector protection in HNSW normalization
