# Face Recognition Backend API

A high-performance REST API for face detection and recognition using YuNet (face detection) and Sface (face recognition) ONNX models. Features PostgreSQL database integration, HNSW-based approximate nearest neighbor search for fast recognition, and WebSocket support for real-time processing.

## 🚀 Quick Start

### Prerequisites
- **Python 3.11 or 3.12** (recommended)
- **Docker and Docker Compose** (recommended)
- **PostgreSQL** (optional, falls back to test images if not configured)

### Docker Deployment (Recommended)

**Start the backend service:**
```bash
docker compose up -d backend
```

The API will be available at: **http://localhost:8000**

**Access API documentation:**
- Interactive API Docs: http://localhost:8000/docs
- OpenAPI Schema: http://localhost:8000/openapi.json

### Local Development Setup

1. **Create virtual environment:**
   ```bash
   python -m venv venvback
   ```

2. **Activate virtual environment:**
   - **Windows (PowerShell):** `.\venvback\Scripts\Activate.ps1`
   - **Windows (CMD):** `venvback\Scripts\activate.bat`
   - **Linux/Mac:** `source venvback/bin/activate`

3. **Install dependencies:**
   ```bash
   cd services/face-recognition
   pip install -r requirements.txt
   ```

4. **Download models:**
   ```bash
   python app/download_models.py
   ```
   This downloads:
   - `face_detection_yunet_2023mar.onnx` (YuNet face detector)
   - `face_recognition_sface_2021dec.onnx` (Sface face recognizer)

5. **Set environment variables:**
   Create a `.env` file in `services/face-recognition/`:
   ```env
   MODELS_PATH=app/models
   CORS_ORIGINS=http://localhost:3000,http://localhost:3001
   USE_DATABASE=true
   DATABASE_URL=postgresql://postgres:postgres@localhost:5432/visitors_db
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=visitors_db
   DB_USER=postgres
   DB_PASSWORD=postgres
   DB_TABLE_NAME=visitors
   ```

6. **Run the API:**
   ```bash
   cd app
   uvicorn face_recog_api:app --host 0.0.0.0 --port 8000 --reload
   ```

## 📋 API Endpoints

### Health & Status
- `GET /api/v1/health` - Health check endpoint
- `GET /api/v1/models/status` - Check if models are loaded
- `GET /api/v1/models/info` - Get model metadata and information
- `GET /api/v1/hnsw/status` - Get HNSW index status (if enabled)

### Face Detection
- `POST /api/v1/detect` - Detect faces in an image
  - **Input:** JSON with `image` (base64) or multipart form with `file`
  - **Output:** List of detected faces with bounding boxes, landmarks, and confidence scores

### Face Recognition
- `POST /api/v1/recognize` - Recognize a face against database visitors
  - **Input:** JSON with `image` (base64) or multipart form with `file`
  - **Output:** Matched visitor ID, confidence score, and top matches
  - **Features:** Uses HNSW index for fast approximate nearest neighbor search

### Face Comparison
- `POST /api/v1/compare` - Compare two faces
  - **Input:** JSON with `image1` and `image2` (base64)
  - **Output:** Similarity score and match verdict

### Feature Extraction
- `POST /api/v1/extract-features` - Extract face feature vectors
  - **Input:** JSON with `image` (base64)
  - **Output:** 128-dimensional feature vector

### Image Validation
- `POST /api/v1/validate-image` - Validate image before processing
  - **Input:** JSON with `image` (base64) or multipart form with `file`
  - **Output:** Validation result with format, size, and data checks

### Real-time Processing
- `WebSocket /ws/realtime` - Real-time face detection/recognition
  - **Protocol:** Send base64 images, receive detection/recognition results
  - **Use cases:** Live camera feeds, video processing

## 🔧 Configuration

### Environment Variables

#### Core Settings
- `MODELS_PATH` - Path to models directory (default: `/app/app/models` in Docker, `app/models` locally)
- `CORS_ORIGINS` - Comma-separated list of allowed CORS origins (default: `*`)

#### Database Configuration
- `USE_DATABASE` - Enable database integration (`true`/`false`, default: `false`)
- `DATABASE_URL` - PostgreSQL connection string
- `DB_HOST` - Database host (default: `localhost`)
- `DB_PORT` - Database port (default: `5432`)
- `DB_NAME` - Database name (default: `visitors_db`)
- `DB_USER` - Database user (default: `postgres`)
- `DB_PASSWORD` - Database password (default: `postgres`)
- `DB_TABLE_NAME` - Table name for visitors (default: `visitors`)
- `DB_VISITOR_ID_COLUMN` - Column name for visitor ID (default: `id`)
- `DB_IMAGE_COLUMN` - Column name for base64 image data (default: `base64Image`)
- `DB_FEATURES_COLUMN` - Column name for pre-extracted features (default: `facefeatures`)
- `DB_VISITOR_LIMIT` - Limit number of visitors loaded (default: `0` = no limit)

#### Model Parameters
- `YUNET_SCORE_THRESHOLD` - Face detection confidence threshold (default: `0.6`)
- `SFACE_SIMILARITY_THRESHOLD` - Face recognition similarity threshold (default: `0.55`)

### Example API Request

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Detect faces
curl -X POST http://localhost:8000/api/v1/detect \
  -H "Content-Type: application/json" \
  -d '{
    "image": "base64_encoded_image_here",
    "score_threshold": 0.6
  }'

# Recognize face
curl -X POST http://localhost:8000/api/v1/recognize \
  -H "Content-Type: application/json" \
  -d '{
    "image": "base64_encoded_image_here",
    "threshold": 0.55
  }'

# Compare two faces
curl -X POST http://localhost:8000/api/v1/compare \
  -H "Content-Type: application/json" \
  -d '{
    "image1": "base64_image1_here",
    "image2": "base64_image2_here"
  }'
```

## 🗄️ Database Integration

The backend supports PostgreSQL for persistent visitor storage. If the database is not configured or unavailable, the service automatically falls back to loading visitors from the `test_images/` directory.

### Database Schema

The service expects a `visitors` table with the following columns:
- `id` - Visitor ID (primary key)
- `base64Image` - Base64-encoded image data
- `facefeatures` - Optional: Pre-extracted 128-dimensional feature vector (JSON array)

### Feature Extraction Script

Extract and store face features in the database for faster recognition:

```bash
python app/extract_features_to_db.py \
  --db-host localhost \
  --db-name visitors_db \
  --db-user postgres \
  --db-password postgres \
  --image-dir test_images \
  --batch-size 10
```

## 🔍 HNSW Index

The backend uses HNSW (Hierarchical Navigable Small World) for fast approximate nearest neighbor search, enabling efficient recognition across large visitor databases.

### Features
- **Fast Recognition:** O(log n) search complexity vs O(n) linear search
- **Automatic Indexing:** Index is built automatically on startup from database visitors
- **Persistent Storage:** Index is saved to disk and reloaded on restart
- **Fallback Support:** Automatically falls back to linear search if HNSW is unavailable

### Index Status

Check index status:
```bash
curl http://localhost:8000/api/v1/hnsw/status
```

## 📦 Project Structure

```
services/face-recognition/
├── app/
│   ├── face_recog_api.py      # FastAPI application and endpoints
│   ├── inference.py            # ML inference (detection & recognition)
│   ├── database.py             # PostgreSQL database operations
│   ├── hnsw_index.py           # HNSW index manager
│   ├── image_loader.py         # Image loading and validation utilities
│   ├── download_models.py      # Model downloader script
│   ├── extract_features_to_db.py  # Feature extraction script
│   └── models/                 # ONNX model files
│       ├── face_detection_yunet_2023mar.onnx
│       └── face_recognition_sface_2021dec.onnx
├── test_images/                # Test images for fallback recognition
├── Dockerfile                  # Container definition
├── requirements.txt            # Python dependencies
└── README.md                   # Service documentation
```

## 🐳 Docker Commands

```bash
# Build backend image
docker compose build backend

# Start backend service
docker compose up -d backend

# View logs
docker compose logs -f backend

# Stop service
docker compose down backend

# Rebuild after code changes
docker compose up --build backend

# Execute commands in container
docker compose exec backend python app/download_models.py
```

## 🧪 Testing

### Test Face Detection
```bash
curl -X POST http://localhost:8000/api/v1/detect \
  -F "file=@path/to/image.jpg"
```

### Test Face Recognition
```bash
curl -X POST http://localhost:8000/api/v1/recognize \
  -F "file=@path/to/image.jpg"
```

### Test Health Endpoint
```bash
curl http://localhost:8000/api/v1/health
```

## 🐛 Troubleshooting

### Models Not Found
Ensure models are in the correct directory:
```bash
# Check models exist
ls services/face-recognition/app/models/
# Should show:
# face_detection_yunet_2023mar.onnx
# face_recognition_sface_2021dec.onnx

# Download models if missing
cd services/face-recognition
python app/download_models.py
```

### Database Connection Issues
- Check PostgreSQL is running: `docker compose ps postgres`
- Verify connection string in environment variables
- Service will automatically fall back to `test_images/` if database is unavailable

### Port Already in Use
Change port mapping in `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"  # Use port 8001 instead
```

### HNSW Index Not Building
- Check database has visitors with valid images
- Review logs: `docker compose logs backend`
- Index building happens synchronously on startup (may take several minutes for large databases)

## 📚 Dependencies

The following are the main dependencies. For the full list and exact versions, see `services/face-recognition/requirements.txt`.

### Core Libraries
- `fastapi` - Modern web framework for building APIs
- `uvicorn` - ASGI server
- `opencv-python` - Computer vision and image processing
- `numpy` - Numerical computing
- `Pillow` - Image processing

### Machine Learning
- `hnswlib` - HNSW approximate nearest neighbor search

### Database
- `psycopg2-binary` - PostgreSQL adapter

### Utilities
- `pydantic` - Data validation
- `python-dotenv` - Environment variable management
- `websockets` - WebSocket support

> **Note:** If your use case requires DeepFace or TensorFlow, you must install them manually, as they are no longer part of the requirements.txt by default.

## 🚢 Production Deployment

### Considerations
1. **Set proper CORS origins** in environment variables
2. **Use environment-specific database credentials**
3. **Enable HNSW index** for large visitor databases
4. **Configure reverse proxy** (nginx/traefik) for SSL termination
5. **Set up monitoring** and health checks
6. **Scale horizontally** using Docker Compose or Kubernetes

### Cloud Deployment Options
- **AWS ECS/Fargate**
- **Google Cloud Run**
- **Azure Container Instances**
- **DigitalOcean App Platform**
- **Kubernetes**

## 📝 License

[Your License Here]
