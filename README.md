# Face Recognition ML Microservice

A modular FastAPI backend for face detection and recognition using ONNX YuNet (detection) and SFace (recognition) models. Integrates with PostgreSQL for persistent visitor storage, leverages HNSWlib for fast nearest neighbor search, and offers both REST and WebSocket endpoints for real-time image processing.

---

## 🚀 Quick Start

### Prerequisites

- **Docker & Docker Compose**
- **Model files** in `app/models/`:
  - `face_detection_yunet_2023mar.onnx`
  - `face_recognition_sface_2021dec.onnx`
- **PostgreSQL** (optional; if not set up, local test_images/ directory is used by default for visitor features)

### Run via Docker (Recommended)

```bash
docker compose up --build
```
- Access API: http://localhost:8000
- API docs: http://localhost:8000/docs
- OpenAPI: http://localhost:8000/openapi.json

### Local Development

1. **Create Python virtualenv:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   cd services/face-recognition
   pip install -r requirements.txt
   ```

3. **Download models (if missing):**
   ```bash
   python app/ml/download_models.py
   ```

4. **Environment variables (.env example):**
   ```env
   MODELS_PATH=app/models
   CORS_ORIGINS=http://localhost:3000
   USE_DATABASE=true
   DATABASE_URL=postgresql://postgres:postgres@localhost:5432/visitors_db
   ```
   *(Set `USE_DATABASE=false` to use local files instead of Postgres)*

5. **Start API:**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

---

## 📚 API Reference

### Health & Status
- `GET /api/v1/health` — Health check
- `GET /api/v1/models/status`, `/api/v1/models/info` — Model status and info
- `GET /api/v1/hnsw/status` — ANN index info

### Face Detection & Recognition
- `POST /api/v1/detect` — Detect faces (`{"image": base64}` or file upload)
- `POST /api/v1/recognize` — Find closest match (`{"image": base64}`, file upload)
- `POST /api/v1/compare` — Compare two images (`{"image1": ..., "image2": ...}`)
- `POST /api/v1/extract-features` — Feature vector for given image
- `POST /api/v1/validate-image` — Validate image input
- `WebSocket /ws/realtime` — Bi-directional real-time detection/recognition

---

## ⚙️ Configuration

### Key Environment Variables

- `MODELS_PATH`: Path to ONNX models (default `/app/app/models`)
- `CORS_ORIGINS`: Allowed origins (comma-separated)
- `USE_DATABASE`: Use Postgres for visitor storage (`true`/`false`)
- `DATABASE_URL` (or DB_* variables)
- `YUNET_SCORE_THRESHOLD`, `SFACE_SIMILARITY_THRESHOLD`: Detection/recognition thresholds

### Example cURL Usage

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Face detection
curl -X POST http://localhost:8000/api/v1/detect \
  -H "Content-Type: application/json" \
  -d '{"image":"..."}'

# Face recognition
curl -X POST http://localhost:8000/api/v1/recognize \
  -H "Content-Type: application/json" \
  -d '{"image":"..."}'

# Compare faces
curl -X POST http://localhost:8000/api/v1/compare \
  -H "Content-Type: application/json" \
  -d '{"image1":"...", "image2":"..."}'
```

---

## 🗄️ Database Integration

- Uses a `visitors` table:
  - `id`: PK
  - `base64Image`: (required)
  - `facefeatures`: (optional, vector as JSON array)
- If no DB config is provided, falls back to `test_images/` for demo/testing.

**Batch feature extraction:**
```bash
python app/ml/extract_features_to_db.py \
  --db-host ... --db-name ... --db-user ... --db-password ... --image-dir test_images --batch-size 10
```

---

## 🔍 Approximate Nearest Neighbor (HNSW)

- HNSW index loads all features for fast search.
- Persists across restarts; falls back to linear search if index is unavailable.

Check:
```bash
curl http://localhost:8000/api/v1/hnsw/status
```

---

## 📦 File Structure

```
services/face-recognition/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── face_recog_api.py               # [Legacy, being migrated]
│   ├── api/
│   ├── ml/
│   ├── db/
│   ├── pipelines/
│   ├── core/
│   └── models/
│       ├── face_detection_yunet_2023mar.onnx
│       └── face_recognition_sface_2021dec.onnx
├── test_images/
├── Dockerfile
├── requirements.txt
├── README.md
```

---

## 🐳 Docker Usage

```bash
docker compose build
docker compose up
docker compose logs -f
docker compose down
docker compose exec face-recognition-api python app/ml/download_models.py
```

---

## 🧪 API Testing

Detect faces:
```bash
curl -X POST http://localhost:8000/api/v1/detect -F "file=@path/to/image.jpg"
```
Recognize face:
```bash
curl -X POST http://localhost:8000/api/v1/recognize -F "file=@path/to/image.jpg"
```

---

## 🐛 Troubleshooting

### Models not found
Ensure `app/models/` contains the ONNX files. Download via:
```bash
docker compose run face-recognition-api python app/ml/download_models.py
```

### Database issues
- Ensure DB is running: `docker compose ps postgres`
- Check `.env` DB settings
- If DB isn't available, service uses `test_images/`

### Port already in use
Adjust `ports:` in `docker-compose.yml` as needed.

### HNSW/index issues
- Check for missing/invalid features
- See `docker compose logs face-recognition-api`

---

## 📄 License

[Your License Here]
