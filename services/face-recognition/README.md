# Face Recognition ML Microservice

A modular, containerized FastAPI microservice for face detection and recognition with ONNX YuNet (detection) and SFace (recognition) models.

---

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose installed
- Model files present in the `app/models/` directory:
  - `face_detection_yunet_2023mar.onnx`
  - `face_recognition_sface_2021dec.onnx`

> **Tip:** Run `python app/ml/download_models.py` in the container to download models if missing.

### Run the Service

```bash
# Build and start (foreground)
docker compose up --build

# Start in background
docker compose up -d --build
```

The API is served at **http://localhost:8000** by default.

### View API Documentation

- **Swagger (interactive):** http://localhost:8000/docs
- **OpenAPI Schema:** http://localhost:8000/openapi.json

---

## 📚 API Reference

### Health

- `GET /health` - Basic health
- `GET /api/v1/health` - API versioned health

### Face Detection

- `POST /api/v1/detect` - Detect faces from base64 JSON image
- `POST /detect` - Detect faces (multipart or base64)

### Face Recognition

- `POST /api/v1/compare` - Compare two faces (JSON, base64)
- `POST /compare` - Compare faces (multipart or base64)

### Feature Extraction

- `POST /extract-features` - Extract face feature vector

### Model Info

- `GET /models/status` - Model load status
- `GET /models/info` - ONNX model metadata

---

## ⚙️ Configuration

### Environment Variables

- `MODELS_PATH` — Path to model directory (`/app/app/models` by default)
- `CORS_ORIGINS` — CSV of allowed CORS origins (default: `*`)

### Example cURL Usage

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Base64 detection
curl -X POST http://localhost:8000/api/v1/detect \
  -H "Content-Type: application/json" \
  -d '{
    "image": "base64_image_data",
    "score_threshold": 0.6
  }'
```

---

## 🐳 Common Docker Commands

```bash
docker compose build               # Build image
docker compose up                  # Start service (foreground)
docker compose up -d               # Start in background
docker compose logs -f             # View logs
docker compose down                # Stop and remove
docker compose up --build          # Rebuild and restart
```

---

## 🗂️ Project Structure

<details>
<summary>Directory Overview</summary>

```
face-recognition/
├── app/
│   ├── __init__.py
│   ├── main.py                        # Main FastAPI entry point
│   ├── api/                           # API layer (routes, dependencies)
│   ├── core/                          # Config, logging, state management
│   ├── ml/                            # Model loading, inference, HNSW, downloaders
│   ├── db/                            # Database logic (optional)
│   ├── pipelines/                     # High-level ML/detection/recognition
│   ├── schemas/                       # Pydantic models / input schemas
│   ├── utils/                         # Shared helpers
│   ├── models/                        # .onnx model files
├── scripts/                           # CLI tools for features/indexing
├── test_images/                       # Example/test images
├── Dockerfile
├── docker-compose.yml
├── requirements.txt                   # Main dependencies
├── requirements-dev.txt               # Dev/test/lint deps
└── README.md
```
</details>

---

## 🔗 Integration

Use this service with:

- Web/JS frontends (React, Next.js, Svelte, etc.)
- Backend APIs (Python, Node, Go, etc.)
- Mobile apps (iOS, Android)
- Any HTTP client

**Integration Example (JS/TS):**
```js
const resp = await fetch('http://localhost:8000/api/v1/detect', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ image: base64Image, score_threshold: 0.6 })
});
const data = await resp.json();
console.log(`Detected ${data.count} faces.`);
```

---

## 🚢 Deploying to Production

1. **Set CORS origins** in Docker Compose or Env:
    ```yaml
    environment:
      - CORS_ORIGINS=https://your.domain,https://api.your.domain
    ```
2. **Different configs/environments**:
    ```bash
    docker compose -f docker-compose.prod.yml up
    ```
3. **Reverse Proxy (optional):** Use nginx/traefik for SSL & routing.
4. **Scaling:**
    ```bash
    docker compose up --scale face-recognition-api=3
    ```
5. **Cloud:** Deploy on AWS ECS/Fargate, Google Cloud Run, Azure Container Instances, DigitalOcean Apps, or Kubernetes.

---

## 🐛 Troubleshooting

### Models not found

Models must be present before starting:

```bash
ls app/models/
# Expected:
# face_detection_yunet_2023mar.onnx
# face_recognition_sface_2021dec.onnx
```
If missing, run:  
```bash
docker compose run face-recognition-api python app/ml/download_models.py
```

### Address/Port in Use

Adjust in `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"
```

### Health or Startup Errors

Check logs:
```bash
docker compose logs face-recognition-api
```

---

## 📄 License

[Your License Here]
