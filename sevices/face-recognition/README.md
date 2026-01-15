# Face Recognition ML Microservice

A standalone Dockerized microservice for face detection and recognition using YuNet (detection) and Sface (recognition) ONNX models.

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Models downloaded in `models/` directory:
  - `face_detection_yunet_2023mar.onnx`
  - `face_recognition_sface_2021dec.onnx`

### Start the Service

```bash
# Build and start
docker compose up --build

# Or run in detached mode
docker compose up -d --build
```

The API will be available at: **http://localhost:8000**

### API Documentation

Once running, access:
- **Interactive API Docs**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## 📋 API Endpoints

### Health Check
- `GET /health` - Basic health check
- `GET /api/v1/health` - Versioned health check

### Face Detection
- `POST /api/v1/detect` - Detect faces in an image (JSON with base64)
- `POST /detect` - Detect faces (multipart form or base64)

### Face Recognition
- `POST /api/v1/compare` - Compare two faces (JSON with base64 images)
- `POST /compare` - Compare faces (multipart form or base64)

### Feature Extraction
- `POST /extract-features` - Extract face feature vectors

### Model Information
- `GET /models/status` - Check if models are loaded
- `GET /models/info` - Get model metadata

## 🔧 Configuration

### Environment Variables

- `MODELS_PATH` - Path to models directory (default: `/app/app/models`)
- `CORS_ORIGINS` - Comma-separated list of allowed CORS origins (default: `*`)

### Example Request

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Detect faces (using base64 image)
curl -X POST http://localhost:8000/api/v1/detect \
  -H "Content-Type: application/json" \
  -d '{
    "image": "base64_encoded_image_here",
    "score_threshold": 0.6
  }'
```

## 🐳 Docker Commands

```bash
# Build image
docker compose build

# Start service
docker compose up

# Start in background
docker compose up -d

# View logs
docker compose logs -f

# Stop service
docker compose down

# Rebuild after code changes
docker compose up --build
```

## 📦 Project Structure

```
face-recognition/
├── app/
│   ├── face_recog_api.py    # FastAPI application
│   ├── inference.py          # ML inference logic
│   ├── download_models.py    # Model downloader
│   └── model_testing.py      # Testing utilities
├── models/                   # ONNX model files
├── Dockerfile                # Container definition
├── docker-compose.yml        # Service orchestration
├── requirements-api.txt      # Python dependencies
└── README.md                 # This file
```

## 🔗 Integration

This microservice can be integrated with:
- Frontend applications (React, Next.js, etc.)
- Other backend services
- Mobile applications
- Any HTTP client

### Example Frontend Integration

```javascript
const response = await fetch('http://localhost:8000/api/v1/detect', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    image: base64Image,
    score_threshold: 0.6
  })
});

const data = await response.json();
console.log(`Detected ${data.count} faces`);
```

## 🚢 Deployment

### Production Considerations

1. **Update CORS origins** in `docker-compose.yml`:
   ```yaml
   environment:
     - CORS_ORIGINS=https://yourdomain.com,https://api.yourdomain.com
   ```

2. **Use environment-specific configs**:
   ```bash
   docker compose -f docker-compose.prod.yml up
   ```

3. **Add reverse proxy** (nginx/traefik) for SSL termination

4. **Scale horizontally**:
   ```bash
   docker compose up --scale face-recognition-api=3
   ```

### Cloud Deployment Options

- **AWS ECS/Fargate**
- **Google Cloud Run**
- **Azure Container Instances**
- **DigitalOcean App Platform**
- **Kubernetes**

## 🐛 Troubleshooting

### Models not found
Ensure models are in `models/` directory before building:
```bash
ls models/
# Should show:
# face_detection_yunet_2023mar.onnx
# face_recognition_sface_2021dec.onnx
```

### Port already in use
Change port mapping in `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"  # Use port 8001 instead
```

### Health check failing
Check logs:
```bash
docker compose logs face-recognition-api
```

## 📝 License

[Your License Here]
