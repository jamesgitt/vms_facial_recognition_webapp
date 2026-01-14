# Face Detection & Recognition Backend API

Streamlined backend service for YuNet + Sface face detection and recognition.

## Quick Start

### 1. Prerequisites
- Docker and Docker Compose installed
- Model files in `models/` directory:
  - `face_detection_yunet_2023mar.onnx`
  - `face_recognition_sface_2021dec.onnx`

### 2. Deploy with Docker

```bash
# Build and run (starts API + PostgreSQL)
docker-compose up -d

# Check logs
docker-compose logs -f

# Check specific service logs
docker-compose logs -f face-api
docker-compose logs -f postgres

# Stop
docker-compose down

# Stop and remove volumes (clean database)
docker-compose down -v
```

**Note**: PostgreSQL will initialize automatically. Wait 10-15 seconds after startup for database to be ready.

### 3. Test API

```bash
# Health check
curl http://localhost:8000/health

# API docs
open http://localhost:8000/docs
```

## API Endpoints

### Detect Faces
```bash
POST /api/v1/detect
Content-Type: application/json

{
  "image": "base64_encoded_image"
}
```

### Recognize Face
```bash
POST /api/v1/recognize
Content-Type: application/json

{
  "image": "base64_encoded_image",
  "threshold": 0.363
}
```

### Register Visitor
```bash
POST /api/v1/register
Content-Type: application/json

{
  "name": "John Doe",
  "email": "john@example.com",
  "phone": "+1234567890",
  "images": ["base64_image1", "base64_image2"]
}
```

### Get Visitor Info
```bash
GET /api/v1/visitor/{visitor_id}
```

## Deployment Options

### Option 1: Docker (Recommended)
```bash
docker-compose up -d
```

### Option 2: Direct Python
```bash
pip install -r requirements_backend.txt
python backend_api.py
```

### Option 3: Cloud Deployment
- **AWS**: ECS, Lambda, or EC2
- **Azure**: Container Instances or App Service
- **Google Cloud**: Cloud Run or Compute Engine

## Environment Variables

```bash
# Required
API_PORT=8000
MODEL_DIR=/app/models
DATABASE_URL=postgresql://admin:password@postgres:5432/visitors

# PostgreSQL connection (if not using DATABASE_URL)
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=visitors
POSTGRES_USER=admin
POSTGRES_PASSWORD=password
```

## Production Considerations

1. **Change default PostgreSQL password** in docker-compose.yml
2. **Add authentication** (JWT tokens)
3. **Enable HTTPS** (use reverse proxy like Nginx)
4. **Add rate limiting**
5. **Use Redis** for caching
6. **Set up monitoring** (Prometheus, Grafana)
7. **Backup PostgreSQL** regularly
8. **Use connection pooling** (already implemented)
9. **Set up database migrations** (Alembic)
10. **Configure PostgreSQL** for performance (shared_buffers, etc.)

## Scaling

- **Horizontal**: Run multiple containers behind load balancer
- **Vertical**: Increase container resources
- **Caching**: Add Redis for feature caching
- **Database**: Use read replicas

## API Documentation

Interactive API docs available at: `http://localhost:8000/docs`
