# VMS Facial Recognition System
## Technical Documentation (Updated)

---

**Version:** 1.2  
**Last Updated:** June 2024  
**Maintainer:** VMS Dev Team

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Directory Structure](#3-directory-structure)
4. [Core Python Modules](#4-core-python-modules)
5. [API Reference](#5-api-reference)
6. [Configuration](#6-configuration)
7. [Deployment Workflow](#7-deployment-workflow)
8. [Common Issues & Troubleshooting](#8-common-issues--troubleshooting)
9. [Quick Reference Guide](#9-quick-reference-guide)

---

## 1. Project Overview

### Purpose
VMS Facial Recognition is an end-to-end system for realtime face detection and recognition, entirely managed by local files (no external database). It supports enrollment, recognition, removal, and lookup of identities via high-performance deep learning models and nearest neighbor search—all accessible as a REST API or via WebSocket streaming.

### Core Features
- **Face Detection**: Real-time, robust detection using YuNet (ONNX)
- **Embedding Extraction**: Computes 128-dim face embeddings via SFace model
- **Efficient Matching**: HNSW (hnswlib) for fast k-NN face search
- **Identity Management**: Fully local, file-based (JSON & binary)
- **REST API**: Modern endpoints via FastAPI (interactive docs included)
- **Real-time WebSocket**: Streaming detection for UI/camera integrations

### Tech Stack

| Component            | Technology               | Notes                              |
|----------------------|-------------------------|-------------------------------------|
| API Backend          | FastAPI (Python)        | REST, WebSocket, OpenAPI           |
| Face Detection       | YuNet (ONNX)            | Detection+landmarks                |
| Face Recognition     | SFace (ONNX)            | 128-dim feature vectors            |
| Approximate NN       | hnswlib (C++/Python)    | Extremely fast local search         |
| Storage              | File System             | JSON, PKL, PNG/JPG, binary ANN     |
| Image Processing     | OpenCV, Pillow          | Any format, no DB                   |
| Frontend             | Next.js (React)         | Camera, Web controls (optional)     |

---

## 2. Architecture

### System Block Diagram

```
CLIENTS
│
├─ Web App (Next.js)      ───┐
├─ Mobile/Web Clients       │
├─ IoT/Camera Devices      ──┘
│
▼
─────────────────────────────
API LAYER (FastAPI App)
│   ├─ REST Endpoints (/api/v1/…)
│   ├─ WebSocket (/ws/realtime)
│   └─ CORS/Error Handlers
▼
─────────────────────────────
PROCESSING LAYER
│   ├─ Image Loader  (filesystem, upload, base64)
│   ├─ Inference Engine (YuNet, SFace)
│   └─ HNSW Index Manager
▼
─────────────────────────────
STORAGE LAYER (Local File System)
    ├─ identities.json / {ID}.png
    ├─ hnsw_visitor_index.bin
    └─ hnsw_visitor_metadata.pkl
```

### End-to-end Data Flow

1. Client submits image (base64, file upload)
2. Image Loader checks, decodes, converts as needed
3. YuNet detector finds faces/landmarks
4. SFace generates 128D embeddings per face
5. HNSWIndex finds closest identities (kNN search)
6. API returns matches, metadata, and scores

---

## 3. Directory Structure

```
VMS_Facial_Recognition/
│
├─ services/face-recognition/
│   ├─ app/
│   │   ├─ face_recog_api.py          # FastAPI endpoints & startup
│   │   ├─ inference.py               # Detection/recognition (ONNX)
│   │   ├─ hnsw_index.py              # HNSW manager
│   │   ├─ image_loader.py            # All loading/utilities
│   │   ├─ identity_store.py          # Local identities
│   │   ├─ download_models.py         # Models fetch utility
│   │   └─ models/
│   │       ├─ face_detection_yunet_2023mar.onnx
│   │       └─ face_recognition_sface_2021dec.onnx
│   ├─ test_images/
│   ├─ Dockerfile
│   ├─ requirements.txt
│   ├─ .env.test
│   └─ venvback/
│
├─ apps/facial_recog_web_app/
│   └─ ... (Next.js/React Frontend)
├─ docker-compose.yml
├─ .env
└─ documentation/
```

---

## 4. Core Python Modules

### 4.1 face_recog_api.py

**Entrypoint FastAPI app. Exposes all REST & WS endpoints; loads models and index on startup.**
```python
app = FastAPI(title="Face Recognition API", version="1.2.0")

@app.on_event("startup")
def startup_event():
    # Loads models, identities, and rebuilds local HNSW index.
    pass
```
**Primary Endpoints:**
- `/api/v1/detect`             (POST: Detect faces)
- `/api/v1/extract-features`   (POST: Get embedding vectors)
- `/api/v1/compare`            (POST: Compare two faces)
- `/api/v1/recognize`          (POST: Identify from enrolled identities)
- `/api/v1/identities/enroll`  (POST: New identity)
- `/api/v1/identities/remove`  (POST: Remove identity)
- `/api/v1/identities`         (GET: List all)
- `/api/v1/health`             (GET: Health check)
- `/api/v1/hnsw/status`        (GET: ANN index status)
- `/ws/realtime`               (WS: Streaming detection)

### 4.2 inference.py

**Detection, embedding, and face feature ops (ONNX runtime).**
- `get_face_detector(models_path)`
- `get_face_recognizer(models_path)`
- `detect_faces(image, return_landmarks=True)`
- `extract_face_features(image, face)`
- `compare_faces(feature1, feature2)`
*YuNet accepts 320x320 input; SFace returns 128 floats per face.*

### 4.3 hnsw_index.py

**ANN index manager over all local identities (built via hnswlib).**
- `HNSWIndexManager` class:
    - `add_identity(id, embedding, metadata)`
    - `add_identities_batch(list)`
    - `search(feature_vec, k=5)`
    - `rebuild_from_local(identities_file)`
    - `save()`, `load()`
- Ann index (`hnsw_visitor_index.bin`) & metadata (`hnsw_visitor_metadata.pkl`) stored locally.

### 4.4 identity_store.py

**Local identity registration and management (JSON/PKL).**
- `load_identities(filepath)`
- `save_identities(filepath, identities)`
- `enroll_identity(info, feature, image_path)`
- `remove_identity(identity_id)`
- `list_identities()`

Example record:
```json
{
  "id": "jane-doe-001",
  "firstName": "Jane",
  "lastName": "Doe",
  "imagePath": "data/jane-doe-001.png",
  "features": [0.23, -0.48, ...]
}
```

### 4.5 image_loader.py

**Robust validation/loading (file, upload, base64).**  
Loads only from local files or directly uploaded images.

---

## 5. API Reference

### Health Check

`GET /api/v1/health`

Response:
```json
{ "status": "healthy", "timestamp": "2024-06-08T09:20:30Z" }
```

---

### Face Detection

`POST /api/v1/detect` (`multipart/form-data`)
- image: <file>
- threshold: float (optional)

Sample Response:
```json
{
  "faces": [
    [x, y, width, height, landmark1_x, landmark1_y, ..., confidence]
  ],
  "count": 1
}
```

---

### Feature Extraction

`POST /api/v1/extract-features` (`multipart/form-data`)
- image: <file>

Sample Response:
```json
{
  "features": [[0.123, -0.456, ...]],
  "num_faces": 1
}
```

---

### Face Comparison

`POST /api/v1/compare` (`multipart/form-data`)
- image1: <file>
- image2: <file>
- threshold: float (optional)

Response:
```json
{ "similarity_score": 0.85, "is_match": true }
```

---

### Face Recognition

`POST /api/v1/recognize` (`multipart/form-data`)
- image: <file>
- threshold: float (optional)

Response:
```json
{
  "identity_id": "john-doe-123",
  "confidence": 0.85,
  "matched": true,
  "firstName": "John",
  "lastName": "Doe",
  "matches": [
    {
      "identity_id": "john-doe-123",
      "match_score": 0.85,
      "is_match": true,
      "firstName": "John",
      "lastName": "Doe"
    }
  ]
}
```

---

### Enroll Identity

`POST /api/v1/identities/enroll` (`multipart/form-data`)
- image: <file>
- firstName
- lastName
- id: (optional)

Response:
```json
{ "identity_id": "jane-doe-001", "status": "enrolled" }
```

---

### Remove Identity

`POST /api/v1/identities/remove` (`application/json`)
```json
{ "identity_id": "john-doe-123" }
```

Response:
```json
{ "identity_id": "john-doe-123", "status": "removed" }
```

---

### List All Identities

`GET /api/v1/identities`

Response (truncated):
```json
[
  { "identity_id": "john-doe-123", "firstName": "John", "lastName": "Doe" },
  ...
]
```

---

### Index Status

`GET /api/v1/hnsw/status`

```json
{
  "available": true,
  "initialized": true,
  "total_vectors": 87,
  "dimension": 128,
  "index_type": "HNSW",
  "m": 8,
  "ef_construction": 40,
  "ef_search": 10,
  "identities_indexed": 87
}
```

---

### Realtime Detection via WebSocket

Connect:
```js
const ws = new WebSocket("ws://localhost:8000/ws/realtime");
```
Yields live face box & recognition data streams.

---

## 6. Configuration

### .env.test (Backend)

```env
DATA_DIR=data
IDENTITIES_FILE=identities.json
INDEX_DIR=models
MODELS_PATH=models
CORS_ORIGINS=*
```

### .env (Docker / Frontend)

```env
API_PORT=8000
FRONTEND_PORT=3000
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## 7. Deployment Workflow

### Local

```bash
cd services/face-recognition
python -m venv venv
source venv/bin/activate  # or .\venvback\Scripts\Activate (Win)
pip install -r requirements.txt
cd app
uvicorn face_recog_api:app --reload --port 8000
```
Frontend:
```bash
cd apps/facial_recog_web_app
pnpm install
pnpm dev
```

### Docker

```bash
docker compose up -d
docker compose logs -f backend
docker compose down
```

---

## 8. Common Issues & Troubleshooting

| Issue               | Typical Cause                 | Resolution                         |
|---------------------|------------------------------|-------------------------------------|
| No face found       | Poor image quality/angle      | Better lighting/position            |
| Model not found     | Missing ONNX model files      | Run `download_models.py`            |
| Index missing       | Not enrolled any identities   | Call `/identities/enroll` endpoint  |
| Low similarity      | Different people              | No action (thresholds OK)           |
| Memory error        | Too many identities loaded    | Reduce count, optimize RAM usage    |

#### Tuning HNSW

| Parameter      | Lower Value (Faster) | Higher Value (More Accurate) |
|----------------|----------------------|------------------------------|
| ef_search      | Less accurate        | More accurate, slower        |
| ef_construction| Quicker index build  | Higher-quality index         |
| M              | Less memory          | More connections/memory      |

#### View Logs

```bash
docker logs facial_recog_backend -f
python face_recog_api.py 2>&1 | tee app.log
```

---

## 9. Quick Reference Guide

### Startup (Local)

```bash
cd services/face-recognition
source venv/bin/activate           # Linux
.\venvback\Scripts\Activate        # Windows
cd app
uvicorn face_recog_api:app --host 0.0.0.0 --port 8000
```

### Quick Test

```bash
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/hnsw/status
curl -F "image=@photo.jpg" http://localhost:8000/api/v1/recognize
```

### Interactive Documentation

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

*End of Updated Documentation*
