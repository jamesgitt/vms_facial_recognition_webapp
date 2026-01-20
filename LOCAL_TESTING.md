# Local Testing Commands

Quick commands to test the backend API locally without Docker.

## Setup (One-time)

```powershell
# Navigate to backend directory
cd services/face-recognition/app

# Activate virtual environment (if using venv)
..\..\..\venv\Scripts\Activate.ps1

# Or if venv is in services/face-recognition:
# ..\..\venv\Scripts\Activate.ps1

# Install dependencies (if not already installed)
pip install -r ..\requirements.txt
```

## Environment Variables

Create a `.env` file in `services/face-recognition/` or set environment variables:

```powershell
# Set environment variables for local testing
$env:USE_DATABASE = "true"
$env:DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/visitors_db"
$env:DB_HOST = "localhost"
$env:DB_PORT = "5432"
$env:DB_NAME = "visitors_db"
$env:DB_USER = "postgres"
$env:DB_PASSWORD = "postgres"
$env:DB_TABLE_NAME = "visitors"
$env:DB_VISITOR_ID_COLUMN = "id"
$env:DB_IMAGE_COLUMN = "base64Image"
$env:MODELS_PATH = "models"  # Relative to app directory
$env:CORS_ORIGINS = "http://localhost:3000,http://localhost:3001"
```

## Run Backend Server

```powershell
# Navigate to app directory
cd services/face-recognition/app

# Run with auto-reload (development mode)
python main.py --reload --host 127.0.0.1 --port 8000

# Or run directly
python -m uvicorn face_recog_api:app --host 127.0.0.1 --port 8000 --reload
```

## Test Endpoints

### Health Check
```powershell
# PowerShell
Invoke-WebRequest -Uri "http://localhost:8000/api/v1/health" | Select-Object -ExpandProperty Content

# Or with curl (if available)
curl http://localhost:8000/api/v1/health
```

### HNSW Status
```powershell
# PowerShell
Invoke-WebRequest -Uri "http://localhost:8000/api/v1/hnsw/status" | Select-Object -ExpandProperty Content | ConvertFrom-Json | ConvertTo-Json -Depth 10

# Or prettier output
$response = Invoke-WebRequest -Uri "http://localhost:8000/api/v1/hnsw/status"
$response.Content | ConvertFrom-Json | Format-List
```

### Model Status
```powershell
Invoke-WebRequest -Uri "http://localhost:8000/models/status" | Select-Object -ExpandProperty Content | ConvertFrom-Json | ConvertTo-Json
```

### Face Detection (POST)
```powershell
# You'll need a base64 image - this is just the structure
$body = @{
    image = "base64_encoded_image_here"
    score_threshold = 0.6
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:8000/api/v1/detect" -Method POST -Body $body -ContentType "application/json" | Select-Object -ExpandProperty Content
```

## Monitor Logs

The server will output logs directly to the terminal. Look for:
- `✓ HNSW index manager initialized`
- `Building HNSW index from X database visitors...`
- `Extracted features: X successful, Y failed`
- `✓ Successfully added X visitors to HNSW index`

## Quick Test Script

Save this as `test_local.ps1`:

```powershell
# Quick test script
Write-Host "Testing Backend API..." -ForegroundColor Green

# Health check
Write-Host "`n1. Health Check:" -ForegroundColor Yellow
try {
    $health = Invoke-WebRequest -Uri "http://localhost:8000/api/v1/health" -ErrorAction Stop
    Write-Host "✓ Health check passed" -ForegroundColor Green
    $health.Content
} catch {
    Write-Host "✗ Health check failed: $_" -ForegroundColor Red
}

# HNSW Status
Write-Host "`n2. HNSW Status:" -ForegroundColor Yellow
try {
    $hnsw = Invoke-WebRequest -Uri "http://localhost:8000/api/v1/hnsw/status" -ErrorAction Stop
    $status = $hnsw.Content | ConvertFrom-Json
    Write-Host "✓ HNSW Status retrieved" -ForegroundColor Green
    Write-Host "  Available: $($status.available)"
    Write-Host "  Initialized: $($status.initialized)"
    Write-Host "  Total Vectors: $($status.total_vectors)"
    Write-Host "  Visitors Indexed: $($status.visitors_indexed)"
} catch {
    Write-Host "✗ HNSW status check failed: $_" -ForegroundColor Red
}

# Model Status
Write-Host "`n3. Model Status:" -ForegroundColor Yellow
try {
    $models = Invoke-WebRequest -Uri "http://localhost:8000/models/status" -ErrorAction Stop
    $modelStatus = $models.Content | ConvertFrom-Json
    Write-Host "✓ Model status retrieved" -ForegroundColor Green
    Write-Host "  Loaded: $($modelStatus.loaded)"
} catch {
    Write-Host "✗ Model status check failed: $_" -ForegroundColor Red
}
```

Run it with:
```powershell
.\test_local.ps1
```

## Stop Server

Press `Ctrl+C` in the terminal where the server is running.
