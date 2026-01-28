FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by OpenCV, hnswlib, and SSL
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    build-essential \
    g++ \
    make \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY app/ ./app/

# Ensure models directory exists (models will be mounted as a volume)
RUN mkdir -p ./app/models

# (Uncomment if models should be copied instead of mounted)
# COPY models/ ./app/models/

WORKDIR /app/app

ENV MODELS_PATH=/app/app/models

# Expose both HTTP (8000) and HTTPS (8443) ports
EXPOSE 8000 8443

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

# Default command - can be overridden by docker-compose
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
