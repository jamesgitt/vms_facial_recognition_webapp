FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_backend.txt .
RUN pip install --no-cache-dir -r requirements_backend.txt

# Copy application code
COPY detection.py .
COPY face_recognition_api.py .
COPY models/ ./models/

# Create directories
RUN mkdir -p visitor_images

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "backend_api.py"]
