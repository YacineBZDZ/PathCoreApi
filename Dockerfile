FROM python:3.11-slim

# HF Spaces requires port 7860
ENV PORT=7860
WORKDIR /app

# Install system deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code and models
COPY main.py .
COPY models/ ./models/

# HF Spaces runs as non-root user 1000 — set permissions
RUN chmod -R 755 /app

EXPOSE 7860

# Use port 7860 for HF Spaces (not $PORT env var)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
