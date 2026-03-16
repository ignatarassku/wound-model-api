FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (smaller image, faster build)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what's needed for inference (data/ is excluded via .dockerignore)
COPY src/ ./src/
COPY checkpoints/best_model.pth ./checkpoints/best_model.pth
COPY main.py .

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Use $PORT env var (Railway sets this automatically)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
