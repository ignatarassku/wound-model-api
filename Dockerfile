# CPU-only PyTorch avoids multi-GB CUDA wheels on CPU hosts.

FROM python:3.11-slim

WORKDIR /app

# OpenCV (ultralytics/cv2) needs X11 libs at import time — slim image omits them by default.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    libxcb1 \
    libglib2.0-0 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cpu
RUN grep -v '^torch' requirements.txt > /tmp/req.txt && pip install --no-cache-dir -r /tmp/req.txt

RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

COPY . .

# Optional build-time fetch of model checkpoint
ARG BINARY_CHECKPOINT_FETCH_URL=
RUN if [ -n "$BINARY_CHECKPOINT_FETCH_URL" ]; then \
      curl -fSL --output checkpoints/best_model.pth "$BINARY_CHECKPOINT_FETCH_URL"; \
    fi

RUN adduser --disabled-password --gecos '' appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

COPY start.sh /app/start.sh

ENTRYPOINT ["/app/start.sh"]
