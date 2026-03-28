FROM python:3.11-slim

WORKDIR /app

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

ARG BINARY_CHECKPOINT_FETCH_URL=
RUN if [ -n "$BINARY_CHECKPOINT_FETCH_URL" ]; then \
      curl -fSL --output checkpoints/best_model.pth "$BINARY_CHECKPOINT_FETCH_URL"; \
    fi

RUN adduser --disabled-password --gecos '' appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
