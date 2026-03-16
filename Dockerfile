FROM python:3.11-slim

WORKDIR /app

# Install system dependencies + git-lfs so LFS files are pulled correctly
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    git \
    git-lfs \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (numpy<2 required for torch compatibility)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and checkpoint
COPY src/ ./src/
COPY main.py .
COPY checkpoints/ ./checkpoints/

# Verify the checkpoint is a real binary (not a Git LFS pointer)
RUN python -c "
import os, sys
path = 'checkpoints/best_model.pth'
size = os.path.getsize(path)
print(f'Checkpoint size: {size} bytes')
if size < 10000:
    with open(path, 'rb') as f:
        header = f.read(200)
    print('File header:', header)
    print('ERROR: Checkpoint looks like an LFS pointer, not a real model file.')
    sys.exit(1)
print('Checkpoint OK - real binary file confirmed')
"

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
