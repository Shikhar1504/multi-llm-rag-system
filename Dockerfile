FROM python:3.11-slim

WORKDIR /app

# Environment
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=""

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# 🔥 CORRECT DEPENDENCY INSTALL FLOW
RUN python -m pip install --upgrade pip \
    # 1. Install CPU-only torch (prevents CUDA garbage)
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision torchaudio \
    # 2. Install REQUIRED deps manually (because we skip them later)
    && pip install --no-cache-dir \
        transformers==4.41.2 \
        scikit-learn==1.5.0 \
        scipy==1.13.1 \
    # 3. Install sentence-transformers WITHOUT overriding torch
    && pip install --no-cache-dir sentence-transformers==3.0.1 --no-deps \
    # 4. Install remaining project deps
    && pip install --no-cache-dir -r requirements.txt

# Preload models (CACHE SAFE)
COPY preload_models.py /tmp/preload_models.py
RUN python /tmp/preload_models.py

# Copy app AFTER models (important for caching)
COPY . .

# Expose port
EXPOSE 8000

# Start server (production)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]