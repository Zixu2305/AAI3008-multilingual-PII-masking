# GPU runtime base (works locally + cloud with NVIDIA Container Toolkit)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps: python, ffmpeg, build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    ffmpeg \
    git curl ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy only requirements first to maximize Docker layer caching
COPY requirements.txt /workspace/requirements.txt

RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install -r /workspace/requirements.txt

# Useful defaults for HuggingFace caching inside mounted volume
ENV HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONPATH=/workspace

# Copy project (optional; compose also mounts repo into /workspace)
COPY . /workspace

CMD ["bash"]
