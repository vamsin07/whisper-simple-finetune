# Image for running whisper-vanilla-finetune on Nautilus.
#
# Build + push:
#   docker build -t gitlab-registry.nautilus.optiputer.net/<your-group>/whisper-vanilla:latest -f nautilus/Dockerfile .
#   docker push gitlab-registry.nautilus.optiputer.net/<your-group>/whisper-vanilla:latest
#
# CUDA 12.1 + cuDNN 9 runtime; PyTorch 2.5. Adjust if you need a different
# CUDA / PyTorch combo to match a specific GPU node's drivers.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

# System libs librosa needs for audio decoding (libsndfile + ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

# Python deps — pin to what the repo was developed against
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the code in last so deps cache survives src/ edits
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY README.md /app/

# Useful defaults for HF caches; the k8s Job overrides HF_HOME via a PVC mount
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/hf \
    TOKENIZERS_PARALLELISM=false

CMD ["python", "src/train.py", "--help"]
