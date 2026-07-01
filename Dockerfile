# ============================================================
# F5-TTS-Vietnamese — Docker image FULL BAKE (CUDA 12.1, offline-ready)
# Base: nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
#
# Bake TOÀN BỘ vào image: code + model weights (~10GB) + dependencies.
# Dùng khi máy đích (vd VM công ty không có mạng) chỉ cần pull/load image và
# chạy ngay, KHÔNG sửa code trên máy đó. Nếu cần sửa code sau khi deploy, xem
# Dockerfile.deps-only (bake model+deps, mount code từ git clone riêng).
#
# Build (với internet):
#   docker build -t f5tts-vi:latest .
#
# Đẩy lên Docker Hub rồi pull xuống máy không có internet:
#   docker push <user>/f5tts-vi:cu121
#   # trên máy đích (cần Docker + nvidia-container-toolkit + reach Docker Hub,
#   # hoặc dùng docker save/load nếu máy đích hoàn toàn không có mạng — xem dưới):
#   docker pull <user>/f5tts-vi:cu121
#
# Nếu máy đích không reach được Docker Hub (mạng nội bộ cô lập hoàn toàn):
#   docker save f5tts-vi:latest | gzip > f5tts-vi.tar.gz   # trên máy build
#   # copy file .tar.gz sang máy đích qua USB/SCP nội bộ, rồi:
#   docker load < f5tts-vi.tar.gz                          # trên máy đích
#
# Run (trên server):
#   docker compose up -d          # dùng docker-compose.yml
#   # hoặc thủ công (models đã baked vào image):
#   docker run --gpus all -p 7860:7860 \
#     -v /path/to/output:/app/tests \
#     -e OLLAMA_URL=http://YOUR_LLM_HOST:PORT/v1 \
#     -e OLLAMA_MODEL=your-model-name \
#     f5tts-vi:latest
#
# LƯU Ý: bake xong image đã tự chứa mọi thứ để chạy INFERENCE hoàn toàn offline
# (HF_HUB_OFFLINE=1). LLM normalize (--llm_normalize) vẫn cần container reach được
# một LLM server qua mạng LAN/nội bộ (không cần internet) — trỏ qua OLLAMA_URL.
# Cần GPU + nvidia-container-toolkit đã cài trên máy đích để dùng --gpus all.
# ============================================================

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ── System env ──────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONIOENCODING=utf-8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Block ALL HuggingFace / transformers network calls at runtime
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    # Disable Gradio telemetry
    GRADIO_ANALYTICS_ENABLED=0 \
    # Point HF cache to /app/ckpts (models baked or mounted here)
    HF_HOME=/app/ckpts \
    TORCH_HOME=/app/ckpts

# ── System packages ──────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    build-essential \
    git \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && python -m pip install --upgrade pip setuptools wheel

# ── App directory ────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ──────────────────────────────
# Copy requirements first (Docker layer cache — only re-run if reqs change)
COPY requirements-docker.txt .

# Step 1: Install PyTorch cu121 first (largest, most likely to need retry)
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1+cu121 \
    torchaudio==2.5.1+cu121 \
    torchvision==0.20.1+cu121

# Step 2: Install remaining dependencies
# Skip torch lines to avoid index conflict; install rest from PyPI
RUN pip install --no-cache-dir \
    vocos==0.1.0 \
    ema-pytorch==0.7.9 \
    "x-transformers==2.19.12" \
    torchdiffeq==0.2.5 \
    einops==0.8.2 \
    einx==0.4.3 \
    "hydra-core==1.3.2" \
    "omegaconf==2.3.0" \
    cached_path==1.8.10 \
    safetensors==0.7.0 \
    "huggingface_hub==0.36.2" \
    "transformers==4.27.4" \
    "tokenizers==0.13.3" \
    "accelerate==1.13.0" \
    soundfile==0.14.0 \
    "librosa==0.9.1" \
    pydub==0.25.1 \
    encodec==0.1.1 \
    "scipy==1.15.3" \
    resampy==0.4.3 \
    vietnormalizer==0.2.3 \
    diskcache==5.6.3 \
    fasttext-wheel==0.9.2 \
    jieba==0.42.1 \
    pypinyin==0.50.0 \
    inflect==7.0.0 \
    anyascii==0.3.2 \
    Unidecode==1.3.7 \
    num2words==0.5.12 \
    cn2an==0.5.22 \
    "gradio==6.10.0" \
    "fastapi==0.136.3" \
    uvicorn==0.49.0 \
    python-multipart==0.0.32 \
    "numpy==1.26.4" \
    tqdm==4.68.1 \
    click==8.4.1 \
    tomli==2.4.1 \
    packaging \
    filelock \
    matplotlib \
    Pillow \
    PyYAML \
    requests \
    pandas

# ── Copy application code ─────────────────────────────────────
COPY src/ ./src/
COPY data/ ./data/
COPY pyproject.toml .
COPY text_pipeline/ ./text_pipeline/
COPY gradio_app.py .
COPY infer_vivoice.py .
COPY eval/ ./eval/
COPY ref.wav .
COPY ref2.wav .
COPY tests/ ./tests/

# ── Bake models vào image (~10GB: vivoice + base + vocos) ────
# Layer riêng → Docker cache không thay đổi nếu model không đổi.
COPY ckpts/ ./ckpts/

# ── Install the f5_tts package itself (editable, from copied src/) ──
RUN pip install --no-cache-dir --no-deps -e .

# ── Runtime dirs ─────────────────────────────────────────────
RUN mkdir -p /app/tests /checkpoints
# infer_cli.py hardcodes vocoder path as "../checkpoints/vocos-mel-24khz"
# relative to WORKDIR=/app → resolves to /checkpoints/vocos-mel-24khz.
# docker-entrypoint.sh tạo symlink /checkpoints/vocos-mel-24khz → /app/ckpts/vocos-mel-24khz.
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# ── Expose Gradio port ───────────────────────────────────────
EXPOSE 7860

# ── Healthcheck ──────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# ── Entrypoint + default command ─────────────────────────────
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["python", "gradio_app.py", "--host", "0.0.0.0", "--port", "7860"]
