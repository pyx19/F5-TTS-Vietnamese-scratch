"""
Bước 10 — Debug tooling.

Biến môi trường:
  TTS_DEBUG=1      → in chi tiết từng bước preprocessing
  TTS_DEBUG_LLM=1  → in raw prompt/response của từng LLM call

Biến môi trường cho LLM endpoint (để trỏ vào server tự host, không phụ thuộc Ollama
trên máy dev — xem README mục "Trỏ vào LLM tự host"):
  OLLAMA_URL       → URL server OpenAI-compatible (Ollama/vLLM/LM Studio/Xinference...)
  OLLAMA_MODEL     → tên model trên server đó (mặc định "qwen2.5:3b" chỉ là placeholder)
  LLM_API_KEY      → API key nếu server yêu cầu Bearer token
"""

import os

DBG = os.environ.get("TTS_DEBUG", "0") == "1"
DBG_LLM = os.environ.get("TTS_DEBUG_LLM", "0") == "1"

# Trong container Docker, "localhost" trỏ vào chính container chứ không phải host
# đang chạy Ollama → phải dùng host.docker.internal.
IN_DOCKER = os.path.exists("/.dockerenv")

DEFAULT_OLLAMA_URL = os.environ.get(
    "OLLAMA_URL",
    "http://host.docker.internal:11434" if IN_DOCKER else "http://localhost:11434",
)

# qwen2.5:3b (Ollama local) chỉ là placeholder mặc định — set OLLAMA_MODEL để trỏ
# sang model đang host thật (vd trên server công ty, có thể không phải Ollama/Qwen).
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
DEFAULT_LLM_API_KEY = os.environ.get("LLM_API_KEY", "")


def dbg(tag: str, msg: str) -> None:
    if DBG:
        print(f"  [DBG:{tag}] {msg}", flush=True)
