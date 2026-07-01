"""
Gradio UI cho F5-TTS Vietnamese — ViVoice single-model VI+EN.

Chạy:
  python gradio_app.py
  python gradio_app.py --share    # tạo public URL
  python gradio_app.py --port 7861
  python gradio_app.py --host 0.0.0.0   # cho phép truy cập từ LAN
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

# ── Windows encoding fix ────────────────────────────────────────────────────
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    # Suppress benign WinError 10054 from asyncio ProactorEventLoop (Gradio cleanup)
    import asyncio, logging
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)

# ── Patch torchaudio.load (Windows torchcodec fix) ──────────────────────────
def _sf_load(path, frame_offset=0, num_frames=-1, normalize=True,
             channels_first=True, **kwargs):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if frame_offset > 0: data = data[frame_offset:]
    if num_frames > 0:   data = data[:num_frames]
    return torch.from_numpy(data.T if channels_first else data), sr

torchaudio.load = _sf_load

BASE_DIR    = Path(__file__).parent
SAMPLE_RATE = 24000
device      = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.insert(0, str(BASE_DIR))
from text_pipeline.debug import (  # noqa: E402
    DEFAULT_LLM_API_KEY as _DEFAULT_LLM_API_KEY,
    DEFAULT_OLLAMA_MODEL as _DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL as _DEFAULT_OLLAMA_URL,
    IN_DOCKER as _IN_DOCKER,
)

print(f"[*] Device: {device.upper()}")
print("[*] Model sẽ được load lần đầu khi bấm Tổng hợp.\n")

if _IN_DOCKER:
    print(f"[*] Chạy trong Docker — Ollama URL mặc định: {_DEFAULT_OLLAMA_URL}")


def _check_llm_health(url: str, model: str, api_key: str = "") -> tuple[bool, str]:
    """
    Kiểm tra LLM server (OpenAI-compatible) đang chạy và model khả dụng.
    Returns: (ok, message)
    """
    import json
    import urllib.request
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        req = urllib.request.Request(
            f"{url.rstrip('/')}/v1/models",
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
        available = [m["id"] for m in data.get("data", [])]
        model_base = model.split(":")[0]
        if not available or any(model_base in m for m in available):
            return True, f"✅ LLM server OK — model: {model}"
        return False, f"⚠️ Model '{model}' không thấy trên server.\nCó sẵn: {', '.join(available[:5])}"
    except Exception as e:
        return False, f"❌ LLM server không phản hồi tại {url}\n({type(e).__name__}: {e})"

# ── Gradio ───────────────────────────────────────────────────────────────────
import gradio as gr


def infer_vivoice_wrap(ref_audio, ref_text, gen_text, nfe_step, cfg_strength,
                       speed_val, llm_normalize, ollama_model, ollama_url, llm_api_key):
    if ref_audio is None:
        return None, "⚠️ Chưa chọn Reference audio"
    if not gen_text.strip():
        return None, "⚠️ Chưa nhập văn bản"

    from infer_vivoice import run as vivoice_run, _find_ckpt, VOCAB_FILE

    ckpt = _find_ckpt()
    if ckpt is None:
        return None, (
            "❌ Model ViVoice chưa được tải.\n"
            "Chạy lệnh này trong terminal rồi khởi động lại:\n"
            "  python download_vivoice.py"
        )
    if not VOCAB_FILE.exists():
        return None, f"❌ Vocab không tìm thấy: {VOCAB_FILE}"

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = Path(f.name)

    try:
        vivoice_run(
            text=gen_text,
            ref_audio=Path(ref_audio),
            ref_text=ref_text,
            output=tmp_path,
            device=device,
            nfe_step=int(nfe_step),
            cfg_strength=float(cfg_strength),
            spd=float(speed_val),
            llm_model=ollama_model.strip() if llm_normalize else None,
            ollama_url=ollama_url.strip(),
            llm_api_key=llm_api_key.strip(),
        )
        audio_data, _ = sf.read(str(tmp_path))
        duration = len(audio_data) / SAMPLE_RATE
        size_kb = tmp_path.stat().st_size / 1024
        status = f"✅ Done — {duration:.1f}s audio, {size_kb:.0f} KB @ {SAMPLE_RATE}Hz"
        if llm_normalize:
            status += "\n💬 LLM normalize: ON"
        return str(tmp_path), status
    except Exception as e:
        return None, f"❌ Lỗi: {e}"


# ── Build UI ─────────────────────────────────────────────────────────────────
_default_ref      = str(BASE_DIR / "ref.wav") if (BASE_DIR / "ref.wav").exists() else None
_default_ref_text = "cả hai bên hãy cố gắng hiểu cho nhau"

with gr.Blocks(title="F5-TTS Vietnamese") as demo:
    gr.Markdown("# 🎙️ F5-TTS Vietnamese")
    gr.Markdown(
        "**Model**: `hynt/F5-TTS-Vietnamese-ViVoice` — Zero-shot voice cloning, "
        "tiếng Việt + tiếng Anh trong một model duy nhất.\n\n"
        "> Hệ thống đang dùng **giọng mặc định** có sẵn. "
        "Để dùng giọng khác, bấm **\"🎤 Thay đổi giọng nói\"** bên dưới."
    )

    with gr.Accordion("🎤 Thay đổi giọng nói", open=False):
        gr.Markdown("""
**Yêu cầu audio mẫu tốt:**
- ⏱️ Độ dài: **5–15 giây** (tối ưu 7–10 giây)
- 🔇 Không có tiếng ồn nền, tiếng vang, hay nhạc nền
- 🎙️ Giọng rõ ràng, tự nhiên — **không** dùng giọng TTS khác làm mẫu
- 📻 Sample rate: 22050 Hz hoặc 24000 Hz, mono

**Transcript** phải khớp **chính xác 100%** nội dung đã nói trong file audio.
""")
        with gr.Row():
            ref_audio = gr.Audio(
                label="📂 Upload file audio mẫu (WAV / MP3 / FLAC)",
                type="filepath",
                value=_default_ref,
                sources=["upload", "microphone"],
            )
            ref_text = gr.Textbox(
                label="📝 Transcript",
                value=_default_ref_text,
                lines=3,
                placeholder="Nhập chính xác những gì được nói trong file audio...",
            )

    with gr.Accordion("⚙️ Tham số nâng cao", open=False):
        gr.Markdown("Giá trị mặc định đã được tinh chỉnh cho mixed VI+EN. Chỉ thay đổi nếu cần.")
        with gr.Row():
            nfe_step     = gr.Slider(16, 128, value=64, step=8,
                                     label="NFE Steps — số bước diffusion")
            cfg_strength = gr.Slider(0.5, 5.0, value=2.8, step=0.1,
                                     label="CFG Strength (2.8 tối ưu cho mixed VI+EN)")
            speed_val    = gr.Slider(0.5, 2.0, value=1.0, step=0.05,
                                     label="Speed — tốc độ đọc")

    gen_text = gr.Textbox(
        label="Văn bản cần đọc",
        placeholder=(
            "Ví dụ: Các tập đoàn Fortune 500 đang đầu tư mạnh vào AI và Machine Learning "
            "để tối ưu ROI và thúc đẩy Digital Transformation."
        ),
        lines=6,
    )

    llm_norm_chk = gr.Checkbox(
        label="💬 LLM normalize (Qwen)",
        value=False,
        info="Dùng Qwen để chuẩn hóa viết tắt/số theo ngữ cảnh. Cần Ollama đang chạy.",
    )

    with gr.Accordion("⚙️ LLM Settings", open=False):
        gr.Markdown(
            "Hỗ trợ mọi server **OpenAI-compatible**: vLLM, LM Studio, Xinference, Ollama...\n\n"
            "Nếu server không phản hồi → tự động fallback về rule-based preprocessing."
        )
        with gr.Row():
            ollama_model_box = gr.Textbox(value=_DEFAULT_OLLAMA_MODEL, label="Model name")
            ollama_url_box   = gr.Textbox(value=_DEFAULT_OLLAMA_URL, label="LLM URL")
            llm_api_key_box  = gr.Textbox(value=_DEFAULT_LLM_API_KEY, label="API Key (nếu cần)", type="password")
        ollama_status_box = gr.Textbox(
            label="Trạng thái LLM",
            interactive=False,
            visible=False,
            lines=2,
        )
        check_btn = gr.Button("🔍 Kiểm tra kết nối", size="sm")

    def on_llm_toggle(enabled: bool, model: str, url: str, api_key: str):
        if not enabled:
            return gr.update(visible=False), ""
        ok, msg = _check_llm_health(url.strip(), model.strip(), api_key.strip())
        return gr.update(visible=True), msg

    def on_check_click(model: str, url: str, api_key: str):
        _, msg = _check_llm_health(url.strip(), model.strip(), api_key.strip())
        return gr.update(visible=True), msg

    llm_norm_chk.change(
        on_llm_toggle,
        inputs=[llm_norm_chk, ollama_model_box, ollama_url_box, llm_api_key_box],
        outputs=[ollama_status_box, ollama_status_box],
    )
    check_btn.click(
        on_check_click,
        inputs=[ollama_model_box, ollama_url_box, llm_api_key_box],
        outputs=[ollama_status_box, ollama_status_box],
    )

    btn        = gr.Button("▶ Tổng hợp", variant="primary", size="lg")
    out_audio  = gr.Audio(label="🔊 Output", type="filepath")
    status_out = gr.Textbox(label="Trạng thái", interactive=False, lines=3)

    btn.click(
        infer_vivoice_wrap,
        inputs=[ref_audio, ref_text, gen_text,
                nfe_step, cfg_strength, speed_val,
                llm_norm_chk, ollama_model_box, ollama_url_box, llm_api_key_box],
        outputs=[out_audio, status_out],
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="F5-TTS Vietnamese — ViVoice Gradio UI")
    ap.add_argument("--share",  action="store_true", help="Tạo public Gradio URL")
    ap.add_argument("--port",   type=int, default=7860)
    ap.add_argument("--host",   type=str, default="127.0.0.1")
    args = ap.parse_args()

    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "0")

    import socket as _socket

    def _print_banner(port: int, host: str, share: bool) -> None:
        host_display = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
        lines = [
            "",
            "─" * 54,
            "  🎙️  F5-TTS Vietnamese — sẵn sàng!",
            "─" * 54,
            f"  Local  :  http://{host_display}:{port}",
        ]
        if host == "0.0.0.0":
            try:
                lan_ip = _socket.gethostbyname(_socket.gethostname())
                lines.append(f"  LAN    :  http://{lan_ip}:{port}")
            except Exception:
                pass
        if share:
            lines.append("  Public :  (xem link Gradio tunnel phía trên)")
        lines += ["─" * 54, "  Nhấn Ctrl+C để dừng.", "─" * 54, ""]
        print("\n".join(lines), flush=True)

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        prevent_thread_lock=True,
    )
    _print_banner(args.port, args.host, args.share)
    demo.block_thread()