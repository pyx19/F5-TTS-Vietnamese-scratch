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
import contextlib
import io
import json
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

# ── Temp output cleanup ──────────────────────────────────────────────────────
# Mỗi lần bấm "Tổng hợp" tạo 1 file .wav tạm (tempfile delete=False — bắt buộc,
# Gradio cần đọc lại file sau khi hàm return). Trước đây các file này nằm thẳng
# trong thư mục temp hệ thống và KHÔNG bao giờ tự xóa, kể cả khi thoát bằng
# Ctrl+C — tích lũy vô hạn qua nhiều session (đã phát hiện thực tế: 18 file,
# ~38MB). Giờ dùng riêng 1 thư mục con để dọn dẹp an toàn (không đụng file tạm
# của app khác), và tự dọn cả lúc khởi động lẫn sau mỗi lần tổng hợp.
_TMP_DIR = Path(tempfile.gettempdir()) / "f5tts-vi-gradio"
_TMP_DIR.mkdir(parents=True, exist_ok=True)
_TMP_KEEP_LATEST = 5


def _cleanup_old_outputs(keep_latest: int = _TMP_KEEP_LATEST) -> None:
    """Xóa các file .wav tạm cũ trong _TMP_DIR, chỉ giữ lại keep_latest file gần nhất."""
    try:
        wavs = sorted(_TMP_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return
    for old in wavs[keep_latest:]:
        try:
            old.unlink()
        except OSError:
            pass


_cleanup_old_outputs(keep_latest=0)  # dọn sạch lúc khởi động — session cũ đã đóng, không còn cần

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


class _Tee(io.StringIO):
    """Ghi đè lên sys.stdout tạm thời — vừa in ra console như bình thường (để
    operator theo dõi khi chạy server), vừa giữ lại nội dung để bóc [WARN]/thống
    kê hiển thị lên UI (text_pipeline chỉ print(), không trả về structured log)."""

    def __init__(self, real_stdout):
        super().__init__()
        self._real = real_stdout

    def write(self, s: str) -> int:
        self._real.write(s)
        return super().write(s)


def _extract_warnings_and_stats(log: str) -> tuple[list[str], str | None]:
    """Bóc các dòng [WARN ...] và dòng thống kê '[*] LLM gate: ...' từ log đã capture."""
    warnings = [line.strip() for line in log.splitlines() if "[WARN" in line]
    stats = next(
        (line.strip()[4:] for line in log.splitlines() if line.strip().startswith("[*] LLM gate:")),
        None,
    )
    return warnings, stats


def _load_ui_examples() -> tuple[list[list[str]], list[str]]:
    """Đọc eval/test_sentences.json, lấy 2 câu/nhóm làm ví dụ mẫu cho UI —
    cùng test set dùng để audit pipeline, nên ví dụ luôn phản ánh đúng khả năng
    thực tế của hệ thống thay vì bịa câu demo tách rời."""
    category_labels = {
        "numbers_units": "Số/đơn vị",
        "admin_abbrev": "Viết tắt hành chính",
        "tech_abbrev": "Viết tắt kỹ thuật",
        "code_switching": "Code-switching",
        "proper_nouns": "Tên riêng",
    }
    path = BASE_DIR / "eval" / "test_sentences.json"
    if not path.exists():
        return [], []
    data = json.loads(path.read_text(encoding="utf-8"))
    per_category: dict[str, int] = {}
    examples, labels = [], []
    for item in data:
        cat = item["category"]
        if per_category.get(cat, 0) >= 2:
            continue
        per_category[cat] = per_category.get(cat, 0) + 1
        examples.append([item["text"]])
        labels.append(f"[{category_labels.get(cat, cat)}] {item['text'][:55]}")
    return examples, labels


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

    with tempfile.NamedTemporaryFile(dir=str(_TMP_DIR), suffix=".wav", delete=False) as f:
        tmp_path = Path(f.name)
    _cleanup_old_outputs()  # dọn output cũ hơn, chỉ giữ _TMP_KEEP_LATEST file gần nhất

    tee = _Tee(sys.stdout)
    try:
        with contextlib.redirect_stdout(tee):
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

        warnings, stats = _extract_warnings_and_stats(tee.getvalue())
        if llm_normalize:
            status += "\n💬 LLM normalize: ON"
            if stats:
                status += f"\n📊 {stats}"
        if warnings:
            status += "\n⚠️ " + "\n⚠️ ".join(warnings)
        return str(tmp_path), status
    except Exception as e:
        return None, f"❌ Lỗi: {e}"


def preview_text(gen_text, llm_normalize, ollama_model, ollama_url, llm_api_key):
    """
    Chạy ĐÚNG pipeline chuẩn hóa mà infer_vivoice.run() dùng thật (preprocess_text
    -> sanitize_for_vivoice -> normalize_for_f5), KHÔNG load model TTS — để văn bản
    xem trước khớp 100% với những gì model sẽ đọc, mà vẫn nhanh (không tốn thời gian
    diffusion/audio synthesis).
    """
    if not gen_text.strip():
        return "", "", gr.update(value="", visible=False)

    from infer_vivoice import PARAGRAPH_SILENCE_SEC, VOCAB_FILE
    from text_pipeline.chunking import DEFAULT_MAX_CHARS, normalize_for_f5
    from text_pipeline.pipeline import preprocess_text
    from text_pipeline.sanitize import sanitize_for_vivoice

    tee = _Tee(sys.stdout)
    try:
        with contextlib.redirect_stdout(tee):
            normalized = preprocess_text(
                gen_text,
                llm_model=ollama_model.strip() if llm_normalize else None,
                ollama_url=ollama_url.strip(),
                llm_api_key=llm_api_key.strip(),
            )
            sanitized = sanitize_for_vivoice(normalized, VOCAB_FILE)
            chunks, paragraph_break_after = normalize_for_f5(
                sanitized, max_chars=DEFAULT_MAX_CHARS,
                llm_model=ollama_model.strip() if llm_normalize else None,
                ollama_url=ollama_url.strip(), llm_api_key=llm_api_key.strip(),
            )
    except Exception as e:
        return "", f"❌ Lỗi khi chuẩn hóa: {e}", gr.update(value="", visible=False)

    preview_lines = []
    for i, c in enumerate(chunks):
        preview_lines.append(f"[{i + 1}/{len(chunks)}] {c}")
        if i in paragraph_break_after:
            preview_lines.append(f"— ⏸ nghỉ {PARAGRAPH_SILENCE_SEC}s (ranh giới đoạn văn) —")
    preview = "\n\n".join(preview_lines)

    warnings, stats = _extract_warnings_and_stats(tee.getvalue())
    stats_parts = [
        f"📦 {len(chunks)} đoạn (≤{DEFAULT_MAX_CHARS} ký tự/đoạn — đúng như lúc tổng hợp thật), "
        f"{len(paragraph_break_after)} ranh giới đoạn văn"
    ]
    if stats:
        stats_parts.append(stats)
    stats_line = " · ".join(stats_parts)

    if warnings:
        return preview, stats_line, gr.update(value="⚠️ " + "\n⚠️ ".join(warnings), visible=True)
    return preview, stats_line, gr.update(value="", visible=False)


# ── Build UI ─────────────────────────────────────────────────────────────────
# Giọng mẫu có sẵn — chọn nhanh bằng radio thay vì phải tự upload/ghi âm.
_VOICE_PRESETS = {
    "👩 Nữ (mặc định)": {
        "audio": str(BASE_DIR / "ref.wav") if (BASE_DIR / "ref.wav").exists() else None,
        "text": "cả hai bên hãy cố gắng hiểu cho nhau",
    },
    "👨 Nam": {
        "audio": str(BASE_DIR / "sample_nam.wav") if (BASE_DIR / "sample_nam.wav").exists() else None,
        "text": (
            "Chào mọi người! Hôm nay mình sẽ review một ứng dụng cực kỳ hot giúp bạn "
            "nâng cao năng suất làm việc mỗi ngày. Hãy cùng explore những tính năng "
            "tuyệt vời này nhé!"
        ),
    },
}
_default_voice_name = "👩 Nữ (mặc định)"
_default_ref = _VOICE_PRESETS[_default_voice_name]["audio"]
_default_ref_text = _VOICE_PRESETS[_default_voice_name]["text"]

with gr.Blocks(title="F5-TTS Vietnamese") as demo:
    gr.Markdown("# 🎙️ F5-TTS Vietnamese")
    gr.Markdown(
        "**Model**: `hynt/F5-TTS-Vietnamese-ViVoice` — Zero-shot voice cloning, "
        "tiếng Việt + tiếng Anh trong một model duy nhất.\n\n"
        "> Hệ thống đang dùng **giọng Nữ mặc định**. "
        "Bấm **\"🎤 Thay đổi giọng nói\"** bên dưới để chọn giọng Nam có sẵn, "
        "hoặc upload/ghi âm giọng riêng."
    )

    with gr.Accordion("🎤 Thay đổi giọng nói", open=False):
        voice_picker = gr.Radio(
            choices=list(_VOICE_PRESETS.keys()) + ["🎛️ Tùy chỉnh (upload/ghi âm riêng)"],
            value=_default_voice_name,
            label="Chọn giọng có sẵn",
        )
        gr.Markdown("""
**Yêu cầu audio mẫu tốt** (khi dùng "Tùy chỉnh"):
- ⏱️ Độ dài: **5–15 giây** (tối ưu 7–10 giây)
- 🔇 Không có tiếng ồn nền, tiếng vang, hay nhạc nền
- 🎙️ Giọng rõ ràng, tự nhiên — **không** dùng giọng TTS khác làm mẫu
- 📻 Sample rate: 22050 Hz hoặc 24000 Hz, mono

**Transcript** phải khớp **chính xác 100%** nội dung đã nói trong file audio.
""")
        with gr.Row():
            ref_audio = gr.Audio(
                label="📂 File audio mẫu (WAV / MP3 / FLAC)",
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

        def _on_voice_pick(name: str):
            preset = _VOICE_PRESETS.get(name)
            if preset is None:  # "Tùy chỉnh" — giữ nguyên audio/text người dùng đang có
                return gr.update(), gr.update()
            return gr.update(value=preset["audio"]), gr.update(value=preset["text"])

        voice_picker.change(_on_voice_pick, inputs=[voice_picker], outputs=[ref_audio, ref_text])

    with gr.Accordion("⚙️ Tham số nâng cao", open=False):
        gr.Markdown("Giá trị mặc định đã được tinh chỉnh cho mixed VI+EN. Chỉ thay đổi nếu cần.")
        with gr.Row():
            nfe_step     = gr.Slider(16, 128, value=64, step=8,
                                     label="NFE Steps — số bước diffusion",
                                     info="Cao hơn = mượt hơn nhưng chậm hơn. Thấp quá dễ rè.")
            cfg_strength = gr.Slider(0.5, 5.0, value=2.8, step=0.1,
                                     label="CFG Strength (2.8 tối ưu cho mixed VI+EN)",
                                     info="Cao = bám sát text/giọng mẫu hơn nhưng dễ cứng. Thấp = tự nhiên hơn nhưng dễ trôi giọng.")
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

    gr.Markdown("---")
    preview_btn = gr.Button("🔍 Xem trước văn bản chuẩn hóa", size="sm")
    gr.Markdown(
        "_Chạy đúng pipeline chuẩn hóa (số → LLM viết tắt → fallback → sanitize → chia đoạn) "
        "mà lúc Tổng hợp thật sẽ dùng, chỉ KHÔNG load model TTS — nên nhanh hơn nhiều. "
        "Mỗi dòng `[n/N]` dưới đây là một đoạn (chunk) sẽ được model đọc riêng biệt._"
    )
    preview_output   = gr.Textbox(label="📖 Văn bản model sẽ đọc (từng đoạn)", interactive=False, lines=6)
    preview_stats    = gr.Textbox(label="📊 Thống kê", interactive=False, lines=1)
    preview_warnings = gr.Textbox(label="⚠️ Cảnh báo", interactive=False, lines=3, visible=False)

    preview_btn.click(
        preview_text,
        inputs=[gen_text, llm_norm_chk, ollama_model_box, ollama_url_box, llm_api_key_box],
        outputs=[preview_output, preview_stats, preview_warnings],
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

    _example_values, _example_labels = _load_ui_examples()
    if _example_values:
        gr.Examples(
            examples=_example_values,
            inputs=[gen_text],
            example_labels=_example_labels,
            label="📋 Ví dụ mẫu theo từng nhóm tính năng (từ bộ test thật của pipeline)",
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