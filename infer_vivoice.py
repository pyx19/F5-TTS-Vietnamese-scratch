"""
ViVoice TTS — Vietnamese + English với một model duy nhất. [scratch rewrite]

ViVoice (hynt/F5-TTS-Vietnamese-ViVoice) fine-tune từ F5TTS_Base với extend embedding:
  - EN token weights giữ nguyên từ F5TTS_Base (100K giờ EN+ZH)
  - VI tokens được học trong cùng acoustic space
  → 1 forward pass xử lý được mixed VI+EN, không cần language routing / 2 model.

Text preprocessing được tách riêng vào package text_pipeline/ (xem plan.md):
  preprocess_text()      — Stage 1-4: numbers -> LLM normalize -> abbrev fallback -> cleanup
  sanitize_for_vivoice()  — Stage 6: strip ký tự ngoài vocab
  normalize_for_f5()      — Stage 9: chunk <= max_chars, ranh giới câu Việt-aware

Debug flags (biến môi trường):
  TTS_DEBUG=1          — in chi tiết từng bước preprocessing
  TTS_DEBUG_LLM=1      — in raw input/output của từng LLM call

Usage:
  python infer_vivoice.py \\
    --ref_audio ref.wav \\
    --ref_text  "cả hai bên hãy cố gắng hiểu cho nhau" \\
    --gen_text  "Hệ thống AI và Machine Learning giúp CEO ra quyết định."

  # Với LLM normalize (khuyến nghị):
  python infer_vivoice.py \\
    --ref_audio ref.wav \\
    --ref_text  "cả hai bên hãy cố gắng hiểu cho nhau" \\
    --gen_file  tests/sample_english2.txt \\
    --llm_normalize --ollama_model qwen2.5:3b
"""

import argparse
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

# ── Windows encoding fix ─────────────────────────────────────────────────────
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ── Patch torchaudio.load (Windows torchcodec fix) ───────────────────────────
def _sf_load(path, frame_offset=0, num_frames=-1, normalize=True,
             channels_first=True, **kwargs):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if frame_offset > 0:
        data = data[frame_offset:]
    if num_frames > 0:
        data = data[:num_frames]
    return torch.from_numpy(data.T if channels_first else data), sr


torchaudio.load = _sf_load

BASE_DIR = Path(__file__).parent

# F5TTS_CKPTS_DIR cho phép model weights nằm TÁCH RỜI khỏi thư mục code — dùng khi
# build Docker image kiểu "deps-only" (bake model+deps, code mount từ ngoài/git clone
# riêng, xem Dockerfile.deps-only + README). Mặc định vẫn là BASE_DIR/ckpts như cũ.
CKPTS_DIR = Path(os.environ.get("F5TTS_CKPTS_DIR", str(BASE_DIR / "ckpts")))
VOCOS_DIR = CKPTS_DIR / "vocos-mel-24khz"
VIVOICE_DIR = CKPTS_DIR / "vivoice"
VOCAB_FILE = VIVOICE_DIR / "vocab.txt"
SAMPLE_RATE = 24000

# f5_tts đã pip-install editable trỏ về F5-TTS-Vietnamese/src ở project gốc (hoặc về
# vị trí baked riêng trong image deps-only) — chèn src/ cạnh file này lên đầu sys.path
# để ưu tiên dùng bản model code đã recycle tại đây, nếu nó tồn tại tại chỗ.
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR))
from text_pipeline.chunking import DEFAULT_MAX_CHARS, normalize_for_f5  # noqa: E402
from text_pipeline.debug import (  # noqa: E402
    DEFAULT_LLM_API_KEY,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    dbg,
)
from text_pipeline.pipeline import preprocess_text  # noqa: E402
from text_pipeline.sanitize import sanitize_for_vivoice  # noqa: E402


def _find_ckpt() -> Path | None:
    for ext in ("*.pt", "*.safetensors"):
        candidates = sorted(VIVOICE_DIR.glob(ext))
        if candidates:
            return candidates[-1]
    return None


# ════════════════════════════════════════════════════════════════════════════
# Model loading (lazy, cached)
# ════════════════════════════════════════════════════════════════════════════

_model_cache: dict = {}


def _get_model(ckpt: Path, device: str):
    key = str(ckpt)
    if key not in _model_cache:
        from omegaconf import OmegaConf
        from hydra.utils import get_class
        from f5_tts.infer.utils_infer import load_model, load_vocoder

        cfg_path = BASE_DIR / "src/f5_tts/configs/F5TTS_Base.yaml"
        model_cfg = OmegaConf.load(str(cfg_path))
        model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
        model_arc = model_cfg.model.arch

        print(f"  [load] {ckpt.name}  ({ckpt.stat().st_size / 1e9:.2f} GB) ...")
        ema = load_model(
            model_cls, model_arc, str(ckpt),
            mel_spec_type="vocos",
            vocab_file=str(VOCAB_FILE),
            device=device,
        )
        vocoder = load_vocoder(
            vocoder_name="vocos",
            is_local=VOCOS_DIR.exists(),
            local_path=str(VOCOS_DIR) if VOCOS_DIR.exists() else None,
            device=device,
        )
        _model_cache[key] = (ema, vocoder)
    return _model_cache[key]


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def run(text: str, ref_audio: Path, ref_text: str, output: Path,
        device: str, nfe_step: int, cfg_strength: float, spd: float,
        llm_model: str | None = None, ollama_url: str = DEFAULT_OLLAMA_URL,
        llm_api_key: str = "", max_chars: int = DEFAULT_MAX_CHARS):
    from f5_tts.infer.utils_infer import (
        preprocess_ref_audio_text, infer_batch_process,
        target_rms, cross_fade_duration, sway_sampling_coef, fix_duration,
    )

    ckpt = _find_ckpt()
    if ckpt is None:
        print("[ERROR] Không tìm thấy model trong ckpts/vivoice/")
        print("        Chạy trước: python download_vivoice.py")
        sys.exit(1)
    if not VOCAB_FILE.exists():
        print(f"[ERROR] Vocab không tìm thấy: {VOCAB_FILE}")
        sys.exit(1)

    print(f"[*] Device      : {device.upper()}")
    print(f"[*] Model       : {ckpt.name}")
    print(f"[*] cfg_strength: {cfg_strength}  nfe_step: {nfe_step}  speed: {spd}")

    raw_preview = text[:100] + ("..." if len(text) > 100 else "")
    print(f"[*] Input       : {raw_preview}")

    normalized = preprocess_text(text, llm_model=llm_model, ollama_url=ollama_url, llm_api_key=llm_api_key)
    sanitized = sanitize_for_vivoice(normalized, VOCAB_FILE)
    if sanitized != normalized:
        dbg("PREP", f"after sanitize_for_vivoice: {sanitized[:120]}")

    chunks = normalize_for_f5(sanitized, max_chars=max_chars)
    print(f"[*] Preprocessed: {sanitized[:100]}{'...' if len(sanitized) > 100 else ''}")
    print(f"[*] Chunks      : {len(chunks)} (max {max_chars} chars/chunk)")

    print("\n[*] Loading model ...")
    t_load = time.perf_counter()
    ema, vocoder = _get_model(ckpt, device)
    print(f"    Done in {time.perf_counter() - t_load:.1f}s")

    print("[*] Preprocessing ref audio ...")
    ref_proc, ref_text_proc = preprocess_ref_audio_text(str(ref_audio), ref_text)

    print("[*] Synthesizing ...")
    t0 = time.perf_counter()

    ref_wave, ref_sr = torchaudio.load(ref_proc)
    audio, out_sr, _spec = next(
        infer_batch_process(
            (ref_wave, ref_sr), ref_text_proc, chunks, ema, vocoder,
            mel_spec_type="vocos",
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=spd,
            fix_duration=fix_duration,
            device=device,
        )
    )

    elapsed = time.perf_counter() - t0
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), audio, out_sr)
    size_kb = output.stat().st_size / 1024
    print(f"    Done in {elapsed:.1f}s — {size_kb:.0f} KB @ {SAMPLE_RATE}Hz")
    print(f"\n[OK] Saved: {output.resolve()}")


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ViVoice single-model VI+EN TTS")
    ap.add_argument("--ref_audio", required=True)
    ap.add_argument("--ref_text", required=True)
    ap.add_argument("--gen_text", default=None)
    ap.add_argument("--gen_file", default=None)
    ap.add_argument("--out", default="tests/output_vivoice.wav")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--nfe_step", type=int, default=64)
    ap.add_argument("--cfg_strength", type=float, default=2.8,
                     help="CFG strength — 2.5-3.0 cho mixed text (thấp hơn 3.5 VI thuần)")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--max_chars", type=int, default=DEFAULT_MAX_CHARS,
                     help="Chunk size tối đa cho normalize_for_f5() (mặc định 200)")
    ap.add_argument("--llm_normalize", action="store_true",
                     help="Dùng LLM để chuẩn hóa text (cần server đang chạy)")
    ap.add_argument("--ollama_model", default=DEFAULT_OLLAMA_MODEL,
                     help="Tên model LLM — mặc định lấy từ env OLLAMA_MODEL, "
                          "fallback 'qwen2.5:3b' (chỉ là placeholder, đổi theo LLM bạn host)")
    ap.add_argument("--ollama_url", default=DEFAULT_OLLAMA_URL,
                     help="URL LLM server OpenAI-compatible — mặc định lấy từ env OLLAMA_URL")
    ap.add_argument("--llm_api_key", default=DEFAULT_LLM_API_KEY,
                     help="API key nếu server yêu cầu — mặc định lấy từ env LLM_API_KEY")
    args = ap.parse_args()

    if args.gen_file:
        gen_text = Path(args.gen_file).read_text(encoding="utf-8").strip()
    elif args.gen_text:
        gen_text = args.gen_text
    else:
        print("[ERROR] Cần --gen_text hoặc --gen_file")
        sys.exit(1)

    run_device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    run(
        text=gen_text,
        ref_audio=Path(args.ref_audio),
        ref_text=args.ref_text,
        output=Path(args.out),
        device=run_device,
        nfe_step=args.nfe_step,
        cfg_strength=args.cfg_strength,
        spd=args.speed,
        llm_model=args.ollama_model if args.llm_normalize else None,
        ollama_url=args.ollama_url,
        llm_api_key=args.llm_api_key,
        max_chars=args.max_chars,
    )
