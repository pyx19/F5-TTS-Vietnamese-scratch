"""
preprocess_text() — orchestrator cho toàn bộ pipeline mô tả trong plan.md:

  Raw Text
     -> _normalize_linebreaks() / _remove_parens()   (dọn cấu trúc trước mọi bước)
     -> [Stage 1] normalize_numbers() + vi_numbers()  (rule-based, TRƯỚC Qwen)
     -> [Stage 2] needs_llm() gate                    (bên trong llm_normalize)
     -> [Stage 3] llm_normalize() / Qwen               (per-sentence, có sanity_ok + cache)
     -> [Stage 4] expand_acronyms_fallback() + expand_abbrevs_fallback()  (safety net sau Qwen)
     -> lowercase + cleanup ký tự nối
     -> [Stage 6] sanitize_for_vivoice()               (strip ký tự ngoài vocab, cần vocab_file)
     -> normalize_for_f5()                             (chunk <= max_chars cho F5-TTS)
"""

import re
from pathlib import Path

from .acronyms import expand_acronyms_fallback
from .chunking import DEFAULT_MAX_CHARS, normalize_for_f5
from .debug import dbg
from .letters import expand_abbrevs_fallback
from .llm_normalize import llm_normalize
from .numbers import normalize_numbers, vi_numbers
from .sanitize import sanitize_for_vivoice


def _normalize_linebreaks(text: str) -> str:
    """
    Chuyển cấu trúc xuống dòng → ranh giới câu để F5-TTS tạo ngắt nghỉ.
    - Mỗi dòng không kết thúc bằng dấu câu → thêm dấu chấm
    - Tab → space
    - Dòng trống → bỏ qua
    Phải chạy TRƯỚC mọi bước khác (kể cả LLM) để giữ đúng cấu trúc.
    """
    out = []
    for line in text.split("\n"):
        line = re.sub(r"\t+", " ", line).strip()
        if not line:
            continue
        if line[-1] not in ".!?,;:":
            line += "."
        out.append(line)
    return " ".join(out)


def _remove_parens(text: str) -> str:
    """
    Xóa dấu ngoặc (), giữ nội dung bên trong.
    TTS model đọc ký tự '(' thành âm ký sinh ("a", "ở", "e"...).
      (Cloud Computing) → Cloud Computing
      (ZTA)             → ZTA  (LLM xử lý tiếp)
    Ngoặc lẻ không có cặp cũng bị xóa.
    """
    text = re.sub(r"\(([^)]*)\)", r" \1", text)
    text = re.sub(r"[()]", "", text)
    text = re.sub(r"\s+([,;:])", r"\1", text)
    return re.sub(r"  +", " ", text).strip()


def preprocess_text(
    text: str,
    llm_model: str | None = None,
    ollama_url: str = "http://localhost:11434",
    llm_api_key: str = "",
) -> str:
    """
    Chạy toàn bộ pipeline normalize (chưa chunk, chưa sanitize vocab).
    Dùng preprocess_and_chunk() nếu cần cả sanitize + chunk cho inference.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    dbg("PREP", f"INPUT: {text[:120]}")

    text = _normalize_linebreaks(text)
    dbg("PREP", f"after _normalize_linebreaks: {text[:120]}")

    text = _remove_parens(text)
    dbg("PREP", f"after _remove_parens: {text[:120]}")

    # ── Stage 1: số xử lý TRƯỚC LLM ─────────────────────────────────────────
    before_num = text
    text = normalize_numbers(text, dbg=dbg)
    if text != before_num:
        dbg("PREP", f"after normalize_numbers: {text[:120]}")

    before_vi = text
    text = vi_numbers(text)
    if text != before_vi:
        dbg("PREP", f"after vi_numbers: {text[:120]}")

    # ── Stage 2+3: LLM gate + normalize (chỉ viết tắt + context) ────────────
    if llm_model:
        normalized = llm_normalize(text, model=llm_model, url=ollama_url, api_key=llm_api_key)
        if normalized:
            text = normalized
            dbg("PREP", f"after LLM: {text[:120]}")
            print(f"[*] LLM normalized OK — {len(text)} chars")
        else:
            print("[WARN] LLM normalize thất bại — dùng rule-based fallback toàn bộ.")

    # ── Stage 4: safety net sau LLM — dict hành chính trước, rồi letter-spell ─
    before_acronym = text
    text = expand_acronyms_fallback(text)
    if text != before_acronym:
        dbg("PREP", f"after expand_acronyms_fallback: {text[:120]}")

    before_abbrev = text
    text = expand_abbrevs_fallback(text)
    if text != before_abbrev:
        dbg("PREP", f"after expand_abbrevs_fallback: {text[:120]}")

    text = text.lower()
    text = re.sub(r":\s+", ", ", text)
    text = re.sub(r"([^\W\d_])-([^\W\d_])", r"\1 \2", text)
    text = re.sub(r"([^\W\d_])/([^\W\d_])", r"\1 \2", text)
    text = re.sub(r"\s+-\s+", " ", text)
    text = re.sub(r"([.!?;,])\s+\.\s+", r"\1 ", text)  # double punct cleanup
    text = re.sub(r"  +", " ", text).strip()

    dbg("PREP", f"FINAL: {text[:120]}")
    return text


def preprocess_and_chunk(
    text: str,
    vocab_file: Path,
    llm_model: str | None = None,
    ollama_url: str = "http://localhost:11434",
    llm_api_key: str = "",
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[str]:
    """preprocess_text() + Stage 6 sanitize_for_vivoice() + normalize_for_f5() chunking."""
    normalized = preprocess_text(text, llm_model=llm_model, ollama_url=ollama_url, llm_api_key=llm_api_key)
    sanitized = sanitize_for_vivoice(normalized, vocab_file)
    if sanitized != normalized:
        dbg("PREP", f"after sanitize_for_vivoice: {sanitized[:120]}")
    return normalize_for_f5(sanitized, max_chars=max_chars)
