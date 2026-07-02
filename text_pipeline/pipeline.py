"""
preprocess_text() — orchestrator cho toàn bộ pipeline mô tả trong plan.md:

  Raw Text
     -> _normalize_linebreaks() / _normalize_symbols() / _remove_brackets()
        (dọn cấu trúc + ký hiệu trước mọi bước)
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
    - Bỏ ký hiệu gạch đầu dòng ("-", "•", "*") ở đầu dòng — đây là marker cấu
      trúc danh sách, không phải nội dung cần đọc thành tiếng ("trừ", "sao"...).
    - Mỗi dòng không kết thúc bằng dấu câu → thêm dấu chấm
    - Tab → space
    - Dòng trống → bỏ qua
    Phải chạy TRƯỚC mọi bước khác (kể cả LLM) để giữ đúng cấu trúc.
    """
    out = []
    for line in text.split("\n"):
        line = re.sub(r"\t+", " ", line).strip()
        line = re.sub(r"^[-•*]\s+", "", line)
        if not line:
            continue
        if line[-1] not in ".!?,;:":
            line += "."
        out.append(line)
    return " ".join(out)


def _normalize_symbols(text: str) -> str:
    """
    Chuẩn hóa ký hiệu hay gây đọc sai/díu vào nhau nếu chỉ xóa trơn:
      P&G, R&D, AT&T  → PG, RD, ATT  (nối liền, không tách rời — chữ hoa đơn lẻ
                         như "P", "G" không khớp ALLCAPS_RE (cần 2-8 ký tự) nên
                         bị hạ thường vô nghĩa; nối liền để expand_abbrevs_fallback()
                         sau LLM phiên âm cả cụm như 1 viết tắt duy nhất)
      &        → " và "  (còn lại, giữa từ/cụm từ thường — vd "A & B Company")
      " ' “ ” ‘ ’ → xóa  (thuật ngữ trong ngoặc kép đọc thẳng, không cần ngắt
                           nghỉ riêng như (), {} — chỉ 1 từ/cụm ngắn)
      ...      → "," (giữa câu, còn nội dung theo sau) hoặc "." (cuối câu/dòng)
                 — giữ nguyên chuỗi 2+ dấu chấm dễ đọc thành nhiều nhịp dừng lạ.
    Chạy TRƯỚC _remove_brackets() để "&" bên trong ngoặc cũng được xử lý
    (vd "(R&D)" → "(RD)" → ", RD,").
    """
    text = re.sub(r"\b([A-Z]{1,4})&([A-Z]{1,4})\b", r"\1\2", text)
    text = re.sub(r"\s*&\s*", " và ", text)
    text = re.sub(r"['\"‘’“”]", "", text)
    text = re.sub(r"\.{2,}(?=\s*\S)", ",", text)  # còn nội dung theo sau -> phẩy
    text = re.sub(r"\.{2,}", ".", text)            # cuối câu/dòng -> chấm
    return re.sub(r"  +", " ", text).strip()


def _remove_brackets(text: str) -> str:
    """
    Xóa dấu ngoặc ()/{}, CHÈN DẤU PHẨY quanh nội dung để tạo ngắt nghỉ.
    TTS model đọc ký tự '(' thành âm ký sinh ("a", "ở", "e"...) — chỉ xóa ngoặc mà
    không chèn dấu câu khiến nội dung chú thích bị đọc díu liền vào câu chính, không
    có pause hợp lý (đúng như người đọc thật sẽ ngừng một nhịp trước/sau chú thích).
      (Cloud Computing) → , Cloud Computing,
      (ZTA)             → , ZTA,  (LLM xử lý tiếp phần viết tắt)
      {Data Silos}      → , Data Silos,
    Ngoặc lẻ không có cặp bị xóa thẳng (không đủ ngữ cảnh để chèn dấu phẩy).
    """
    text = re.sub(r"\(([^)]*)\)", r", \1,", text)
    text = re.sub(r"\{([^}]*)\}", r", \1,", text)
    text = re.sub(r"[(){}]", "", text)
    # Dọn dấu phẩy thừa sinh ra khi nội dung trong ngoặc đứng cạnh dấu câu khác
    text = re.sub(r",\s*,+", ",", text)          # ",," -> ","
    text = re.sub(r",\s*([.!?;:])", r"\1", text)  # ", ." -> "."
    text = re.sub(r"^\s*,\s*", "", text)          # phẩy thừa đầu câu
    text = re.sub(r",\s*$", "", text)             # phẩy thừa cuối câu
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

    text = _normalize_symbols(text)
    dbg("PREP", f"after _normalize_symbols: {text[:120]}")

    text = _remove_brackets(text)
    dbg("PREP", f"after _remove_brackets: {text[:120]}")

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
    text = re.sub(r":\s*", ", ", text)  # \s* (không phải \s+) — bắt cả ":" dính liền chữ sau
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
