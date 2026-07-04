"""
preprocess_text() — orchestrator cho toàn bộ pipeline mô tả trong plan.md:

  Raw Text
     -> _normalize_linebreaks() / _normalize_symbols() / _remove_brackets()
        (dọn cấu trúc + ký hiệu trước mọi bước)
     -> [Stage 1] normalize_numbers() + vi_numbers()  (rule-based, TRƯỚC Qwen)
     -> expand_acronyms_fallback()                     (dict hành chính cứng, TRƯỚC Qwen —
                                                         cùng lý do với số: tránh Qwen "sửa"
                                                         sai viết tắt vốn đã rõ nghĩa)
     -> [Stage 2] needs_llm() gate                    (bên trong llm_normalize)
     -> [Stage 3] llm_normalize() / Qwen               (per-sentence, chỉ còn viết tắt kỹ
                                                         thuật + context, có sanity_ok + cache)
     -> expand_known_abbrevs_case_insensitive()        (bắt token ALL-CAPS đã biết bị LLM
                                                         hạ thường mà không spell, vd "ZTA"
                                                         -> LLM trả "zta" -> spell lại)
     -> [Stage 4] expand_abbrevs_fallback()            (safety net sau Qwen — spell nốt
                                                         ALL-CAPS kỹ thuật còn sót)
     -> lowercase + cleanup ký tự nối
     -> [Stage 6] sanitize_for_vivoice()               (strip ký tự ngoài vocab, cần vocab_file)
     -> normalize_for_f5()                             (chunk <= max_chars cho F5-TTS)
"""

import re
from pathlib import Path

from .acronyms import expand_acronyms_fallback
from .chunking import DEFAULT_MAX_CHARS, PARAGRAPH_BREAK_MARKER, normalize_for_f5
from .debug import dbg
from .letters import ALLCAPS_RE, expand_abbrevs_fallback, expand_known_abbrevs_case_insensitive
from .llm_normalize import llm_normalize
from .numbers import normalize_numbers, vi_numbers
from .sanitize import sanitize_for_vivoice


def _split_into_paragraphs(text: str) -> list[str]:
    """
    Tách text thành list các ĐOẠN VĂN, ngăn cách bởi dòng trống trong text gốc.
    Trong mỗi đoạn:
    - Bỏ ký hiệu gạch đầu dòng ("-", "•", "*") ở đầu dòng — đây là marker cấu
      trúc danh sách, không phải nội dung cần đọc thành tiếng ("trừ", "sao"...).
    - Mỗi dòng không kết thúc bằng dấu câu → thêm dấu chấm.
    - Tab → space.
    Cuối mỗi đoạn: ép kết thúc bằng DẤU CHẤM, bất kể dòng cuối vốn kết bằng dấu
    gì (",", ";", ":") — dòng trống là ranh giới đoạn/section mạnh, phẩy/hai
    chấm quá yếu cho ranh giới lớn như vậy (quan sát thực tế: "...Báo cáo số
    45/BC-BTP, <dòng trống> QUYẾT NGHỊ:" đọc dồn/mất ngắt nghỉ đúng chỗ cần dừng
    nhất). Phải chạy TRƯỚC mọi bước khác (kể cả LLM) để giữ đúng cấu trúc — mỗi
    đoạn sau đó được xử lý ĐỘC LẬP qua toàn bộ pipeline (xem preprocess_text()).
    """
    paragraphs: list[str] = []
    current: list[str] = []
    for line in text.split("\n"):
        line = re.sub(r"\t+", " ", line).strip()
        line = re.sub(r"^[-•*]\s+", "", line)
        if not line:
            if current:
                paragraphs.append(_finalize_paragraph_lines(current))
                current = []
            continue
        if line[-1] not in ".!?,;:":
            line += "."
        current.append(line)
    if current:
        paragraphs.append(_finalize_paragraph_lines(current))
    return paragraphs


def _finalize_paragraph_lines(lines: list[str]) -> str:
    joined = " ".join(lines)
    joined = re.sub(r"[,;:]+$", "", joined).rstrip()
    if joined and joined[-1] not in ".!?":
        joined += "."
    return joined


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
    Xóa dấu ngoặc ()/{}, CHÈN DẤU CÂU quanh nội dung để tạo ngắt nghỉ.
    TTS model đọc ký tự '(' thành âm ký sinh ("a", "ở", "e"...) — chỉ xóa ngoặc mà
    không chèn dấu câu khiến nội dung chú thích bị đọc díu liền vào câu chính, không
    có pause hợp lý (đúng như người đọc thật sẽ ngừng một nhịp trước/sau chú thích).

    Mở ngoặc → DẤU PHẨY (nghỉ nhẹ trước chú thích). Đóng ngoặc → DẤU CHẤM PHẨY
    (mạnh hơn phẩy) — quan sát thực tế: dùng phẩy ở cả 2 đầu vẫn nghe "hơi nhanh"
    ngay sau khi đóng ngoặc, vì cả câu vẫn nằm trong 1 chunk/synthesis call liên
    tục. Chấm phẩy được `chunking._SENTENCE_SPLIT_RE` coi là ranh giới CHUNK thật
    sự (như dấu chấm) — buộc phần còn lại của câu sang 1 lần tổng hợp riêng, đảm
    bảo có khoảng nghỉ (crossfade giữa 2 chunk) thay vì chỉ trông chờ vào việc
    model tự đọc dấu phẩy đủ chậm.
      (Cloud Computing) → , Cloud Computing;
      (ZTA)             → , ZTA;  (LLM xử lý tiếp phần viết tắt)
      {Data Silos}      → , Data Silos;
    Ngoặc lẻ không có cặp bị xóa thẳng (không đủ ngữ cảnh để chèn dấu câu).
    """
    text = re.sub(r"\(([^)]*)\)", r", \1;", text)
    text = re.sub(r"\{([^}]*)\}", r", \1;", text)
    text = re.sub(r"[(){}]", "", text)
    # Dọn dấu câu thừa sinh ra khi nội dung trong ngoặc đứng cạnh dấu câu khác —
    # dấu MẠNH HƠN luôn thắng (chấm/hỏi/than thắng chấm phẩy, chấm phẩy thắng phẩy).
    text = re.sub(r",\s*,+", ",", text)              # ",," -> ","
    text = re.sub(r";\s*;+", ";", text)              # ";;" -> ";"
    text = re.sub(r";\s*,+", ";", text)              # ";," -> ";" (mã đóng ngoặc rồi gặp phẩy có sẵn trong text gốc)
    text = re.sub(r",\s*([.!?;:])", r"\1", text)      # ", ." / ", ;" -> "." / ";"
    text = re.sub(r";\s*([.!?])", r"\1", text)        # "; ." -> "."
    text = re.sub(r"^\s*[,;]\s*", "", text)           # phẩy/chấm phẩy thừa đầu câu
    text = re.sub(r"[,;]\s*$", "", text)              # phẩy/chấm phẩy thừa cuối câu
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

    Text được tách thành các ĐOẠN VĂN (ngăn bởi dòng trống), mỗi đoạn chạy qua
    TOÀN BỘ pipeline (kể cả LLM) ĐỘC LẬP, rồi nối lại bằng PARAGRAPH_BREAK_MARKER
    — marker được chèn SAU CÙNG nên không bao giờ phải đi qua LLM, tránh rủi ro
    bị Qwen làm hỏng/xóa mất. normalize_for_f5() sẽ tách lại theo marker này để
    biết chỗ nào cần chèn thêm khoảng lặng mạnh hơn ranh giới câu thường.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    dbg("PREP", f"INPUT: {text[:120]}")

    paragraphs = _split_into_paragraphs(text)
    processed = [
        _process_paragraph(p, llm_model=llm_model, ollama_url=ollama_url, llm_api_key=llm_api_key)
        for p in paragraphs
    ]
    result = f" {PARAGRAPH_BREAK_MARKER} ".join(p for p in processed if p)
    dbg("PREP", f"FINAL: {result[:120]}")
    return result


def _process_paragraph(
    text: str,
    llm_model: str | None,
    ollama_url: str,
    llm_api_key: str,
) -> str:
    """Chạy toàn bộ pipeline normalize trên 1 ĐOẠN VĂN đơn lẻ (đã tách bởi _split_into_paragraphs)."""
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

    # ── Viết tắt hành chính (dict cứng, không mơ hồ) — TRƯỚC LLM, giống số ───
    # Qwen2.5:3B đôi khi "sửa" sai một viết tắt đã có nghĩa rõ ràng (đã thấy qua
    # audit thật — vd bỏ sót hoặc biến dạng "36-NQ/TW"). Vì fallback chạy SAU LLM
    # chỉ vá được chỗ LLM bỏ sót chứ không sửa lại được chỗ LLM đã biến dạng text
    # gốc, nên chuyển lên đây để LLM không bao giờ thấy các viết tắt này nữa.
    before_acronym = text
    text = expand_acronyms_fallback(text)
    if text != before_acronym:
        dbg("PREP", f"after expand_acronyms_fallback: {text[:120]}")

    # ── Stage 2+3: LLM gate + normalize (chỉ viết tắt kỹ thuật + context) ───
    if llm_model:
        # Ghi nhớ các token ALL-CAPS TRƯỚC khi vào LLM — dùng để bắt lại trường
        # hợp LLM hạ thường mà không spell (vd "ZTA" -> LLM trả về "zta" nguyên
        # văn, đọc díu vì không còn khớp ALL-CAPS cho fallback thường).
        known_allcaps = set(ALLCAPS_RE.findall(text))

        normalized = llm_normalize(text, model=llm_model, url=ollama_url, api_key=llm_api_key)
        if normalized:
            text = normalized
            dbg("PREP", f"after LLM: {text[:120]}")
            print(f"[*] LLM normalized OK — {len(text)} chars")

            before_known = text
            text = expand_known_abbrevs_case_insensitive(text, known_allcaps)
            if text != before_known:
                dbg("PREP", f"after expand_known_abbrevs_case_insensitive: {text[:120]}")
        else:
            print("[WARN] LLM normalize thất bại — dùng rule-based fallback toàn bộ.")

    # ── Stage 4: safety net sau LLM — spell nốt ALL-CAPS kỹ thuật còn sót ────
    before_abbrev = text
    text = expand_abbrevs_fallback(text)
    if text != before_abbrev:
        dbg("PREP", f"after expand_abbrevs_fallback: {text[:120]}")

    text = text.lower()
    text = re.sub(r":\s*", ", ", text)  # \s* (không phải \s+) — bắt cả ":" dính liền chữ sau
    text = re.sub(r"([^\W\d_])-([^\W\d_])", r"\1 \2", text)
    text = re.sub(r"([^\W\d_])/([^\W\d_])", r"\1 \2", text)
    # Gạch ngang có khoảng trắng bao quanh (" - ") đóng vai trò ngoặc đơn/chú
    # thích (vd "User Experience - UX"), KHÁC với gạch nối compound-word đã xử lý
    # ở dòng trên (không có khoảng trắng, "on-premise" → "on premise"). Trước đây
    # thu về 1 khoảng trắng trơn khiến TTS đọc díu không phân biệt được ranh giới
    # cụm từ gốc và phần chú thích/viết tắt theo sau — giờ chèn dấu phẩy để tạo pause.
    text = re.sub(r"\s+-\s+", ", ", text)
    text = re.sub(r",\s*,+", ",", text)  # dọn phẩy thừa nếu đứng cạnh dấu phẩy khác
    text = re.sub(r",\s*([.!?;:])", r"\1", text)
    text = re.sub(r"([.!?;,])\s+\.\s+", r"\1 ", text)  # double punct cleanup
    text = re.sub(r"  +", " ", text).strip()

    return text


def preprocess_and_chunk(
    text: str,
    vocab_file: Path,
    llm_model: str | None = None,
    ollama_url: str = "http://localhost:11434",
    llm_api_key: str = "",
    max_chars: int = DEFAULT_MAX_CHARS,
) -> tuple[list[str], set[int]]:
    """preprocess_text() + Stage 6 sanitize_for_vivoice() + normalize_for_f5() chunking."""
    normalized = preprocess_text(text, llm_model=llm_model, ollama_url=ollama_url, llm_api_key=llm_api_key)
    sanitized = sanitize_for_vivoice(normalized, vocab_file)
    if sanitized != normalized:
        dbg("PREP", f"after sanitize_for_vivoice: {sanitized[:120]}")
    return normalize_for_f5(
        sanitized, max_chars=max_chars,
        llm_model=llm_model, ollama_url=ollama_url, llm_api_key=llm_api_key,
    )
