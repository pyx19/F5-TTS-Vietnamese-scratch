"""
Bước 9 — normalize_for_f5(): chunk text đã normalize thành các đoạn ≤ max_chars
để đưa vào F5-TTS/ViVoice (model hoạt động tốt nhất với chunk ngắn).

Thứ tự ưu tiên điểm cắt:
  1. Ranh giới câu: . ! ? … ;
  2. Nếu 1 câu vẫn > max_chars và không còn dấu câu: cắt tại dấu phẩy
  3. Nếu vẫn > max_chars và KHÔNG còn dấu câu/phẩy nào: nhờ LLM đề xuất điểm
     ngắt ngữ nghĩa (chunk_llm.py) — TÙY CHỌN, chỉ dùng khi có llm_model. Có
     validate nghiêm ngặt, sai thì fallback bước 4.
  4. Cắt tại khoảng trắng gần nhất trước max_chars (không bao giờ cắt giữa
     một từ/số đã expand) — luôn là fallback cuối cùng, không phụ thuộc LLM.

QUAN TRỌNG: việc GỘP (pack) các đoạn ngắn lại cho gần đầy max_chars CHỈ diễn ra
TRONG PHẠM VI 1 câu, không bao giờ gộp đuôi câu này với đầu câu sau — tránh 1
chunk chứa nội dung của 2 câu khác nhau (đã quan sát: gây đọc dồn/nhanh vì F5-TTS
dự đoán duration tuyến tính theo số ký tự, không biết đây là ranh giới câu). Dự
án ưu tiên accuracy hơn latency (không streaming) nên chấp nhận nhiều chunk nhỏ
hơn là gộp tối đa để giảm số lượng — xem memory "accuracy-over-latency".
"""

import re

DEFAULT_MAX_CHARS = 200

# Marker ranh giới ĐOẠN VĂN (paragraph, ngăn cách bởi dòng trống trong text gốc)
# — chèn bởi pipeline.preprocess_text() SAU KHI đã xử lý xong toàn bộ pipeline
# (kể cả LLM) trên từng đoạn riêng lẻ, nên marker này không bao giờ phải "sống
# sót" qua 1 lệnh gọi LLM. Chỉ dùng chữ thường a-z để chắc chắn qua được vocab
# whitelist của sanitize_for_vivoice() và không đụng độ với acronym-dict/case-
# insensitive lookup nào khác. normalize_for_f5() tách lại theo marker này để
# biết ranh giới đoạn nào cần chèn THÊM khoảng lặng lúc synthesis (mạnh hơn ranh
# giới câu thường — xem infer_vivoice.py).
PARAGRAPH_BREAK_MARKER = "zzzparabreakzzz"
_PARAGRAPH_SPLIT_RE = re.compile(r"\s*" + re.escape(PARAGRAPH_BREAK_MARKER) + r"\s*")

# Chunk ngắn hơn ngưỡng này (vd tiêu đề "Điều 2.", "CHÍNH PHỦ.") dễ bị F5-TTS đọc
# dồn/cụt nếu đứng riêng — model dự đoán duration tuyến tính theo số ký tự, đoạn
# quá ngắn không đủ "chỗ" để có ngắt nghỉ tự nhiên trước/sau. Gộp về phía sau
# (_merge_tiny_chunks) thay vì để đứng 1 mình.
_MIN_CHUNK_CHARS = 20

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…;])\s+")
_COMMA_SPLIT_RE = re.compile(r"(?<=,)\s+")


def _split_on(pattern: re.Pattern, text: str) -> list[str]:
    return [p.strip() for p in pattern.split(text) if p.strip()]


def _split_long_piece(
    piece: str,
    max_chars: int,
    llm_model: str | None = None,
    ollama_url: str = "",
    llm_api_key: str = "",
) -> list[str]:
    if len(piece) <= max_chars:
        return [piece]

    # Thử cắt tại dấu phẩy trước.
    comma_parts = _split_on(_COMMA_SPLIT_RE, piece)
    if len(comma_parts) > 1:
        chunks = []
        for part in comma_parts:
            chunks.extend(_split_long_piece(part, max_chars, llm_model, ollama_url, llm_api_key))
        return _pack(chunks, max_chars)

    # Không còn dấu câu/phẩy nào — thử nhờ LLM đề xuất điểm ngắt ngữ nghĩa
    # trước khi phải cắt mù theo khoảng trắng (dễ tách rời giữa 1 cụm từ).
    if llm_model:
        from .chunk_llm import llm_split_points

        llm_parts = llm_split_points(piece, max_chars, llm_model, ollama_url, api_key=llm_api_key)
        if llm_parts:
            chunks = []
            for part in llm_parts:
                chunks.extend(_split_long_piece(part, max_chars, llm_model, ollama_url, llm_api_key))
            return _pack(chunks, max_chars)

    # Fallback cuối cùng — cắt tại khoảng trắng gần nhất trước max_chars.
    words = piece.split(" ")
    chunks, current = [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_chars and current:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _pack(pieces: list[str], max_chars: int) -> list[str]:
    """Gộp các đoạn ngắn liền kề lại cho đầy max_chars, tránh chunk quá vụn."""
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}".strip() if current else piece
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks


def _merge_tiny_chunks(chunks: list[str], max_chars: int, min_chars: int = _MIN_CHUNK_CHARS) -> list[str]:
    """
    Gộp chunk quá ngắn (< min_chars, vd tiêu đề "điều hai.", "chính phủ.") vào
    chunk KẾ TIẾP — khác với việc KHÔNG gộp xuyên câu ở normalize_for_f5(), đây là
    ngoại lệ có chủ đích: 1 tiêu đề/nhãn đứng riêng quá ngắn để đọc tự nhiên một
    mình, nhưng gộp với câu/tiêu đề theo ngay sau nó (thường liên quan trực tiếp,
    vd "Điều 2." + "Các Nhiệm vụ...") vẫn hợp lý về ngữ nghĩa và nghe tự nhiên hơn.
    """
    if not chunks:
        return chunks
    merged: list[str] = []
    i = 0
    while i < len(chunks):
        current = chunks[i]
        while len(current) < min_chars and i + 1 < len(chunks):
            candidate = f"{current} {chunks[i + 1]}".strip()
            if len(candidate) > max_chars:
                break
            current = candidate
            i += 1
        merged.append(current)
        i += 1
    return merged


def normalize_for_f5(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    llm_model: str | None = None,
    ollama_url: str = "",
    llm_api_key: str = "",
) -> tuple[list[str], set[int]]:
    """
    Chunk văn bản đã normalize thành list các đoạn ≤ max_chars cho F5-TTS.

    llm_model tùy chọn: nếu set, dùng LLM hỗ trợ chọn điểm ngắt ngữ nghĩa cho
    các đoạn quá dài mà không còn dấu câu/phẩy (xem chunk_llm.py). Không set
    thì chunking hoạt động y hệt như trước (thuần rule-based).

    Trả về (chunks, paragraph_break_after): paragraph_break_after là tập hợp
    các INDEX trong `chunks` mà ngay SAU chunk đó là một ranh giới đoạn văn
    (PARAGRAPH_BREAK_MARKER) — dùng để chèn thêm khoảng lặng lúc synthesis,
    mạnh hơn ranh giới câu thường (xem infer_vivoice.py).
    """
    if not text.strip():
        return [], set()

    all_chunks: list[str] = []
    paragraph_break_after: set[int] = set()
    paragraphs = [p for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()]
    for para in paragraphs:
        para = " ".join(para.split())

        # QUAN TRỌNG: pack() theo TỪNG CÂU riêng biệt, KHÔNG gộp chung tất cả
        # pieces của mọi câu lại rồi mới pack — nếu không, đuôi câu này có thể
        # bị nhét chung 1 chunk với đầu câu tiếp theo (đã quan sát thực tế: 1
        # chunk chứa cả "...nhà nước." + toàn bộ câu sau, khiến F5-TTS đọc
        # dồn/nhanh vì duration dự đoán theo số ký tự, không phân biệt được đây
        # là 2 câu khác nhau). Dự án này ưu tiên accuracy hơn latency (không
        # cần streaming) nên chấp nhận có nhiều chunk nhỏ hơn thay vì gộp tối
        # đa để giảm số lượng.
        para_chunks: list[str] = []
        for sentence in _split_on(_SENTENCE_SPLIT_RE, para):
            pieces = _split_long_piece(sentence, max_chars, llm_model, ollama_url, llm_api_key)
            para_chunks.extend(_pack(pieces, max_chars))
        para_chunks = _merge_tiny_chunks(para_chunks, max_chars)

        if not para_chunks:
            continue
        all_chunks.extend(para_chunks)
        paragraph_break_after.add(len(all_chunks) - 1)

    # Ranh giới sau chunk CUỐI CÙNG là vô nghĩa — không có gì theo sau để cần
    # khoảng lặng thêm.
    paragraph_break_after.discard(len(all_chunks) - 1)
    return all_chunks, paragraph_break_after
