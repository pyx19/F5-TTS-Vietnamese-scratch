"""
Bước 9 — normalize_for_f5(): chunk text đã normalize thành các đoạn ≤ max_chars
để đưa vào F5-TTS/ViVoice (model hoạt động tốt nhất với chunk ngắn).

Thứ tự ưu tiên điểm cắt:
  1. Ranh giới câu: . ! ? … ;
  2. Nếu 1 câu vẫn > max_chars và không còn dấu câu: cắt tại dấu phẩy
  3. Nếu vẫn > max_chars: cắt tại khoảng trắng gần nhất trước max_chars
     (không bao giờ cắt giữa một từ/số đã expand).
"""

import re

DEFAULT_MAX_CHARS = 200

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…;])\s+")
_COMMA_SPLIT_RE = re.compile(r"(?<=,)\s+")


def _split_on(pattern: re.Pattern, text: str) -> list[str]:
    return [p.strip() for p in pattern.split(text) if p.strip()]


def _split_long_piece(piece: str, max_chars: int) -> list[str]:
    if len(piece) <= max_chars:
        return [piece]

    # Thử cắt tại dấu phẩy trước.
    comma_parts = _split_on(_COMMA_SPLIT_RE, piece)
    if len(comma_parts) > 1:
        chunks = []
        for part in comma_parts:
            chunks.extend(_split_long_piece(part, max_chars))
        return _pack(chunks, max_chars)

    # Không còn dấu câu nào — cắt tại khoảng trắng gần nhất trước max_chars.
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


def normalize_for_f5(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """Chunk văn bản đã normalize thành list các đoạn ≤ max_chars cho F5-TTS."""
    text = " ".join(text.split())
    if not text:
        return []

    sentences = _split_on(_SENTENCE_SPLIT_RE, text)
    pieces: list[str] = []
    for sentence in sentences:
        pieces.extend(_split_long_piece(sentence, max_chars))

    return _pack(pieces, max_chars)
