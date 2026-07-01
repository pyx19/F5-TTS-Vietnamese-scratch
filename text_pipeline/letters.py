"""
Bước 3 — Bảng phiên âm chữ cái tiếng Anh (American English) đọc bằng giọng Việt.

_EN_LETTER là single source of truth: cả rule-based fallback (_expand_abbrevs_fallback)
lẫn prompt LLM (LETTER_TABLE, xem llm_normalize.py) đều sinh ra từ dict này, để tránh
hai nơi có hai bảng phiên âm mâu thuẫn nhau.
"""

import re

EN_LETTER = {
    "A": "ây", "B": "bi", "C": "xi", "D": "đi",
    "E": "i", "F": "ép", "G": "gờ", "H": "ếch",
    "I": "ai", "J": "giây", "K": "cây", "L": "eo",
    "M": "em", "N": "en", "O": "âu", "P": "pi",
    "Q": "kiu", "R": "a rờ", "S": "ét", "T": "ti",
    "U": "iu", "V": "vi", "W": "đâp bờ liu", "X": "ích",
    "Y": "uai", "Z": "zi",
    "Đ": "đê",  # chữ Việt xuất hiện trong viết tắt hành chính (NĐ-CP)
}

# Sinh bảng dạng "A=ây  B=bi  C=xi ..." để nhúng vào prompt LLM — đồng bộ 100% với code.
LETTER_TABLE = "  ".join(f"{k}={v}" for k, v in EN_LETTER.items())

# Khớp viết tắt ALL-CAPS (dùng cho rule-based fallback và gate LLM call)
ALLCAPS_RE = re.compile(r"\b[A-ZĐĂÂÊÔƯƠ]{2,8}\b")


def expand_abbrevs_fallback(text: str) -> str:
    """
    Rule-based fallback khi LLM không khả dụng hoặc bỏ sót.
    Chỉ xử lý ALL-CAPS đơn giản theo EN_LETTER — không cố gắng hiểu context.
    Phải chạy TRƯỚC lowercase.
    """

    def _en_spell(m: re.Match) -> str:
        return " ".join(EN_LETTER.get(c, c) for c in m.group())

    return ALLCAPS_RE.sub(_en_spell, text)
