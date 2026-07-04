"""
Bước 3 — Bảng phiên âm chữ cái (English-style + Vietnamese-style) đọc bằng giọng Việt.

_EN_LETTER là single source of truth cho viết tắt TIẾNG ANH (AI, CPU, GPU...): cả
rule-based fallback (_expand_abbrevs_fallback) lẫn prompt LLM (LETTER_TABLE, xem
llm_normalize.py) đều sinh ra từ dict này.

VN_LETTER là bảng riêng cho MÃ VĂN BẢN HÀNH CHÍNH (NĐ-CP, QH15, NQ/TW...) — đọc theo
tên chữ cái tiếng Việt ("nờ, đê, xê, pê"), KHÁC với cách đọc tiếng Anh ("en, đi, xi,
pi"). Dùng bởi numbers.py._normalize_doc_codes() — xem docstring ở đó để biết lý do
mã văn bản cần đọc kiểu "ký hiệu" (spell chữ cái) thay vì dịch ra ngữ nghĩa.
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


def expand_known_abbrevs_case_insensitive(text: str, known_tokens: set[str]) -> str:
    """
    Safety net riêng cho trường hợp: LLM HẠ THƯỜNG một viết tắt đã biết là ALL-CAPS
    TRƯỚC khi vào LLM, thay vì spell đúng (quan sát thực tế: "ZTA" → LLM trả về
    "zta" nguyên văn, không spell thành "zi ti ây" — vì lowercase nên
    expand_abbrevs_fallback() (yêu cầu ALL-CAPS) không bắt được, TTS đọc díu
    "zta" như 1 từ vô nghĩa).

    CHỈ match case-insensitive với các token đã XÁC NHẬN là ALL-CAPS trong text
    TRƯỚC khi gọi LLM (truyền vào qua known_tokens) — an toàn, không đụng tới các
    từ tiếng Anh thường khác mà LLM cố tình giữ nguyên (deploy, server, model...).
    """
    if not known_tokens:
        return text
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(t) for t in sorted(known_tokens, key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    )

    def _spell(m: re.Match) -> str:
        return " ".join(EN_LETTER.get(c, c) for c in m.group().upper())

    return pattern.sub(_spell, text)


# Bảng chữ cái tiếng Việt — dùng riêng cho mã văn bản pháp luật (numbers.py).
VN_LETTER = {
    "A": "a", "Ă": "á", "Â": "ớ",
    "B": "bê", "C": "xê", "D": "dê", "Đ": "đê",
    "E": "e", "Ê": "ê",
    "F": "ép", "G": "gờ", "H": "hát",
    "I": "i", "J": "gi", "K": "ca",
    "L": "lờ", "M": "mờ", "N": "nờ",
    "O": "o", "Ô": "ô", "Ơ": "ơ",
    "P": "pê", "Q": "quy", "R": "rờ",
    "S": "sờ", "T": "tê",
    "U": "u", "Ư": "ư",
    "V": "vê", "W": "vê kép",
    "X": "ích", "Y": "i dài", "Z": "dét",
}


def spell_vn_letters(text: str) -> str:
    """Phiên âm từng chữ cái theo tên gọi tiếng Việt (dùng cho mã văn bản, không phải viết tắt tiếng Anh)."""
    return " ".join(VN_LETTER.get(c.upper(), c) for c in text if c.strip())
