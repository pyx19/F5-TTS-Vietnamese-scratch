"""
sanitize_for_vivoice() — strip ký tự ngoài vocab của model ViVoice.

vocab.txt liệt kê mỗi token được phép trên một dòng (đa số là ký tự đơn).
Ký tự lạ lọt vào input (control chars, emoji, ký hiệu hiếm...) có thể tạo
artifact âm thanh — strip trước khi đưa vào F5-TTS, chạy SAU khi mọi bước
normalize khác đã xong.
"""

from pathlib import Path

_allowed_chars: set[str] | None = None


def _load_allowed_chars(vocab_file: Path) -> set[str]:
    chars: set[str] = {" ", "\n"}
    with open(vocab_file, "r", encoding="utf-8") as f:
        for line in f:
            token = line.rstrip("\n")
            if len(token) == 1:
                chars.add(token)
    return chars


def _get_allowed_chars(vocab_file: Path) -> set[str]:
    global _allowed_chars
    if _allowed_chars is None:
        _allowed_chars = _load_allowed_chars(vocab_file)
    return _allowed_chars


def sanitize_for_vivoice(text: str, vocab_file: Path) -> str:
    """Thay mọi ký tự không có trong vocab bằng khoảng trắng, rồi gộp khoảng trắng thừa."""
    allowed = _get_allowed_chars(vocab_file)
    out = "".join(c if c in allowed else " " for c in text)
    return " ".join(out.split())
