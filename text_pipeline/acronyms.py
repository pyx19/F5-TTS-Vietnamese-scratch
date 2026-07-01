"""
Bước 6 — _expand_abbrevs_fallback() sau LLM: ACRONYM_DICT cho viết tắt hành chính.

Đây là safety net thứ nhất: nếu Qwen miss (không expand "UBND" → "ủy ban nhân dân")
hoặc Ollama down hoàn toàn, rule-based vẫn catch được các viết tắt hành chính phổ
biến nhất. Phải chạy TRƯỚC letters.expand_abbrevs_fallback() (spell letter-by-letter),
nếu không "UBND" sẽ bị đọc thành "u bi en đi" thay vì "ủy ban nhân dân".

Match CASE-INSENSITIVE: quan sát thực tế khi test với Qwen2.5:3b thật cho thấy model
đôi khi bỏ sót việc expand ("36-NQ/TW" → trả về "nq tw" viết thường thay vì "nghị quyết
trung ương") — nếu fallback chỉ match chữ hoa chính xác thì sẽ không bắt được các
trường hợp này, làm mất tác dụng "safety net". Match theo thứ tự dài→ngắn để tránh
khớp nhầm một phần của token dài hơn (vd tránh "TT" khớp bên trong "TTg" trước khi
"TTg" được xử lý).
"""

import json
import re
from pathlib import Path

_DICT_PATH = Path(__file__).parent / "acronym_dict.json"

with open(_DICT_PATH, "r", encoding="utf-8") as f:
    ACRONYM_DICT: dict[str, str] = json.load(f)

# Lookup case-insensitive: chuẩn hóa key về upper() để so khớp, giữ dict gốc cho output.
_UPPER_TO_VALUE = {k.upper(): v for k, v in ACRONYM_DICT.items()}

_KEYS_BY_LENGTH_DESC = sorted(ACRONYM_DICT, key=len, reverse=True)
_ACRONYM_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _KEYS_BY_LENGTH_DESC) + r")\b",
    re.IGNORECASE,
)


def expand_acronyms_fallback(text: str) -> str:
    """Thay thế các token khớp trong ACRONYM_DICT, không phân biệt hoa/thường."""
    return _ACRONYM_RE.sub(lambda m: _UPPER_TO_VALUE[m.group(1).upper()], text)
