"""
Bước 7 — _sanity_ok() validation.

Detect khi LLM output bất thường (paraphrase, hallucination, dịch sang tiếng khác)
để rollback về câu gốc thay vì push garbage vào F5-TTS.
"""

import re

CJK_RE = re.compile(r"[一-鿿㐀-䶿]")

HAS_DIACRITIC = re.compile(
    r"[àáạảãăắằặẳẵâấầậẩẫèéẹẻẽêếềệểễìíịỉĩòóọỏõôốồổỗơớờợởỡùúụủũưứừựửữỳýỵỷỹđ]",
    re.IGNORECASE,
)


def contains_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text))


def sanity_ok(original: str, result: str, idx: int = 0, dbg=None) -> bool:
    """
    Kiểm tra output LLM không phá text:
      1. Không rỗng
      2. Không phình hơn 4x (LLM thêm giải thích thừa)
      3. Không mất quá 6% từ tiếng Việt có dấu
    """
    if not result.strip():
        if dbg:
            dbg("SANITY", f"[{idx}] FAIL: output rỗng")
        return False

    ratio = len(result) / max(len(original), 1)
    if ratio > 4.0:
        if dbg:
            dbg("SANITY", f"[{idx}] FAIL: phình {ratio:.1f}x")
        return False

    vi_orig = [w.lower() for w in original.split() if HAS_DIACRITIC.search(w)]
    if len(vi_orig) < 4:
        return True

    result_words = result.lower().split()
    missing = sum(
        max(0, vi_orig.count(w) - result_words.count(w))
        for w in set(vi_orig)
    )
    lost_ratio = missing / len(vi_orig)
    if lost_ratio > 0.06:
        if dbg:
            dbg("SANITY", f"[{idx}] FAIL: mất {missing}/{len(vi_orig)} từ VI ({lost_ratio:.0%})")
        return False
    return True
