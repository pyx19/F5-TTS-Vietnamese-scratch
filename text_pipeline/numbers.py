"""
Bước 2 — normalize_numbers() chạy TRƯỚC LLM.

Root cause của hallucination phổ biến nhất: Qwen2.5-3B không reliable với số
phức tạp (vd: "15.5 tỷ USD" → LLM tự bịa "15% 50 triệu"). Bằng cách xử lý số
bằng regex xong hết TRƯỚC khi đưa câu vào LLM, Qwen không bao giờ thấy raw
numbers nữa — chỉ còn phải xử lý viết tắt/ký hiệu chữ.
"""

import re

from .letters import EN_LETTER

QUARTER_VI = {1: "một", 2: "hai", 3: "ba", 4: "bốn"}

ROMAN_VI = {
    "XII": "mười hai", "VIII": "tám", "VII": "bảy", "XI": "mười một",
    "III": "ba", "IV": "tư", "VI": "sáu", "IX": "chín",
    "II": "hai", "X": "mười", "V": "năm", "I": "một",
}
ROMAN_PATTERN = re.compile(
    r"(?i)\b(quý|điều|chương|phần)\s+"
    r"(XII|VIII|VII|XI|III|IV|VI|IX|II|X|V|I)\b"
)

MOBILE_GEN = {"2": "hai", "3": "ba", "4": "bốn", "5": "năm"}

# Match số nguyên standalone — không match số nằm trong từ như "F5" hay "3090s"
DIGIT_RE = re.compile(r"(?<![A-Za-z\d])\d+(?![A-Za-z\d])")


def _normalize_doc_codes(text: str) -> str:
    """
    Chuẩn hóa mã văn bản pháp luật VN — tách / và - bằng dấu cách.
      57/2022/QH15   → 57 2022 QH 15
      64/2007/NĐ-CP  → 64 2007 NĐ CP
      36-NQ/TW       → 36 NQ TW
    """

    def _split(m: re.Match) -> str:
        return re.sub(r"[/\-]", " ", m.group()).strip()

    text = re.sub(
        r"\b\d{1,4}(?:/\d{2,4})?/[A-ZĐĂÂ][A-ZĐĂÂa-z0-9]*(?:-[A-ZĐĂÂa-z0-9]+)*\b",
        _split, text,
    )
    text = re.sub(r"\b\d{1,4}-[A-ZĐĂÂ]{2,}/[A-ZĐĂÂ]{2,}\b", _split, text)
    text = re.sub(r"\bQH(\d+)\b", r"QH \1", text)
    return text


def normalize_numbers(text: str, dbg=None) -> str:
    """
    Regex safety net cho số + đơn vị.
    Khi có LLM, chạy TRƯỚC để LLM không thấy raw numbers.
    Khi không có LLM, chạy như bước preprocessing chính.

    Thứ tự quan trọng (đặc thù → tổng quát):
      1. 24/7
      2. Mobile gen (5G, 4G)
      3. Mã văn bản pháp luật
      4. Số La Mã (quý/điều/chương)
      5. ms (trước rule đơn vị storage để tránh nhầm M/S)
      6. Đơn vị storage: TB/GB/MB/KB  → lowercase (TTS đọc letter-by-letter)
      7. Đơn vị tần số: GHz/MHz/kHz   → lowercase
      8. Đơn vị băng thông: Gbps/Mbps → lowercase
      9. fps
     10. Thập phân + %
     11. Nguyên + %
     12. Quý tài chính: Q2/2025
     13. Dấu chấm nghìn: 10.000 → 10000
     14. Thập phân còn sót
    """
    before = text

    # 1. 24/7
    text = re.sub(r"\b24\s*/\s*7\b", "hai mươi tư bảy", text)

    # 2. Mobile generation
    text = re.sub(r"\b([2-5])G\b", lambda m: f"{MOBILE_GEN[m.group(1)]} {EN_LETTER['G']}", text)

    # 3. Mã văn bản
    text = _normalize_doc_codes(text)

    # 4. Số La Mã
    text = ROMAN_PATTERN.sub(
        lambda m: f"{m.group(1)} {ROMAN_VI[m.group(2).upper()]}", text
    )

    # 5. ms — trước storage để tránh "MBps" bị nhầm
    text = re.sub(r"(\d+)\s*ms\b", lambda m: f"{m.group(1)} mi li giây", text, flags=re.IGNORECASE)

    # 6. Storage units — lowercase để TTS đọc letter-by-letter theo EN
    text = re.sub(
        r"(\d+(?:[.,]\d+)?)\s*(TB|GB|MB|KB)\b",
        lambda m: f"{m.group(1)} {m.group(2).lower()}",
        text, flags=re.IGNORECASE,
    )

    # 7. Frequency
    text = re.sub(
        r"(\d+(?:[.,]\d+)?)\s*(GHz|MHz|kHz)\b",
        lambda m: f"{m.group(1)} {m.group(2).lower()}",
        text,
    )

    # 8. Bandwidth
    text = re.sub(
        r"(\d+(?:[.,]\d+)?)\s*(Gbps|Mbps|Kbps)\b",
        lambda m: f"{m.group(1)} {m.group(2).lower()}",
        text, flags=re.IGNORECASE,
    )

    # 9. fps
    text = re.sub(r"(\d+)\s*fps\b", lambda m: f"{m.group(1)} fps", text)

    # 10. Thập phân + %
    text = re.sub(
        r"(\d+)[.,](\d+)\s*%",
        lambda m: f"{m.group(1)} phẩy {m.group(2)} phần trăm",
        text,
    )
    # 11. Nguyên + %
    text = re.sub(r"(\d+)\s*%", lambda m: f"{m.group(1)} phần trăm", text)

    # 12. Quý tài chính
    def _quarter(m: re.Match) -> str:
        q, y = int(m.group(1)), m.group(2)
        return f"quý {QUARTER_VI.get(q, str(q))} năm {y}"

    text = re.sub(r"\bQ([1-4])[/\-](\d{2,4})\b", _quarter, text)

    # 13. Dấu chấm nghìn
    text = re.sub(
        r"(\d{1,3})(\.\d{3})+(?!\d)",
        lambda m: m.group(0).replace(".", ""),
        text,
    )

    # 14. Thập phân còn sót
    text = re.sub(
        r"\b(\d+)[,.](\d+)\b",
        lambda m: f"{m.group(1)} phẩy {m.group(2)}",
        text,
    )

    if dbg and text != before:
        dbg("NUM", f"  IN : {before[:120]}")
        dbg("NUM", f"  OUT: {text[:120]}")
    return text


_vn_normalizer = None


def _get_vn_normalizer():
    global _vn_normalizer
    if _vn_normalizer is None:
        try:
            from vietnormalizer import VietnameseNormalizer
            _vn_normalizer = VietnameseNormalizer()
        except Exception:
            pass
    return _vn_normalizer


def vi_numbers(text: str) -> str:
    """
    Chuyển số nguyên thuần túy → chữ Việt qua vietnormalizer.
      35   → ba mươi lăm
      500  → năm trăm
      15   → mười lăm   (sau normalize_numbers: "15 phẩy 5" → "mười lăm phẩy năm")
      2030 → hai nghìn không trăm ba mươi

    Không động đến số nằm trong token hỗn hợp (F5, H2O, gpt-4o).
    Chạy SAU normalize_numbers() — lúc đó % và ms đã được convert rồi,
    chỉ còn số nguyên bare cần xử lý.
    """
    vn = _get_vn_normalizer()
    if vn is None:
        return text

    def _convert(m: re.Match) -> str:
        try:
            return vn.normalize(m.group()).strip()
        except Exception:
            return m.group()

    return DIGIT_RE.sub(_convert, text)
