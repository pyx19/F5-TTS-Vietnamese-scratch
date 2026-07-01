"""
Bước 5 — _needs_llm() gate.

Skip Qwen với câu thuần Việt → giảm latency, giảm risk hallucinate.
Câu cần LLM khi có: ALL-CAPS abbrev, mixed-case brand, mixed alnum, hoặc
số + đơn vị kỹ thuật còn sót (storage/frequency/bandwidth/fps — normalize_numbers
đã lowercase các đơn vị này chứ không xoá, nên vẫn cần LLM quyết định ngữ cảnh
đọc tự nhiên xung quanh).
"""

import re

NEEDS_LLM_RE = re.compile(
    r"\b[A-ZĐĂÂÊÔƯƠ]{2,8}\b"  # ALL-CAPS abbrev: AI, CPU, UBND
    r"|[A-Z][a-z]+[A-Z][A-Za-z]*"  # mixed-case brand: SaaS, IoT, ChatGPT
    r"|\b[A-Z]{1,4}\d+[A-Za-z0-9\-]*\b"  # mixed alnum: RTX4090, GPT-4o, F5-TTS
    r"|\d+\s*(?:TB|GB|MB|KB|GHz|MHz|kHz|Gbps|Mbps|Kbps|fps)\b"  # số+đơn vị
)


def needs_llm(sentence: str) -> bool:
    """Gate: câu có token nào cần LLM xử lý không."""
    return bool(NEEDS_LLM_RE.search(sentence))


class SkipRatioTracker:
    """Đếm số câu skip/total để log tỷ lệ skip (kỳ vọng ~40-60% với văn bản hành chính)."""

    def __init__(self) -> None:
        self.total = 0
        self.skipped = 0

    def record(self, skipped: bool) -> None:
        self.total += 1
        if skipped:
            self.skipped += 1

    @property
    def skip_ratio(self) -> float:
        return self.skipped / self.total if self.total else 0.0

    def summary(self) -> str:
        return f"{self.skipped}/{self.total} câu skip LLM ({self.skip_ratio:.0%})"
