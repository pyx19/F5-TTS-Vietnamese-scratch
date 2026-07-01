"""
Bước 8 — diskcache cho kết quả LLM normalize.

Văn bản hành chính lặp lại nhiều pattern (cùng viết tắt, cùng cụm từ) → cache
hit rate cao sau warm-up. Cache key = md5(sentence đưa vào LLM), value = câu
đã normalize. Expire sau 7 ngày để tự động dọn khi prompt/model thay đổi.
"""

import hashlib
from pathlib import Path
from typing import Optional

CACHE_DIR = Path(__file__).parent.parent / ".cache" / "llm_normalize"
EXPIRE_SECONDS = 7 * 24 * 3600

_cache = None
_disabled = False


def _get_cache():
    global _cache, _disabled
    if _disabled:
        return None
    if _cache is None:
        try:
            import diskcache
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            _cache = diskcache.Cache(str(CACHE_DIR))
        except Exception:
            _disabled = True
            return None
    return _cache


def _key(model: str, sentence: str) -> str:
    return hashlib.md5(f"{model}::{sentence}".encode("utf-8")).hexdigest()


class CacheStats:
    def __init__(self) -> None:
        self.hits = 0
        self.misses = 0

    def record(self, hit: bool) -> None:
        if hit:
            self.hits += 1
        else:
            self.misses += 1

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_ratio(self) -> float:
        return self.hits / self.total if self.total else 0.0

    def summary(self) -> str:
        return f"cache {self.hits}/{self.total} hit ({self.hit_ratio:.0%})"


STATS = CacheStats()


def get(model: str, sentence: str) -> Optional[str]:
    cache = _get_cache()
    if cache is None:
        return None
    value = cache.get(_key(model, sentence))
    STATS.record(hit=value is not None)
    return value


def set(model: str, sentence: str, normalized: str) -> None:
    cache = _get_cache()
    if cache is None:
        return
    cache.set(_key(model, sentence), normalized, expire=EXPIRE_SECONDS)
