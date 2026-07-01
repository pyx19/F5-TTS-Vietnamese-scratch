import unittest

from . import _pathfix  # noqa: F401
from text_pipeline import cache


class TestCache(unittest.TestCase):
    def test_get_set_roundtrip_or_graceful_degrade(self):
        """
        Nếu diskcache có cài: set() rồi get() phải trả đúng giá trị.
        Nếu diskcache KHÔNG cài (vd offline dev machine): get()/set() không được raise,
        chỉ âm thầm no-op — pipeline vẫn chạy được, chỉ mất lợi ích cache.
        """
        try:
            cache.set("test-model", "câu test", "câu đã normalize")
            result = cache.get("test-model", "câu test")
        except Exception as e:  # pragma: no cover
            self.fail(f"cache.get/set không được raise exception: {e}")

        if cache._disabled:
            self.assertIsNone(result)
        else:
            self.assertEqual(result, "câu đã normalize")

    def test_miss_returns_none(self):
        result = cache.get("test-model", "câu chưa từng cache trước đó xyz123")
        self.assertIsNone(result)

    def test_stats_tracker(self):
        stats = cache.CacheStats()
        stats.record(hit=True)
        stats.record(hit=False)
        stats.record(hit=True)
        self.assertEqual(stats.hits, 2)
        self.assertEqual(stats.misses, 1)
        self.assertAlmostEqual(stats.hit_ratio, 2 / 3)


if __name__ == "__main__":
    unittest.main()
