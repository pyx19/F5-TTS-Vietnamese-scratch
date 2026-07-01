import unittest

from . import _pathfix  # noqa: F401
from text_pipeline.gate import SkipRatioTracker, needs_llm


class TestGate(unittest.TestCase):
    def test_pure_vietnamese_skipped(self):
        self.assertFalse(needs_llm("Hôm nay trời đẹp quá"))

    def test_allcaps_needs_llm(self):
        self.assertTrue(needs_llm("Hệ thống AI mới"))

    def test_mixed_case_brand_needs_llm(self):
        self.assertTrue(needs_llm("Ứng dụng SaaS phổ biến"))

    def test_mixed_alnum_needs_llm(self):
        self.assertTrue(needs_llm("Card RTX4090 mạnh mẽ"))

    def test_unit_leftover_needs_llm(self):
        # NEEDS_LLM_RE khớp case-sensitive; trong pipeline thật, normalize_numbers() đã
        # lowercase đơn vị TRƯỚC khi gate chạy nên nhánh này chỉ khớp khi test gate.py
        # độc lập với chữ hoa gốc (trước khi qua normalize_numbers).
        self.assertTrue(needs_llm("băng thông 900 Mbps"))


class TestSkipRatioTracker(unittest.TestCase):
    def test_ratio(self):
        tracker = SkipRatioTracker()
        tracker.record(True)
        tracker.record(True)
        tracker.record(False)
        self.assertAlmostEqual(tracker.skip_ratio, 2 / 3)
        self.assertIn("2/3", tracker.summary())


if __name__ == "__main__":
    unittest.main()
