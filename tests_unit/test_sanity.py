import unittest

from . import _pathfix  # noqa: F401
from text_pipeline.sanity import contains_cjk, sanity_ok


class TestSanity(unittest.TestCase):
    def test_empty_output_fails(self):
        self.assertFalse(sanity_ok("câu gốc", ""))

    def test_normal_output_passes(self):
        self.assertTrue(sanity_ok(
            "Generative AI và Machine Learning",
            "Generative ây ai và Machine Learning",
        ))

    def test_inflated_output_fails(self):
        original = "AI phát triển"
        bloated = original * 6
        self.assertFalse(sanity_ok(original, bloated))

    def test_lost_vietnamese_words_fails(self):
        original = "hệ thống công nghệ thông tin quốc gia đang phát triển mạnh mẽ"
        result = "hệ thống ai ti"
        self.assertFalse(sanity_ok(original, result))

    def test_cjk_detection(self):
        self.assertTrue(contains_cjk("这是中文"))
        self.assertFalse(contains_cjk("đây là tiếng việt"))


if __name__ == "__main__":
    unittest.main()
