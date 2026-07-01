import unittest

from . import _pathfix  # noqa: F401
from text_pipeline.letters import EN_LETTER, LETTER_TABLE, expand_abbrevs_fallback


class TestLetters(unittest.TestCase):
    def test_letter_table_matches_dict(self):
        for k, v in EN_LETTER.items():
            self.assertIn(f"{k}={v}", LETTER_TABLE)

    def test_expand_allcaps(self):
        result = expand_abbrevs_fallback("Hệ thống AI mới")
        self.assertIn("ây ai", result)

    def test_expand_multi_letter(self):
        result = expand_abbrevs_fallback("CPU nhanh hơn")
        self.assertEqual(result, "xi pi iu nhanh hơn")

    def test_does_not_touch_lowercase(self):
        result = expand_abbrevs_fallback("deploy model")
        self.assertEqual(result, "deploy model")

    def test_does_not_touch_single_letter(self):
        # ALLCAPS_RE yêu cầu 2-8 ký tự, 1 ký tự đơn không bị match
        result = expand_abbrevs_fallback("Vitamin C tốt")
        self.assertEqual(result, "Vitamin C tốt")


if __name__ == "__main__":
    unittest.main()
