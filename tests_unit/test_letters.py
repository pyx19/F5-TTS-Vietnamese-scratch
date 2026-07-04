import unittest

from . import _pathfix  # noqa: F401
from text_pipeline.letters import (
    EN_LETTER,
    LETTER_TABLE,
    VN_LETTER,
    expand_abbrevs_fallback,
    expand_known_abbrevs_case_insensitive,
    spell_vn_letters,
)


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

    def test_spell_vn_letters_basic(self):
        # Khác EN_LETTER: N/C/P đọc kiểu Việt (nờ/xê/pê), không phải kiểu Anh (en/xi/pi)
        self.assertEqual(spell_vn_letters("NĐCP"), "nờ đê xê pê")

    def test_spell_vn_letters_case_insensitive(self):
        self.assertEqual(spell_vn_letters("nđcp"), "nờ đê xê pê")

    def test_spell_vn_letters_lowercase_r_in_code(self):
        # "TTr" trong mã văn bản (vd 102/TTr-BTTTT) — chữ "r" viết thường vẫn phải
        # được spell như chữ cái bình thường, không bị bỏ qua.
        self.assertEqual(spell_vn_letters("TTr"), "tê tê rờ")

    def test_vn_letter_differs_from_en_letter(self):
        # Xác nhận 2 bảng KHÁC nhau có chủ đích — VN_LETTER không phải bản sao EN_LETTER
        self.assertNotEqual(VN_LETTER["N"], EN_LETTER["N"])
        self.assertNotEqual(VN_LETTER["C"], EN_LETTER["C"])
        self.assertNotEqual(VN_LETTER["P"], EN_LETTER["P"])

    def test_w_reads_ve_kep(self):
        self.assertEqual(VN_LETTER["W"], "vê kép")

    def test_known_abbrev_lowercased_by_llm_gets_spelled(self):
        # Mô phỏng đúng bug thực tế: LLM hạ thường "ZTA" -> "zta" mà không spell.
        result = expand_known_abbrevs_case_insensitive(
            "theo tiêu chuẩn zero trust architecture, zta.", {"ZTA"}
        )
        self.assertIn("zi ti ây", result)

    def test_known_abbrev_does_not_touch_unrelated_lowercase_words(self):
        # An toàn: không đụng tới các từ tiếng Anh thường LLM cố tình giữ nguyên
        result = expand_known_abbrevs_case_insensitive(
            "deploy model lên server", {"ZTA"}
        )
        self.assertEqual(result, "deploy model lên server")

    def test_known_abbrev_empty_set_is_noop(self):
        result = expand_known_abbrevs_case_insensitive("giữ nguyên hoàn toàn", set())
        self.assertEqual(result, "giữ nguyên hoàn toàn")


if __name__ == "__main__":
    unittest.main()
