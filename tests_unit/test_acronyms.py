import unittest

from . import _pathfix  # noqa: F401
from text_pipeline.acronyms import expand_acronyms_fallback


class TestAcronyms(unittest.TestCase):
    def test_ubnd_expanded(self):
        self.assertIn("ủy ban nhân dân", expand_acronyms_fallback("UBND tỉnh ban hành"))

    def test_hdnd_expanded(self):
        self.assertIn("hội đồng nhân dân", expand_acronyms_fallback("HĐND đã họp"))

    def test_ttg_expanded_before_tt(self):
        # TTg phải khớp trọn "thủ tướng chính phủ", không bị TT ("thông tư") nuốt mất chữ g
        result = expand_acronyms_fallback("Quyết định của TTg")
        self.assertIn("thủ tướng chính phủ", result)
        self.assertNotIn("thông tưg", result)

    def test_tt_alone_expanded(self):
        self.assertIn("thông tư", expand_acronyms_fallback("Ban hành TT hướng dẫn"))

    def test_does_not_touch_unrelated_word(self):
        result = expand_acronyms_fallback("con cá tra")
        self.assertEqual(result, "con cá tra")

    def test_case_insensitive_catches_llm_lowercased_output(self):
        # Quan sát thực tế: Qwen2.5:3b đôi khi bỏ sót việc expand và trả về viết tắt
        # dưới dạng chữ thường (vd "nq tw" thay vì "nghị quyết trung ương") — fallback
        # phải bắt được cả trường hợp này, không chỉ chữ hoa nguyên bản.
        result = expand_acronyms_fallback("nghị quyết số ba mươi sáu nq tw ngày một")
        self.assertIn("nghị quyết trung ương", result)

    def test_new_entries_from_reference_list(self):
        # Bổ sung từ 20260701_Danh_sach_tu_viet_tat.xlsx
        self.assertIn("phó tổng giám đốc", expand_acronyms_fallback("PTGĐ đã ký duyệt"))
        self.assertIn("tổng công ty", expand_acronyms_fallback("TCT báo cáo kết quả"))
        self.assertIn("công nghệ thông tin", expand_acronyms_fallback("phòng CNTT"))
        self.assertIn("về việc", expand_acronyms_fallback("V/v triển khai kế hoạch"))

    def test_multiword_key_with_space(self):
        self.assertIn("quân chủng phòng không không quân", expand_acronyms_fallback("thuộc QC PKKQ"))


if __name__ == "__main__":
    unittest.main()
