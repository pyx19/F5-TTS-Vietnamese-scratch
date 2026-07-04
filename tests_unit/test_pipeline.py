import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import _pathfix  # noqa: F401
from text_pipeline.chunking import PARAGRAPH_BREAK_MARKER
from text_pipeline.pipeline import preprocess_and_chunk, preprocess_text


class TestPipelineRuleBasedOnly(unittest.TestCase):
    """Không dùng LLM (llm_model=None) — chỉ test rule-based stages (1, 4, 6, cleanup)."""

    def test_numbers_expanded(self):
        result = preprocess_text("Độ trễ giảm còn 200ms.")
        self.assertIn("mi li giây", result)

    def test_admin_acronym_expanded_by_fallback(self):
        result = preprocess_text("UBND tỉnh đã phê duyệt.")
        self.assertIn("ủy ban nhân dân", result)

    def test_tech_allcaps_spelled_by_fallback(self):
        result = preprocess_text("Hệ thống AI mới.")
        self.assertIn("ây ai", result)

    def test_parens_removed(self):
        result = preprocess_text("Điện toán đám mây (Cloud Computing) đang phổ biến.")
        self.assertNotIn("(", result)
        self.assertNotIn(")", result)

    def test_parens_insert_pause_comma_before_semicolon_after(self):
        # Mở ngoặc -> phẩy (nghỉ nhẹ); đóng ngoặc -> CHẤM PHẨY (mạnh hơn, tạo ranh
        # giới chunk thật — trước đây chỉ dùng phẩy cả 2 đầu, đọc "hơi nhanh" ngay
        # sau khi đóng ngoặc vì vẫn nằm trong 1 chunk/synthesis call liên tục).
        result = preprocess_text("Điện toán đám mây (Cloud Computing) đang phổ biến.")
        self.assertIn("mây, cloud computing; đang", result)

    def test_parens_at_end_of_sentence_no_double_punctuation(self):
        # "(ZTA)." không được để lại ";." thừa trước dấu chấm cuối câu.
        result = preprocess_text("Theo tiêu chuẩn Zero Trust Architecture (ZTA).")
        self.assertNotIn(";.", result)
        self.assertNotIn(",.", result)
        self.assertTrue(result.rstrip().endswith("."))

    def test_curly_braces_insert_pause_comma_before_semicolon_after(self):
        result = preprocess_text("Thư mục dự án {Data Silos} cần dọn dẹp.")
        self.assertNotIn("{", result)
        self.assertNotIn("}", result)
        self.assertIn("dự án, data silos; cần", result)

    def test_ampersand_between_words_becomes_va(self):
        result = preprocess_text("Bộ Nông nghiệp & Phát triển nông thôn.")
        self.assertIn("nông nghiệp và phát triển", result)
        self.assertNotIn("&", result)

    def test_ampersand_initialism_spelled_as_one_token(self):
        # "P&G" nối liền thành "PG" trước khi bị đọc là 2 chữ đơn lẻ vô nghĩa
        result = preprocess_text("Tập đoàn P&G vừa công bố báo cáo.")
        self.assertIn("pi gờ", result)
        self.assertNotIn("&", result)

    def test_quotes_stripped(self):
        result = preprocess_text('Các "ốc đảo dữ liệu" đang gây nghẽn mạch.')
        self.assertNotIn('"', result)
        self.assertIn("ốc đảo dữ liệu", result)

    def test_ellipsis_mid_sentence_becomes_comma(self):
        result = preprocess_text("Các đơn vị liên quan... sẽ phối hợp triển khai.")
        self.assertIn("liên quan, sẽ phối hợp", result)

    def test_ellipsis_end_of_sentence_becomes_period(self):
        result = preprocess_text("Dữ liệu đã được cập nhật...")
        self.assertTrue(result.rstrip().endswith("."))
        self.assertNotIn(",", result)

    def test_leading_bullet_dash_stripped(self):
        result = preprocess_text("Danh sách:\n- Việc một\n- Việc hai")
        self.assertNotIn("- việc", result)
        self.assertIn("việc một", result)

    def test_standalone_dash_becomes_pause_not_bare_space(self):
        # " - " (khoảng trắng bao quanh) đóng vai trò ngoặc đơn/chú thích — trước
        # đây thu về 1 khoảng trắng trơn khiến TTS đọc díu "user experience iu ích"
        # không phân biệt được ranh giới cụm từ gốc/viết tắt theo sau.
        result = preprocess_text("Trải nghiệm người dùng (User Experience - UX) là ưu tiên.")
        self.assertIn("user experience, iu ích", result)

    def test_compound_word_dash_still_becomes_space_not_comma(self):
        # "on-premise" (không có khoảng trắng quanh gạch nối) là compound word,
        # KHÔNG phải chú thích — vẫn phải là khoảng trắng, không chèn phẩy thừa.
        result = preprocess_text("Triển khai theo mô hình On-premise.")
        self.assertIn("on premise", result)
        self.assertNotIn("on, premise", result)

    def test_doc_code_pause_comma_not_slash(self):
        # Mã văn bản: nghỉ hơi (phẩy) tại chỗ gạch chéo, không đọc "gạch chéo".
        result = preprocess_text("Nghị định số 64/2007/NĐ-CP quy định chi tiết.")
        self.assertIn("nờ đê xê pê", result)
        self.assertNotIn("/", result)

    def test_doc_code_leading_zero_preserved_end_to_end(self):
        result = preprocess_text("Mã số văn bản 07/2024/QH15.")
        self.assertIn("không bảy", result)

    def test_doc_code_pause_after_code_not_inside(self):
        # Nghỉ NGAY SAU khi mã đọc xong (trước "ngày..."), KHÔNG nghỉ ở giữa mã
        # (giữa "NQ" và "TW") — mã phải đọc liền một mạch như 1 ký hiệu.
        result = preprocess_text("Căn cứ Nghị quyết số 36-NQ/TW ngày 01 tháng 7 năm 2014.")
        self.assertIn("vê kép, ngày", result)
        self.assertNotIn("tê, vê kép", result)

    def test_blank_line_paragraph_break_forces_period(self):
        # Dòng trống = ranh giới đoạn văn mạnh — giờ được đánh dấu bằng
        # PARAGRAPH_BREAK_MARKER (thay vì chỉ nâng dấu câu), và đoạn TRƯỚC marker
        # vẫn phải kết thúc bằng dấu CHẤM bất kể dòng gốc vốn kết bằng dấu gì
        # (quan sát thực tế: "...Báo cáo số 45/BC-BTP,\n\nQUYẾT NGHỊ:" đọc dồn/mất
        # ngắt nghỉ vì dấu phẩy quá yếu cho một ranh giới đoạn/section lớn).
        text = "Báo cáo số 45,\n\nQUYẾT NGHỊ:\nĐiều 1. Nội dung."
        result = preprocess_text(text)
        self.assertIn(PARAGRAPH_BREAK_MARKER, result)
        before_marker = result.split(PARAGRAPH_BREAK_MARKER)[0]
        self.assertTrue(before_marker.rstrip().endswith("."))

    def test_no_blank_line_keeps_original_punctuation(self):
        # Không có dòng trống -> giữ nguyên hành vi cũ (dấu phẩy/hai chấm như bình
        # thường), không bị ép thành chấm một cách không cần thiết, và KHÔNG chèn
        # marker ranh giới đoạn văn (chỉ 1 đoạn duy nhất).
        text = "Câu một,\ncâu hai tiếp tục."
        result = preprocess_text(text)
        self.assertIn("câu một, câu hai", result)
        self.assertNotIn(PARAGRAPH_BREAK_MARKER, result)

    def test_llm_lowercased_allcaps_gets_spelled_not_left_verbatim(self):
        # Mô phỏng đúng bug thực tế: Qwen hạ thường "ZTA" -> "zta" mà không spell.
        # expand_known_abbrevs_case_insensitive() phải bắt lại được sau khi LLM trả về.
        with patch(
            "text_pipeline.pipeline.llm_normalize",
            return_value="theo tiêu chuẩn zero trust architecture, zta.",
        ):
            result = preprocess_text(
                "Theo tiêu chuẩn Zero Trust Architecture (ZTA).", llm_model="fake-model",
            )
        self.assertIn("zi ti ây", result)
        self.assertNotIn("zta", result)

    def test_multiple_blank_line_paragraphs_each_get_marker(self):
        # 3 đoạn (2 dòng trống ngăn cách) -> đúng 2 marker ranh giới đoạn văn.
        text = "Đoạn một nội dung.\n\nĐoạn hai nội dung.\n\nĐoạn ba nội dung."
        result = preprocess_text(text)
        self.assertEqual(result.count(PARAGRAPH_BREAK_MARKER), 2)

    def test_output_is_lowercase(self):
        result = preprocess_text("Việt Nam đang phát triển.")
        self.assertEqual(result, result.lower())

    def test_linebreaks_become_sentence_boundaries(self):
        result = preprocess_text("Dòng một\nDòng hai")
        self.assertIn(".", result)


class TestPreprocessAndChunk(unittest.TestCase):
    def setUp(self):
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)  # tránh PermissionError khi unlink trên Windows (fd còn mở)
        # vocab tối thiểu: chữ thường có dấu cơ bản + số + dấu câu + khoảng trắng
        chars = list("abcdđeghiklmnoprstuvyâêôơưáàảãạ .,!?0123456789")
        Path(path).write_text("\n".join(chars), encoding="utf-8")
        self.vocab_file = Path(path)

    def tearDown(self):
        self.vocab_file.unlink(missing_ok=True)

    def test_returns_chunk_list_within_limit(self):
        text = "Đây là một câu ví dụ dài để kiểm tra việc chia đoạn cho mô hình tổng hợp giọng nói. " * 3
        chunks, _ = preprocess_and_chunk(text, self.vocab_file, max_chars=60)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 60)


if __name__ == "__main__":
    unittest.main()
