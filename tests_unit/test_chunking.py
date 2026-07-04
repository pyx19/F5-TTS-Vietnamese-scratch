import unittest
from unittest.mock import patch

from . import _pathfix  # noqa: F401
from text_pipeline.chunking import PARAGRAPH_BREAK_MARKER, normalize_for_f5


class TestChunking(unittest.TestCase):
    def test_short_text_single_chunk(self):
        chunks, _ = normalize_for_f5("Câu ngắn gọn.", max_chars=200)
        self.assertEqual(len(chunks), 1)

    def test_respects_max_chars(self):
        text = ("Đây là một câu dài. " * 20).strip()
        chunks, _ = normalize_for_f5(text, max_chars=50)
        for c in chunks:
            self.assertLessEqual(len(c), 50)

    def test_splits_on_sentence_boundary(self):
        text = "Câu một kết thúc. Câu hai kết thúc! Câu ba kết thúc?"
        chunks, _ = normalize_for_f5(text, max_chars=20)
        self.assertGreater(len(chunks), 1)
        joined = " ".join(chunks)
        self.assertIn("Câu một", joined)
        self.assertIn("Câu ba", joined)

    def test_no_word_is_split(self):
        text = "mười lăm phẩy năm tỷ đô la mỹ đầu tư vào nghiên cứu phát triển công nghệ mới"
        chunks, _ = normalize_for_f5(text, max_chars=30)
        original_words = set(text.split())
        chunk_words = set(" ".join(chunks).split())
        self.assertEqual(original_words, chunk_words)

    def test_empty_text(self):
        self.assertEqual(normalize_for_f5(""), ([], set()))

    def test_long_word_run_without_punctuation_falls_back_to_whitespace(self):
        text = "một hai ba bốn năm sáu bảy tám chín mười mười một mười hai mười ba mười bốn"
        chunks, _ = normalize_for_f5(text, max_chars=15)
        for c in chunks:
            self.assertLessEqual(len(c), 15)

    def test_no_llm_model_never_calls_llm(self):
        # llm_model=None (mặc định) -> không được gọi chunk_llm dù đoạn dài không dấu câu
        text = "một hai ba bốn năm sáu bảy tám chín mười mười một mười hai mười ba mười bốn"
        with patch("text_pipeline.chunk_llm.llm_split_points") as mocked:
            normalize_for_f5(text, max_chars=15)
            mocked.assert_not_called()

    def test_llm_split_points_used_when_valid(self):
        text = "một hai ba bốn năm sáu bảy tám chín mười mười một mười hai mười ba mười bốn"
        with patch(
            "text_pipeline.chunk_llm.llm_split_points",
            return_value=["một hai ba bốn năm", "sáu bảy tám chín mười", "mười một mười hai mười ba mười bốn"],
        ):
            chunks, _ = normalize_for_f5(text, max_chars=200, llm_model="qwen2.5:3b", ollama_url="http://x")
        self.assertEqual(" ".join(chunks).split(), text.split())

    def test_llm_split_points_none_falls_back_to_whitespace(self):
        # LLM trả None (lỗi/không hợp lệ) -> vẫn phải cắt được, không crash
        text = "một hai ba bốn năm sáu bảy tám chín mười mười một mười hai mười ba mười bốn"
        with patch("text_pipeline.chunk_llm.llm_split_points", return_value=None):
            chunks, _ = normalize_for_f5(text, max_chars=15, llm_model="qwen2.5:3b", ollama_url="http://x")
        for c in chunks:
            self.assertLessEqual(len(c), 15)
        self.assertEqual(" ".join(chunks).split(), text.split())

    def test_pack_never_merges_across_sentence_boundary(self):
        # 2 câu ĐỦ DÀI (trên ngưỡng _MIN_CHUNK_CHARS, không phải fragment tiêu đề),
        # gộp chung vẫn < max_chars — nhưng KHÔNG được gộp thành 1 chunk (đã quan
        # sát thực tế: gộp xuyên câu khiến F5-TTS đọc dồn/nhanh vì duration dự đoán
        # tuyến tính theo ký tự, không biết đây là 2 ý khác nhau).
        text = "Đây là câu thứ nhất, đủ dài để không bị coi là tiêu đề ngắn. Đây là câu thứ hai, cũng đủ dài tương tự."
        chunks, _ = normalize_for_f5(text, max_chars=200)
        self.assertEqual(len(chunks), 2)
        self.assertIn("câu thứ nhất", chunks[0].lower())
        self.assertIn("câu thứ hai", chunks[1].lower())

    def test_tiny_chunk_merges_with_next(self):
        # Chunk quá ngắn (< _MIN_CHUNK_CHARS, vd tiêu đề "Điều 2.") PHẢI gộp với
        # chunk kế tiếp — khác với câu đủ dài (test ở trên) không được gộp.
        text = "Điều 2. Các Nhiệm vụ và Giải pháp Trọng tâm."
        chunks, _ = normalize_for_f5(text, max_chars=200)
        self.assertEqual(len(chunks), 1)

    def test_multiple_tiny_chunks_chain_merge(self):
        # 2 tiêu đề ngắn liên tiếp ("Điều 3." + "Tổ chức Thực hiện.") đều dưới
        # ngưỡng -> gộp thành 1, nhưng KHÔNG gộp tiếp câu dài phía sau.
        text = "Điều 3. Tổ chức Thực hiện. Đây là nội dung chi tiết đủ dài của điều khoản này, không nên bị gộp thêm nữa."
        chunks, _ = normalize_for_f5(text, max_chars=200)
        self.assertEqual(len(chunks), 2)
        self.assertIn("điều 3", chunks[0].lower())
        self.assertIn("tổ chức thực hiện", chunks[0].lower())
        self.assertIn("nội dung chi tiết", chunks[1].lower())

    def test_tiny_chunk_not_merged_if_would_exceed_max_chars(self):
        # Nếu gộp làm vượt max_chars thì KHÔNG gộp, chấp nhận chunk ngắn đứng riêng
        # (đây là ưu tiên accuracy hơn latency — không cắt xén nội dung để gộp bằng mọi giá).
        text = "Điều 4. " + ("Một câu rất dài lặp lại nhiều lần để vượt quá giới hạn ký tự cho phép. " * 4)
        chunks, _ = normalize_for_f5(text.strip(), max_chars=100)
        for c in chunks:
            self.assertLessEqual(len(c), 100)

    def test_pack_still_merges_pieces_within_same_sentence(self):
        # Pack vẫn phải gộp các mảnh NHỎ của CÙNG 1 câu (không tách vụn không cần thiết)
        text = "Một câu, có, nhiều, mệnh đề, ngắn, nhưng, vẫn, là, một, câu, duy nhất."
        chunks, _ = normalize_for_f5(text, max_chars=200)
        self.assertEqual(len(chunks), 1)

    def test_paragraph_marker_splits_into_groups_with_break_after_index(self):
        # PARAGRAPH_BREAK_MARKER phải tách thành 2 nhóm chunk riêng, với đúng 1
        # ranh giới đoạn văn nằm SAU chunk cuối của nhóm đầu (index 0).
        text = f"Điều một, đủ dài để không bị coi là fragment. {PARAGRAPH_BREAK_MARKER} Điều hai, cũng đủ dài tương tự."
        chunks, paragraph_break_after = normalize_for_f5(text, max_chars=200)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(paragraph_break_after, {0})

    def test_no_paragraph_marker_means_no_breaks(self):
        text = "Câu một kết thúc. Câu hai kết thúc."
        _, paragraph_break_after = normalize_for_f5(text, max_chars=200)
        self.assertEqual(paragraph_break_after, set())

    def test_paragraph_break_never_after_last_chunk(self):
        # Marker ở NGAY CUỐI text (không có nội dung theo sau) không được tạo ra
        # 1 "ranh giới" vô nghĩa sau chunk cuối cùng.
        text = f"Nội dung duy nhất, đủ dài để không bị coi là fragment. {PARAGRAPH_BREAK_MARKER}"
        chunks, paragraph_break_after = normalize_for_f5(text, max_chars=200)
        self.assertNotIn(len(chunks) - 1, paragraph_break_after)


if __name__ == "__main__":
    unittest.main()
