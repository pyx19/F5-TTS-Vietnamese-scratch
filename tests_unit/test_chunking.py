import unittest

from . import _pathfix  # noqa: F401
from text_pipeline.chunking import normalize_for_f5


class TestChunking(unittest.TestCase):
    def test_short_text_single_chunk(self):
        chunks = normalize_for_f5("Câu ngắn gọn.", max_chars=200)
        self.assertEqual(len(chunks), 1)

    def test_respects_max_chars(self):
        text = ("Đây là một câu dài. " * 20).strip()
        chunks = normalize_for_f5(text, max_chars=50)
        for c in chunks:
            self.assertLessEqual(len(c), 50)

    def test_splits_on_sentence_boundary(self):
        text = "Câu một kết thúc. Câu hai kết thúc! Câu ba kết thúc?"
        chunks = normalize_for_f5(text, max_chars=20)
        self.assertGreater(len(chunks), 1)
        joined = " ".join(chunks)
        self.assertIn("Câu một", joined)
        self.assertIn("Câu ba", joined)

    def test_no_word_is_split(self):
        text = "mười lăm phẩy năm tỷ đô la mỹ đầu tư vào nghiên cứu phát triển công nghệ mới"
        chunks = normalize_for_f5(text, max_chars=30)
        original_words = set(text.split())
        chunk_words = set(" ".join(chunks).split())
        self.assertEqual(original_words, chunk_words)

    def test_empty_text(self):
        self.assertEqual(normalize_for_f5(""), [])

    def test_long_word_run_without_punctuation_falls_back_to_whitespace(self):
        text = "một hai ba bốn năm sáu bảy tám chín mười mười một mười hai mười ba mười bốn"
        chunks = normalize_for_f5(text, max_chars=15)
        for c in chunks:
            self.assertLessEqual(len(c), 15)


if __name__ == "__main__":
    unittest.main()
