import os
import tempfile
import unittest
from pathlib import Path

from . import _pathfix  # noqa: F401
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
        chunks = preprocess_and_chunk(text, self.vocab_file, max_chars=60)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 60)


if __name__ == "__main__":
    unittest.main()
