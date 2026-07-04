import json
import unittest
from unittest.mock import MagicMock, patch

from . import _pathfix  # noqa: F401
from text_pipeline import cache
from text_pipeline.chunk_llm import llm_split_points


def _fake_response(content: str):
    body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


class TestLlmSplitPoints(unittest.TestCase):
    def setUp(self):
        # Tránh cache hit giữa các test (cache.py dùng chung 1 instance module-level)
        cache._disabled = True

    def test_valid_split_points_accepted(self):
        text = "căn cứ nghị quyết số năm mươi bảy của quốc hội về dự án trọng điểm quốc gia"
        llm_reply = "căn cứ nghị quyết số năm mươi bảy ||| của quốc hội về dự án trọng điểm quốc gia"
        with patch("urllib.request.urlopen", return_value=_fake_response(llm_reply)):
            result = llm_split_points(text, max_chars=50, model="qwen2.5:3b", url="http://x")
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        # Không được tách rời cụm "trọng điểm quốc gia"
        self.assertTrue(any("trọng điểm quốc gia" in part for part in result))

    def test_content_altered_rejected(self):
        # LLM tự thêm/bớt từ -> validate phải fail, trả None để chunking.py fallback
        text = "căn cứ nghị quyết số năm mươi bảy của quốc hội"
        llm_reply = "căn cứ nghị quyết số ||| năm mươi tám của quốc hội"  # đổi "bảy" thành "tám"
        with patch("urllib.request.urlopen", return_value=_fake_response(llm_reply)):
            result = llm_split_points(text, max_chars=50, model="qwen2.5:3b", url="http://x")
        self.assertIsNone(result)

    def test_no_delimiter_inserted_rejected(self):
        text = "căn cứ nghị quyết số năm mươi bảy của quốc hội"
        with patch("urllib.request.urlopen", return_value=_fake_response(text)):
            result = llm_split_points(text, max_chars=50, model="qwen2.5:3b", url="http://x")
        self.assertIsNone(result)

    def test_connection_error_returns_none(self):
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            result = llm_split_points("câu bất kỳ", max_chars=50, model="qwen2.5:3b", url="http://x")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
