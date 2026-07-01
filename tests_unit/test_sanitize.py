import os
import tempfile
import unittest
from pathlib import Path

from . import _pathfix  # noqa: F401
from text_pipeline import sanitize as sanitize_mod
from text_pipeline.sanitize import sanitize_for_vivoice


class TestSanitize(unittest.TestCase):
    def setUp(self):
        # Reset module-level cache giữa các test vì _get_allowed_chars() cache theo lần gọi đầu tiên.
        sanitize_mod._allowed_chars = None
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)  # tránh PermissionError khi unlink trên Windows (fd còn mở)
        Path(path).write_text("\n".join(list("abcdefiêuơ .,!?")), encoding="utf-8")
        self.vocab_file = Path(path)

    def tearDown(self):
        self.vocab_file.unlink(missing_ok=True)
        sanitize_mod._allowed_chars = None

    def test_strips_disallowed_chars(self):
        result = sanitize_for_vivoice("abc#$%def", self.vocab_file)
        self.assertNotIn("#", result)
        self.assertNotIn("$", result)

    def test_keeps_allowed_chars(self):
        result = sanitize_for_vivoice("cabê deiơu.", self.vocab_file)
        self.assertEqual(result, "cabê deiơu.")

    def test_collapses_whitespace(self):
        result = sanitize_for_vivoice("abc @@@ def", self.vocab_file)
        self.assertEqual(result, "abc def")


if __name__ == "__main__":
    unittest.main()
