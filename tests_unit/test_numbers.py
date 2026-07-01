import unittest

from . import _pathfix  # noqa: F401
from text_pipeline.numbers import normalize_numbers, vi_numbers


class TestNormalizeNumbers(unittest.TestCase):
    def test_ms(self):
        self.assertEqual(normalize_numbers("Độ trễ 200ms"), "Độ trễ 200 mi li giây")

    def test_storage_units_lowercased(self):
        self.assertIn("10 gb", normalize_numbers("Ổ cứng 10GB"))

    def test_frequency_units(self):
        # Số thập phân được tách "phẩy" TRƯỚC nên "3.5" -> "3 phẩy 5"; đơn vị lowercase.
        self.assertIn("3 phẩy 5 ghz", normalize_numbers("Xung nhịp 3.5GHz"))

    def test_bandwidth_units(self):
        self.assertIn("900 mbps", normalize_numbers("Tốc độ 900Mbps"))

    def test_24_7(self):
        self.assertIn("hai mươi tư bảy", normalize_numbers("Vận hành 24/7"))

    def test_mobile_gen(self):
        result = normalize_numbers("Mạng 5G mới")
        self.assertIn("năm gờ", result)

    def test_decimal_percent(self):
        # normalize_numbers() chỉ xử lý dấu thập phân/% -> chữ; số nguyên -> chữ Việt là việc của vi_numbers()
        self.assertEqual(normalize_numbers("CAGR 22.4%"), "CAGR 22 phẩy 4 phần trăm")

    def test_integer_percent(self):
        self.assertEqual(normalize_numbers("giảm 15%"), "giảm 15 phần trăm")

    def test_quarter_finance(self):
        self.assertIn("quý ba năm 2024", normalize_numbers("hoàn thành trước Q3/2024"))

    def test_thousand_dot(self):
        self.assertEqual(normalize_numbers("10.000 tỷ đồng"), "10000 tỷ đồng")

    def test_doc_code_split(self):
        result = normalize_numbers("văn bản 57/2022/QH15")
        self.assertIn("57 2022 QH 15", result)

    def test_doc_code_split_dash(self):
        result = normalize_numbers("Nghị quyết 36-NQ/TW")
        self.assertIn("36 NQ TW", result)

    def test_roman_numeral_quarter(self):
        # IV -> "tư" (theo cách gọi tự nhiên "quý tư"), không phải "bốn"
        result = normalize_numbers("trước quý IV")
        self.assertIn("quý tư", result)

    def test_leftover_decimal(self):
        self.assertEqual(normalize_numbers("15.5 tỷ USD"), "15 phẩy 5 tỷ USD")

    def test_leftover_decimal_then_vi_numbers(self):
        # Kết hợp 2 stage đúng thứ tự pipeline: normalize_numbers() rồi vi_numbers()
        result = vi_numbers(normalize_numbers("15.5 tỷ USD"))
        self.assertIn("mười lăm phẩy năm", result)

    def test_does_not_touch_mixed_token(self):
        # số nằm trong token hỗn hợp như F5, RTX4090 không bị regex số nguyên chạm vào
        result = normalize_numbers("mô hình F5-TTS")
        self.assertIn("F5-TTS", result)


class TestViNumbers(unittest.TestCase):
    def test_skips_when_normalizer_missing(self):
        # Không raise exception ngay cả khi vietnormalizer không cài — trả nguyên input hoặc converted.
        result = vi_numbers("35 con mèo")
        self.assertIsInstance(result, str)

    def test_does_not_touch_mixed_alnum(self):
        result = vi_numbers("F5 model và H2O")
        self.assertIn("F5", result)
        self.assertIn("H2O", result)


if __name__ == "__main__":
    unittest.main()
