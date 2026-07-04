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
        # Không còn xử lý đặc biệt — rơi xuống rule tỷ lệ chung, nhất quán với "39/40"
        self.assertIn("24 trên 7", normalize_numbers("Vận hành 24/7"))

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
        # Số phẩy-ngăn-cách (nghỉ hơi thay vì đọc "gạch chéo"); mã "QH15" SPELL kiểu
        # Việt ("quy hát"), không dịch nghĩa thành "quốc hội" (đây là mã định danh).
        result = normalize_numbers("văn bản 57/2022/QH15")
        self.assertIn("57, 2022, quy hát 15", result)

    def test_doc_code_split_dash(self):
        result = normalize_numbers("Nghị quyết 36-NQ/TW")
        self.assertIn("36, nờ quy tê vê kép", result)

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

    def test_full_date_dmy(self):
        result = normalize_numbers("Sinh ngày 15/03/1990 tại Hà Nội.")
        self.assertIn("ngày 15 tháng 3 năm 1990", result)
        self.assertNotIn("ngày ngày", result)  # không lặp "ngày" khi source đã có sẵn

    def test_full_date_dmy_without_ngay_prefix(self):
        result = normalize_numbers("Có hiệu lực từ 15/03/2024.")
        self.assertIn("ngày 15 tháng 3 năm 2024", result)

    def test_date_dm_with_ngay_no_year(self):
        result = normalize_numbers("Hoàn thành trước ngày 30/6.")
        self.assertIn("ngày 30 tháng 6", result)

    def test_month_year_with_thang_prefix(self):
        result = normalize_numbers("Phiên bản phát hành tháng 07/2024.")
        self.assertIn("tháng 7 năm 2024", result)

    def test_ratio_ratio_fallback(self):
        # "39/40" không khớp ngày (40 không phải tháng hợp lệ) -> tỷ lệ "trên"
        self.assertIn("39 trên 40", normalize_numbers("Chỉ tiêu đạt 39/40 điểm."))

    def test_ratio_ambiguous_two_part_no_keyword(self):
        # Không có "ngày"/"tháng" đứng trước -> mặc định coi là tỷ lệ, không phải ngày
        self.assertIn("3 trên 5", normalize_numbers("Tỷ lệ thắng 3/5 trận đấu."))

    def test_24_7_not_confused_with_date(self):
        # "24/7/2024" là ngày, không phải idiom "24/7" (test_24_7 ở trên đã cover idiom riêng)
        result = normalize_numbers("Trung tâm hoạt động 24/7/2024 không nghỉ.")
        self.assertIn("ngày 24 tháng 7 năm 2024", result)
        self.assertNotIn("hai mươi tư bảy", result)

    def test_doc_code_and_date_combined(self):
        result = normalize_numbers("Nghị định số 64/2007/NĐ-CP ngày 15/03/2024.")
        self.assertIn("64, 2007, nờ đê xê pê", result)
        self.assertIn("ngày 15 tháng 3 năm 2024", result)

    def test_invalid_date_falls_back_unchanged(self):
        # Tháng > 12 -> không phải ngày hợp lệ, giữ nguyên cho rule khác/fallback xử lý
        result = normalize_numbers("Mã số 99/88/2024 không phải ngày.")
        self.assertIn("99/88/2024", result)

    def test_year_range(self):
        result = normalize_numbers("Kế hoạch giai đoạn 2021-2025.")
        self.assertIn("2021 đến 2025", result)

    def test_year_range_does_not_touch_short_hyphenated_numbers(self):
        # Không phải năm 4 chữ số -> không đụng vào (tránh nhầm ID/số điện thoại)
        result = normalize_numbers("Trang 10-15 của tài liệu.")
        self.assertNotIn("đến", result)

    def test_doc_code_mixed_case_and_multiple_letters(self):
        # "TTr-BTTTT" — chữ "r" thường trong "TTr" vẫn phải được spell đầy đủ,
        # không bị rớt/lẫn với "BTTTT" (8 chữ cái tổng cộng: T,T,r,B,T,T,T,T)
        result = normalize_numbers("Tờ trình số 102/TTr-BTTTT.")
        self.assertIn("102, tê tê rờ bê tê tê tê tê", result)

    def test_doc_code_bare_qh_with_digits(self):
        result = normalize_numbers("Quốc hội khóa QH15 đã thông qua.")
        self.assertIn("quy hát 15", result)

    def test_doc_code_leading_zero_preserved(self):
        # "07" là mã định danh, số 0 đầu CÓ ý nghĩa — phải đọc "không bảy",
        # không phải "bảy" (mất số 0 nếu để vi_numbers() xử lý như giá trị số).
        result = normalize_numbers("Mã số văn bản 07/2024/QH15.")
        self.assertIn("không bảy, 2024", result)

    def test_doc_code_no_leading_zero_unaffected(self):
        # Số không có số 0 đầu vẫn giữ nguyên dạng số (để vi_numbers() xử lý sau)
        result = normalize_numbers("Nghị định số 64/2007/NĐ-CP.")
        self.assertIn("64, 2007,", result)

    def test_doc_code_trailing_pause_after_code_ends(self):
        # Nghỉ hơi NGAY SAU khi mã đọc xong, KHÔNG ở giữa mã (giữa NQ và TW).
        result = normalize_numbers("Nghị quyết 36-NQ/TW ngày 01 tháng 7 năm 2014.")
        self.assertIn("vê kép, ngày", result)
        self.assertNotIn("tê, vê kép", result)

    def test_doc_code_no_double_comma_before_existing_punctuation(self):
        # Mã đứng ngay trước dấu câu khác trong text gốc -> không tạo ",," hay ",."
        self.assertNotIn(",,", normalize_numbers("Báo cáo số 45/BC-BTP,"))
        self.assertNotIn(",.", normalize_numbers("Báo cáo số 45/BC-BTP."))


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
