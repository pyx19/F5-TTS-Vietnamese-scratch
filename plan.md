# Vietnamese TTS Normalization Pipeline — Implementation Plan
**Stack:** F5-TTS + ViVoice (`hynt/F5-TTS-Vietnamese-ViVoice`) + Qwen2.5-3B (Ollama local)  
**Priority:** Accuracy > Speed | **Domain:** Văn bản hành chính + Kỹ thuật/Công nghệ

---

## Tổng quan kiến trúc

```
Raw Text
   │
   ▼
[Stage 1] normalize_numbers()          ← Rule-based, TRƯỚC Qwen (fix hallucination)
   │
   ▼
[Stage 2] _needs_llm() gate            ← Skip Qwen nếu câu thuần Việt
   │         │
   │    FALSE (thuần Việt)
   │         └──────────────────────┐
   │ TRUE                           │
   ▼                                │
[Stage 3] _llm_normalize() / Qwen   │  ← Batch per-sentence, temperature=0
   │                                │
   ▼                                │
[Stage 4] _expand_abbrevs_fallback()◄┘  ← Safety net sau Qwen
   │
   ▼
[Stage 5] _sanity_ok() validation      ← Reject output bất thường
   │
   ▼
[Stage 6] sanitize_for_vivoice()       ← Strip ký tự ngoài vocab
   │
   ▼
normalize_for_f5() → chunk ≤ 200 chars
   │
   ▼
F5-TTS / ViVoice inference
```

---

## Các bước thực hiện

### Bước 1 — Audit pipeline hiện tại
**Mục tiêu:** Xác định chính xác những gì đang bị đọc sai trước khi sửa.

- [ ] Chạy test set ~50 câu đại diện (hành chính + kỹ thuật)
- [ ] Log output text sau normalize, so sánh với audio thực tế
- [ ] Phân loại lỗi theo nhóm:
  - Số + đơn vị (`15.5 tỷ USD`, `200ms`, `10GB`)
  - Viết tắt hành chính (`NQ/TW`, `123/QĐ-TTg`, `UBND`)
  - Viết tắt kỹ thuật (`API`, `LLM`, `GPU`, `CAGR`)
  - Code-switching (`On-premise`, `Cloud API`, `review`)
  - Tên riêng (`ViVoice`, `ChatGPT`, `RTX 5060`)
- [ ] Ghi nhận các hallucination pattern của Qwen hiện tại

**Expected output:** Danh sách lỗi có độ ưu tiên, làm baseline để đo cải thiện.

---

### Bước 2 — Tách `normalize_numbers()` ra khỏi LLM flow
**Mục tiêu:** Fix root cause hallucination — Qwen3B không reliable với số phức tạp.

- [ ] Implement `normalize_numbers()` chạy **trước** `_llm_normalize()`
- [ ] Implement `_vi_numbers()` — số tiếng Việt đầy đủ
- [ ] Handle các pattern cụ thể:
  ```
  15.5 tỷ USD    → "mười lăm phẩy năm tỷ đô la mỹ"
  200ms          → "hai trăm mi li giây"
  10GB           → "mười gi ga bai"
  Q3/2024        → "quý ba năm hai không hai tư"
  123/QĐ-TTg     → "một hai ba quyết định thủ tướng chính phủ"
  ```
- [ ] Unit test riêng cho từng pattern, không phụ thuộc Qwen

**Expected output:** 0% hallucination trên số + đơn vị; Qwen không bao giờ thấy raw numbers.

---

### Bước 3 — Xây dựng `_EN_LETTER` dict + inject vào Qwen prompt
**Mục tiêu:** Qwen dùng đúng 1 bảng phiên âm, không tự sáng tạo.

- [ ] Định nghĩa `_EN_LETTER` là **single source of truth**:
  ```python
  _EN_LETTER = {
      'A':'ây', 'B':'bi', 'C':'si', 'D':'đi', 'E':'i',
      'F':'ép', 'G':'ji', 'H':'ét', 'I':'ai', 'J':'jay',
      'K':'cây', 'L':'eo', 'M':'em', 'N':'en', 'O':'ô',
      'P':'pi', 'Q':'kiu', 'R':'a', 'S':'ét', 'T':'ti',
      'U':'diu', 'V':'vi', 'W':'đáp liu', 'X':'ếch',
      'Y':'oai', 'Z':'zi'
  }
  ```
- [ ] Build `_LETTER_TABLE` string từ dict trên, inject vào system prompt:
  ```
  Bảng phiên âm (BẮT BUỘC dùng đúng):
  A=ây, B=bi, C=si, D=đi, E=i, F=ép, G=ji, H=ét, I=ai ...
  ```
- [ ] Xóa toàn bộ hardcode phiên âm khác trong codebase

**Expected output:** Consistent letter pronunciation, không còn lỗi `CAGR → "gờ ây a rờ em"`.

---

### Bước 4 — Thiết kế Qwen prompt chống hallucination
**Mục tiêu:** Qwen chỉ làm đúng nhiệm vụ expand viết tắt, không paraphrase, không dịch.

- [ ] System prompt với hard constraints:
  ```
  KHÔNG dịch từ tiếng Anh sang tiếng Việt.
  KHÔNG paraphrase câu.
  KHÔNG thêm/bớt từ ngoài viết tắt cần expand.
  CHỈ thay thế ký hiệu/viết tắt bằng cách đọc.
  ```
- [ ] Few-shot examples cho từng loại lỗi đã ghi nhận ở Bước 1:
  ```
  Input:  "Generative AI đang phát triển"
  Output: "Generative ây ai đang phát triển"
  # KHÔNG: "GPU đang phát triển"

  Input:  "hệ thống IT nội bộ"
  Output: "hệ thống ai ti nội bộ"
  # KHÔNG: "hệ thống ai" (truncated)
  ```
- [ ] `temperature=0, top_k=1` cho deterministic output
- [ ] Request JSON schema output để validate dễ hơn

**Expected output:** Loại bỏ các hallucination pattern đã biết; output stable qua nhiều lần chạy.

---

### Bước 5 — Implement `_needs_llm()` gate
**Mục tiêu:** Skip Qwen với câu thuần Việt → giảm latency, giảm risk hallucinate.

- [ ] Detect câu cần Qwen dựa trên:
  ```python
  # Có ký tự Latin viết hoa liền 2+
  re.search(r'[A-Z]{2,}', text)
  # Có từ tiếng Anh (Latin lowercase không phải số)
  re.search(r'\b[a-z]{3,}\b', text)  # loại trừ "và", "là"...
  # Có ký hiệu kỹ thuật: /, %, @, #
  ```
- [ ] Log skip ratio để monitor (kỳ vọng ~40-60% câu thuần Việt được skip)

**Expected output:** Latency giảm đáng kể với văn bản hành chính thuần Việt.

---

### Bước 6 — `_expand_abbrevs_fallback()` sau Qwen
**Mục tiêu:** Safety net — nếu Qwen miss hoặc fail, rule-based vẫn catch được.

- [ ] Áp dụng `_EN_LETTER` dict để spell-out các token ALL-CAPS còn sót
- [ ] Áp dụng `ACRONYM_DICT` (JSON) cho các viết tắt hành chính phổ biến:
  ```json
  {
    "UBND": "ủy ban nhân dân",
    "HĐND": "hội đồng nhân dân",
    "NQ/TW": "nờ quy tê vê kép",
    "QĐ-TTg": "quyết định thủ tướng chính phủ"
  }
  ```
- [ ] Fallback chạy **sau** Qwen, không trước (tránh double-process)

**Expected output:** 0 trường hợp ALL-CAPS còn sót vào F5-TTS.

---

### Bước 7 — `_sanity_ok()` validation
**Mục tiêu:** Detect khi nào Qwen output bất thường, rollback về input gốc.

- [ ] Implement 3 checks:
  ```python
  def _sanity_ok(original: str, normalized: str) -> bool:
      # 1. Empty output
      if not normalized.strip():
          return False
      # 2. Length inflation > 4x (Qwen đang paraphrase)
      if len(normalized) > len(original) * 4:
          return False
      # 3. Diacritic word loss > 6%
      vi_words_in  = count_vietnamese_words(original)
      vi_words_out = count_vietnamese_words(normalized)
      if vi_words_in > 0 and vi_words_out / vi_words_in < 0.94:
          return False
      return True
  ```
- [ ] Nếu fail → log warning + dùng input gốc (không crash)

**Expected output:** Pipeline không bao giờ push garbage vào F5-TTS.

---

### Bước 8 — `diskcache` cho Qwen results
**Mục tiêu:** Văn bản hành chính lặp lại nhiều pattern → cache hit rate cao.

- [ ] Implement cache key = `md5(normalized_sentence)`
- [ ] Cache expire = 7 ngày
- [ ] Log cache hit/miss ratio
- [ ] Kỳ vọng: sau warm-up, >70% câu có viết tắt được serve từ cache

**Expected output:** Latency Qwen gần bằng 0 với văn bản thường xuyên dùng.

---

### Bước 9 — `normalize_for_f5()` chunking
**Mục tiêu:** F5-TTS hoạt động tốt nhất với chunk ngắn, split đúng chỗ.

- [ ] Split tại dấu câu: `.`, `!`, `?`, `…`, `;`
- [ ] Max chunk = 200 chars (dựa trên ViVoice context window)
- [ ] Không split giữa số đã expand (`"mười lăm phẩy năm tỷ"` không được bị cắt)
- [ ] Xử lý edge case: câu > 200 chars không có dấu câu → split tại dấu phẩy

**Expected output:** Không còn audio bị cắt câu giữa chừng hoặc mất đoạn.

---

### Bước 10 — Debug tooling + evaluation
**Mục tiêu:** Dễ reproduce lỗi, đo được cải thiện.

- [ ] `TTS_DEBUG=1` → log từng stage output ra stderr
- [ ] `TTS_DEBUG_LLM=1` → log raw Qwen prompt + response
- [ ] Eval script: chạy test set từ Bước 1, so sánh trước/sau
- [ ] Metric chính: % token đọc sai trên manual review
- [ ] Docker: đảm bảo Ollama URL tự detect `host.docker.internal:11434` khi trong container

**Expected output:** Có thể reproduce và fix bất kỳ lỗi nào trong < 10 phút.

---

## Thứ tự ưu tiên thực hiện

| Bước | Ưu tiên | Lý do |
|------|---------|-------|
| 1 (Audit) | 🔴 Làm trước | Không có baseline thì không đo được |
| 2 (Numbers) | 🔴 Cao nhất | Root cause của hallucination phổ biến nhất |
| 3 (Letter dict) | 🔴 Cao | Consistency issue ảnh hưởng mọi viết tắt |
| 4 (Prompt) | 🔴 Cao | Qwen behavior không predictable nếu không có |
| 7 (Sanity) | 🟡 Trung bình | Safety net, làm sớm để tự tin test |
| 5 (Gate) | 🟡 Trung bình | Latency, không ảnh hưởng accuracy |
| 6 (Fallback) | 🟡 Trung bình | Defense-in-depth |
| 8 (Cache) | 🟢 Sau | Optimization, không phải correctness |
| 9 (Chunking) | 🟢 Sau | Chỉ cần nếu có long-form text |
| 10 (Debug) | 🟢 Liên tục | Làm song song từ đầu |

---

## Expected outcomes tổng thể

| Metric | Hiện tại (ước tính) | Sau khi xong |
|--------|---------------------|--------------|
| Số + đơn vị đọc đúng | ~60% | >98% |
| Viết tắt kỹ thuật đúng | ~70% | >95% |
| Viết tắt hành chính đúng | ~50% | >95% |
| Hallucination rate | Cao (known patterns) | <2% |
| Latency (câu thuần Việt) | Gọi Qwen mỗi câu | 0 (skip) |
| Latency (câu có viết tắt, cached) | ~1-3s | <50ms |