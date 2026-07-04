"""
Bước 3/4 — LLM normalize: Qwen xử lý viết tắt kỹ thuật + quyết định theo ngữ cảnh.

Thiết kế (sau phân tích lỗi thực tế trên văn bản hành chính + kỹ thuật):
  - Số được xử lý TRƯỚC LLM bằng rule-based (xem numbers.py) → loại bỏ điểm yếu
    lớn nhất của Qwen2.5-3B: hallucinate khi phải xử lý số phức tạp
    (vd "15.5 tỷ USD" → LLM tự bịa "15% 50 triệu").
  - Viết tắt hành chính (UBND, QĐ, NQ...) cũng được xử lý TRƯỚC LLM bằng
    ACRONYM_DICT cứng (xem acronyms.py + pipeline.py) — cùng lý do: Qwen2.5:3B
    đôi khi "sửa" sai một viết tắt đã có nghĩa rõ ràng, không nhất quán, và vì
    ACRONYM_DICT chạy sau khi kết quả LLM đã cố định nên fallback cũ KHÔNG sửa lại
    được nếu LLM đã biến dạng text gốc. Chuyển lên trước LLM để loại hẳn rủi ro này.
  - LLM chỉ còn nhiệm vụ: viết tắt KỸ THUẬT (tiếng Anh) + quyết định theo ngữ cảnh.
  - Bảng chữ cái inject từ letters.EN_LETTER → không bao giờ mâu thuẫn với code.
  - temperature=0 cho deterministic output.
  - Kết quả được cache theo diskcache (Bước 8) để giảm latency với văn bản lặp.
"""

import json
import re
import urllib.request

from . import cache
from .debug import DBG_LLM, dbg
from .gate import SkipRatioTracker, needs_llm
from .letters import LETTER_TABLE
from .sanity import contains_cjk, sanity_ok

_TTS_NORM_SYSTEM = """Bạn là công cụ tiền xử lý văn bản cho hệ thống TTS (Text-to-Speech) tiếng Việt.
Nhiệm vụ: nhận một câu đã được xử lý sẵn về số, chuyển viết tắt và ký hiệu còn lại thành dạng đọc được.
CHỈ trả về câu đã chỉnh, không giải thích."""

_TTS_NORM_USER_TEMPLATE = """Chuẩn hóa câu sau để TTS tiếng Việt đọc tự nhiên. Dùng ngữ cảnh câu để phán đoán đúng.

QUAN TRỌNG: Mọi số trong câu ĐÃ được chuyển thành chữ trước đó. KHÔNG thay đổi bất kỳ số nào còn sót lại, KHÔNG thêm "phần trăm", KHÔNG đổi đơn vị.

━━━ NGUYÊN TẮC ━━━

[A] PHÂN TÍCH NGỮ CẢNH TRƯỚC KHI XỬ LÝ:
  • Câu đưa vào ĐÃ được xử lý sẵn viết tắt hành chính tiếng Việt (quyết định, ủy
    ban nhân dân, nghị định, thông tư, nghị quyết, chỉ thị, thành phố, trung
    ương, chính phủ, quốc hội, giáo sư, phó giáo sư, tiến sĩ, thạc sĩ, bác sĩ,
    kỹ sư, cử nhân...) — đây LÀ KẾT QUẢ ĐÚNG, GIỮ NGUYÊN, không dịch ngược lại
    thành viết tắt, không diễn giải lại, không paraphrase.
  • Việc còn lại của bạn là viết tắt KỸ THUẬT/TIẾNG ANH (AI, CPU, API, SDK, IT, LLM)
  • Từ tiếng Việt in hoa (TỈNH, THÀNH PHỐ, VIỆT NAM) → viết thường, KHÔNG spell từng chữ

[B] VIẾT TẮT TIẾNG ANH KỸ THUẬT → phiên âm từng chữ cái ĐÚNG theo bảng sau:
  Bảng tên chữ cái Anh-Mỹ (BẮT BUỘC dùng đúng bảng này, KHÔNG được suy đoán):
  {letter_table}

  Ví dụ BẮT BUỘC ĐÚNG — sai một chữ là lỗi:
  AI →"ây ai"          IT →"ai ti"          LLM→"eo eo em"
  API→"ây pi ai"       CPU→"xi pi iu"       GPU→"gờ pi iu"
  SDK→"ét đi cây"      HTTP→"ếch ti ti pi"  URL→"iu a rờ eo"
  ROI→"a rờ âu ai"     KPI→"cây pi ai"      CEO→"xi i âu"
  GDP→"gờ đi pi"       CAGR→"xi ây gờ a rờ" HR→"ếch a rờ"
  USD→"iu ét đi"       VRAM→"vi a rờ ây em"

  CHÚ Ý ĐẶC BIỆT — các lỗi phổ biến cần tránh:
  • IT phải là "ai ti" (2 âm tiết) — KHÔNG được rút gọn thành "ai" dù đứng cuối câu
  • LLM phải là "eo eo em" (3 âm tiết) — không phải "eo ai" hay "eo em"
  • Generative AI: "Generative" giữ nguyên, chỉ spell "AI"→"ây ai"
    → Đúng: "Generative ây ai"   Sai: "Gờ pi iu" (không được thay Generative AI bằng GPU)
  • CAGR phải là "xi ây gờ a rờ" — không thêm "em" hay chữ nào thừa

  Ngoại lệ (đọc như từ, không spell): COVID→"cô vít"  NATO→"na tô"  ASEAN→"a xê an"

[C] VIẾT TẮT MIXED-CASE → phiên âm từng chữ theo bảng [B]:
  SaaS→"ét ây ây ét"   IoT→"ai âu ti"   OAuth→"âu âu ếch"
  ChatGPT→"Chat gờ pi ti"   DevOps→"DevOps" (đọc tự nhiên)

[D] TÊN SẢN PHẨM MIXED ALPHANUMERIC → spell chữ + giữ nguyên số:
  RTX 4090→"a rờ ti ích 4090"   GPT-4o→"gờ pi ti 4o"
  M2 Pro→"em 2 pro"              F5-TTS→"ép 5 ti ti ét"

[E] SỐ VÀ ĐƠN VỊ — ĐÃ ĐƯỢC XỬ LÝ TRƯỚC, KHÔNG THAY ĐỔI:
  Tất cả số trong câu đã được rule-based xử lý. Nhiệm vụ của bạn là KHÔNG đụng vào số.
  • "mười lăm phẩy năm tỷ đô la" → giữ nguyên, KHÔNG đổi thành "phần trăm"
  • "hai trăm ms" → giữ nguyên
  • Đơn vị lowercase còn lại (gb, ghz, mbps) → giữ nguyên

[F] CỤM TỪ KỸ THUẬT CÓ GẠCH NGANG → bỏ gạch ngang:
  End-to-end→"end to end"   Text-to-Speech→"text to speech"   fine-tuning→"fine tuning"

[G] GIỮ NGUYÊN — KHÔNG thay đổi, KHÔNG dịch:
  • Tiếng Việt thông thường, kể cả cụm hành chính đã mở rộng sẵn (xem mục [A])
  • Thuật ngữ kỹ thuật ghép (KHÔNG dịch sang Việt):
    On-premise (KHÔNG dịch thành "trên premises")
    Cloud API (KHÔNG tách, KHÔNG đổi thứ tự)
    Machine Learning, Digital Transformation, Text-to-Speech, End-to-end
    SaaS platform (KHÔNG bỏ "platform")
  • Từ Anh thông dụng: deploy, server, model, training, inference, platform,
    cloud, internet, framework, backend, startup, pipeline, workflow, encryption

━━━ VÍ DỤ ━━━

Câu: "Generative AI và Machine Learning đã trở thành lĩnh vực phát triển nhanh nhất của IT."
Kết quả: "Generative ây ai và Machine Learning đã trở thành lĩnh vực phát triển nhanh nhất của ai ti."

Câu: "Các tập đoàn đang đầu tư hơn mười lăm phẩy năm tỷ đô la vào nghiên cứu các hệ thống LLM tiên tiến."
Kết quả: "Các tập đoàn đang đầu tư hơn mười lăm phẩy năm tỷ đô la vào nghiên cứu các hệ thống eo eo em tiên tiến."

Câu: "Hệ thống Text-to-Speech triển khai theo mô hình On-premise qua Cloud API."
Kết quả: "Hệ thống text to speech triển khai theo mô hình On-premise qua Cloud ây pi ai."

Câu: "Điều này giúp tối ưu hóa ROI lên đến bốn mươi lăm phần trăm nhờ cơ chế End-to-end Encryption."
Kết quả: "Điều này giúp tối ưu hóa a rờ âu ai lên đến bốn mươi lăm phần trăm nhờ cơ chế end to end Encryption."

Câu: "Với tốc độ CAGR đạt hai mươi hai phẩy bốn phần trăm, các SaaS platform sẽ phủ sóng rộng."
Kết quả: "Với tốc độ xi ây gờ a rờ đạt hai mươi hai phẩy bốn phần trăm, các ét ây ây ét platform sẽ phủ sóng rộng."

Câu: "Theo quyết định mười lăm năm hai nghìn không trăm hai mươi ba của ủy ban nhân dân tỉnh, hệ thống AI sẽ deploy trên GPU RTX 4090 với hai mươi bốn gb VRAM."
Kết quả: "Theo quyết định mười lăm năm hai nghìn không trăm hai mươi ba của ủy ban nhân dân tỉnh, hệ thống ây ai sẽ deploy trên gờ pi iu a rờ ti ích 4090 với hai mươi bốn gb vi a rờ ây em."
# Chú ý: "quyết định", "ủy ban nhân dân" GIỮ NGUYÊN — đã được xử lý sẵn trước khi
# tới bạn, KHÔNG được đổi lại thành "QĐ"/"UBND" hay diễn giải khác đi.

━━━━━━━━━━━━━━━━━━━━━━━━━━━

Câu: {text}
Kết quả:"""


def _split_sentences(text: str) -> list[str]:
    """
    Tách câu theo \\n rồi tách thêm trên [.!?] + chữ hoa.
    Giữ cấu trúc văn bản pháp luật (mỗi "Căn cứ..." trên 1 dòng).
    """
    result = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        segmented = re.sub(r"([.!?])\s+(?=[A-ZĐĂÂÊÔƯƠÀ-ɏ])", r"\1\n", line)
        result.extend(s.strip() for s in segmented.split("\n") if s.strip())
    return result


def _call_llm(sentence: str, model: str, url: str, timeout: int,
               idx: int, api_key: str = "") -> str | None:
    """
    Gọi LLM qua OpenAI-compatible API (/v1/chat/completions) cho một câu.
    Tương thích: vLLM, LM Studio, Xinference, Ollama.
    Trả về None nếu lỗi kết nối, CJK slip, hoặc sanity check thất bại.
    """
    cached = cache.get(model, sentence)
    if cached is not None:
        if DBG_LLM:
            print(f"\n  [LLM_IN  #{idx}] {sentence}  (cache hit)", flush=True)
        return cached

    user_content = _TTS_NORM_USER_TEMPLATE.format(
        letter_table=LETTER_TABLE,
        text=sentence,
    )

    if DBG_LLM:
        print(f"\n  [LLM_IN  #{idx}] {sentence}", flush=True)

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": _TTS_NORM_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "temperature": 0,
        "max_tokens": min(len(sentence) * 4, 512),
    }).encode()

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = json.loads(resp.read())["choices"][0]["message"]["content"].strip()

        if DBG_LLM:
            print(f"  [LLM_OUT #{idx}] {raw}", flush=True)

        # Strip prefix LLM hay tự thêm ("Kết quả:", quotes...)
        result = re.sub(r"^(?:kết quả|output|câu)[:\s]*", "", raw, flags=re.IGNORECASE).strip()
        result = result.strip("\"'")

        if contains_cjk(result):
            print(f"  [WARN #{idx}] Chứa CJK → fallback rule-based cho câu này")
            return None

        if not sanity_ok(sentence, result, idx, dbg=dbg):
            print(f"  [WARN #{idx}] Sanity fail → giữ câu gốc")
            return None

        cache.set(model, sentence, result)
        return result

    except Exception as e:
        print(f"  [WARN #{idx}] {type(e).__name__}: {e}")
        return None


def llm_normalize(text: str, model: str = "qwen2.5:3b",
                   url: str = "http://localhost:11434",
                   timeout: int = 60, api_key: str = "") -> str | None:
    """
    Normalize text cho TTS bằng LLM, per-sentence.
    Returns None khi LLM hoàn toàn không kết nối được (câu đầu tiên fail).
    """
    sentences = _split_sentences(text)
    total = len(sentences)
    print(f"[*] LLM normalize: {total} câu | model={model} | url={url}")

    results = []
    ollama_ok = True
    skip_tracker = SkipRatioTracker()

    for i, sent in enumerate(sentences):
        label = f"[{i + 1}/{total}]"

        skip = not needs_llm(sent)
        skip_tracker.record(skip)
        if skip:
            dbg("LLM", f"{label} SKIP (no token cần xử lý): {sent[:70]}")
            results.append(sent)
            continue

        if not ollama_ok:
            results.append(sent)
            continue

        normalized = _call_llm(sent, model, url, timeout, i + 1, api_key)

        if normalized is None and not results:
            # Lỗi câu đầu → LLM có thể down hoàn toàn
            ollama_ok = False
            results.append(sent)
            return None

        if normalized is None:
            results.append(sent)  # giữ gốc, rule-based xử lý sau
        else:
            preview = normalized[:90] + ("..." if len(normalized) > 90 else "")
            print(f"  {label} {preview}")
            results.append(normalized)

    print(f"[*] LLM gate: {skip_tracker.summary()} | {cache.STATS.summary()}")
    return " ".join(results)
