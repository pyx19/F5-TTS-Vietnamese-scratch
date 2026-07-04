"""
Bước 9 (mở rộng) — llm_split_points(): nhờ LLM chèn điểm ngắt ngữ nghĩa cho các
đoạn quá dài mà không còn dấu câu/dấu phẩy để cắt.

Vấn đề: chunking.py rule-based, hết dấu câu/phẩy thì cắt mù theo số ký tự gần
max_chars — dễ cắt giữa 1 cụm danh từ liền, vd:
  "...chủ trương đầu tư dự án trọng điểm / quốc gia; căn cứ nghị quyết..."
("dự án trọng điểm quốc gia" là 1 cụm, không nên tách rời "trọng điểm" khỏi
"quốc gia" chỉ vì chạm giới hạn ký tự).

Chỉ dùng khi có llm_model (tùy chọn, giống llm_normalize). Validate NGHIÊM NGẶT
— so khớp nội dung sau khi bỏ hết khoảng trắng phải giữ NGUYÊN 100% so với input,
LLM chỉ được phép chèn dấu phân đoạn, không được thêm/bớt/đổi từ nào. Sai thì trả
None, chunking.py tự fallback về cách cắt theo khoảng trắng cũ — không bao giờ
làm hỏng văn bản dù LLM có lỗi.
"""

import json
import re
import urllib.request

from . import cache
from .debug import DBG_LLM, dbg

_CHUNK_MARK = "|||"

_CHUNK_SYSTEM = (
    "Bạn là công cụ chia câu tiếng Việt thành các đoạn ngắn để đọc thành tiếng. "
    "CHỈ trả về câu đã chèn dấu phân đoạn, không giải thích, không thêm bớt từ nào."
)

_CHUNK_USER_TEMPLATE = """Chèn dấu "{mark}" vào câu sau tại các điểm ngắt hợp lý về ngữ nghĩa (ranh giới cụm từ/mệnh đề), sao cho mỗi đoạn giữa 2 dấu "{mark}" không quá {max_chars} ký tự.

BẮT BUỘC:
- KHÔNG thêm, bớt, hay đổi bất kỳ từ nào trong câu — CHỈ được chèn dấu "{mark}".
- KHÔNG chèn "{mark}" vào giữa một cụm danh từ/số/tên riêng liền nhau (vd cụm
  "dự án trọng điểm quốc gia" phải giữ nguyên trọn cụm, không tách rời).
- Mỗi đoạn nên là 1 cụm từ/mệnh đề trọn vẹn về nghĩa, không cắt nửa chừng.

Câu: {text}
Kết quả:"""


def llm_split_points(
    text: str,
    max_chars: int,
    model: str,
    url: str,
    timeout: int = 30,
    api_key: str = "",
) -> list[str] | None:
    """
    Trả về list các đoạn theo gợi ý ngắt của LLM, hoặc None nếu LLM lỗi/kết quả
    không hợp lệ (chunking.py tự fallback khi nhận None).
    """
    cache_model_key = f"{model}::chunk{max_chars}"
    cached = cache.get(cache_model_key, text)
    if cached is not None:
        return json.loads(cached)

    user_content = _CHUNK_USER_TEMPLATE.format(mark=_CHUNK_MARK, max_chars=max_chars, text=text)

    if DBG_LLM:
        print(f"\n  [CHUNK_LLM_IN] {text}", flush=True)

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": _CHUNK_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "temperature": 0,
        "max_tokens": min(len(text) * 3, 1024),
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
    except Exception as e:
        dbg("CHUNK_LLM", f"lỗi gọi LLM: {type(e).__name__}: {e}")
        return None

    if DBG_LLM:
        print(f"  [CHUNK_LLM_OUT] {raw}", flush=True)

    raw = re.sub(r"^(?:kết quả|output|câu)[:\s]*", "", raw, flags=re.IGNORECASE).strip()
    raw = raw.strip("\"'")

    parts = [p.strip() for p in raw.split(_CHUNK_MARK) if p.strip()]
    if len(parts) < 2:
        dbg("CHUNK_LLM", "LLM không chèn được điểm ngắt hữu ích — bỏ qua")
        return None

    # Validate: nội dung (bỏ khoảng trắng) phải khớp 100% với input — LLM chỉ
    # được chèn dấu phân đoạn, không được thêm/bớt/đổi từ.
    original_no_space = re.sub(r"\s+", "", text)
    rejoined_no_space = re.sub(r"\s+", "", "".join(parts))
    if original_no_space != rejoined_no_space:
        dbg("CHUNK_LLM", "validate fail — LLM đã đổi nội dung, bỏ qua gợi ý")
        return None

    cache.set(cache_model_key, text, json.dumps(parts, ensure_ascii=False))
    return parts
