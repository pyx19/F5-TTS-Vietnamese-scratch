# F5-TTS-Vietnamese-scratch

Bản viết lại của `F5-TTS-Vietnamese` theo [`plan.md`](plan.md): tách pipeline chuẩn hóa
văn bản (text normalization) ra khỏi script inference, thành package `text_pipeline/`
với từng stage độc lập, có thể test riêng.

Model TTS (F5-TTS / ViVoice) và checkpoint được **recycle** từ `../F5-TTS-Vietnamese`
— không copy lại, không huấn luyện lại.

## Kiến trúc pipeline

```
Raw Text
   │
   ▼
_normalize_linebreaks() / _remove_parens()      (pipeline.py — dọn cấu trúc)
   │
   ▼
[Stage 1] normalize_numbers() + vi_numbers()     (numbers.py — RULE-BASED, TRƯỚC LLM)
   │
   ▼
[Stage 2] needs_llm() gate                       (gate.py — skip câu thuần Việt)
   │
   ▼
[Stage 3] llm_normalize() / Qwen                 (llm_normalize.py — per-sentence,
   │                                               có cache + sanity_ok bên trong)
   ▼
[Stage 4] expand_acronyms_fallback()             (acronyms.py — dict hành chính)
          + expand_abbrevs_fallback()            (letters.py — spell ALL-CAPS còn sót)
   │
   ▼
lowercase + cleanup dấu nối
   │
   ▼
[Stage 6] sanitize_for_vivoice()                 (sanitize.py — strip ký tự ngoài vocab)
   │
   ▼
normalize_for_f5()                               (chunking.py — chunk ≤ max_chars)
   │
   ▼
F5-TTS / ViVoice inference (infer_vivoice.py)
```

## Cấu trúc thư mục

```
text_pipeline/          package chuẩn hóa văn bản — xem plan.md cho từng Bước
  numbers.py             Bước 2 — số + đơn vị
  letters.py              Bước 3 — bảng phiên âm chữ cái (single source of truth)
  llm_normalize.py        Bước 3/4 — prompt Qwen + gọi LLM (OpenAI-compatible API)
  gate.py                 Bước 5 — skip LLM cho câu thuần Việt
  acronyms.py             Bước 6 — ACRONYM_DICT (viết tắt hành chính)
  acronym_dict.json       data cho acronyms.py
  sanity.py               Bước 7 — validate output LLM, rollback nếu bất thường
  cache.py                Bước 8 — diskcache cho kết quả LLM (tự tắt nếu chưa cài)
  chunking.py             Bước 9 — normalize_for_f5(), chunk ≤ max_chars
  sanitize.py             Stage 6 — strip ký tự ngoài vocab.txt
  debug.py                Bước 10 — TTS_DEBUG / TTS_DEBUG_LLM, Docker Ollama URL
  pipeline.py             orchestrator — preprocess_text() / preprocess_and_chunk()

eval/
  test_sentences.json     52 câu test (số/đơn vị, viết tắt hành chính, kỹ thuật,
                           code-switching, tên riêng)
  audit.py                chạy test set qua pipeline, log theo nhóm, so sánh baseline
  baseline_rule_based.json  kết quả audit không dùng LLM (rule-based only)

tests_unit/              unit test (unittest, không cần pytest) cho từng module

src/f5_tts/              model code F5-TTS (recycle nguyên bản từ F5-TTS-Vietnamese)
ckpts/                   junction trỏ về ../F5-TTS-Vietnamese/ckpts (không copy weights)

infer_vivoice.py         CLI inference — dùng text_pipeline + infer_batch_process
gradio_app.py            UI Gradio — gọi infer_vivoice.run()
download_vivoice.py      tải checkpoint ViVoice từ HuggingFace (recycle)

Dockerfile                    build FULL BAKE — code + model + deps cùng 1 image
docker-compose.yml             compose cho Dockerfile (full bake)
Dockerfile.deps-only           build DEPS-ONLY — chỉ model + deps, code mount ngoài
docker-compose.deps-only.yml   compose cho Dockerfile.deps-only
docker-entrypoint.sh           entrypoint dùng chung cho cả 2 image
```

## Cài đặt

Dùng chung `.venv` với `F5-TTS-Vietnamese` (đã cài F5-TTS + phụ thuộc). Cần thêm:

```bash
pip install diskcache   # Bước 8 — cache kết quả LLM normalize (tùy chọn, tự tắt nếu thiếu)
```

> Máy dev hiện tại **không có kết nối mạng** nên chưa cài được `diskcache` (pip install
> fail do không resolve được pypi.org) — `cache.py` tự phát hiện thiếu package và
> no-op êm (không raise lỗi), pipeline vẫn chạy đúng, chỉ mất lợi ích cache. Cài lại khi
> có mạng để bật cache thật.

**Ollama**: đã kiểm tra trên máy — server đang chạy tại `http://localhost:11434` với
model `qwen2.5:3b` (3.1B, Q4_K_M) sẵn sàng dùng cho `--llm_normalize`.

`ckpts/` là junction tới model đã tải sẵn ở `../F5-TTS-Vietnamese/ckpts`. Nếu chạy độc
lập không có project gốc, tải lại bằng:

```bash
python download_vivoice.py
```

## Chạy inference

```bash
python infer_vivoice.py \
  --ref_audio ref.wav \
  --ref_text  "cả hai bên hãy cố gắng hiểu cho nhau" \
  --gen_text  "Hệ thống AI và Machine Learning giúp CEO ra quyết định." \
  --llm_normalize --ollama_model qwen2.5:3b
```

Hoặc UI: `python gradio_app.py`

Không dùng LLM (rule-based only, nhanh hơn nhưng kém chính xác với viết tắt hiếm):
bỏ `--llm_normalize`.

Debug từng bước:

```bash
TTS_DEBUG=1 python infer_vivoice.py ...       # log mỗi stage preprocessing
TTS_DEBUG_LLM=1 python infer_vivoice.py ...   # log raw prompt/response của Qwen
```

## Trỏ vào LLM tự host

Qwen2.5:3b qua Ollama trên laptop chỉ là **placeholder mặc định trong code** — ở môi
trường thật (đặc biệt khi deploy công ty), trỏ sang LLM bạn tự host qua 3 biến môi
trường (đọc bởi `text_pipeline/debug.py`, ưu tiên hơn giá trị mặc định hard-code):

| Biến           | Ý nghĩa                                          | Mặc định nếu không set                     |
|----------------|---------------------------------------------------|---------------------------------------------|
| `OLLAMA_URL`   | URL server OpenAI-compatible (`/v1/chat/completions`) | `http://localhost:11434` (hoặc `http://host.docker.internal:11434` nếu chạy trong Docker) |
| `OLLAMA_MODEL` | Tên model trên server đó                          | `qwen2.5:3b`                                 |
| `LLM_API_KEY`  | Bearer token nếu server yêu cầu xác thực           | `""` (không gửi header Authorization)        |

Hoạt động với **mọi** server OpenAI-compatible, không chỉ Ollama: vLLM, LM Studio,
Xinference, TGI, hay API tự viết miễn implement đúng `/v1/chat/completions`.

```bash
# CLI — set qua biến môi trường
OLLAMA_URL="http://10.0.0.5:8000/v1" OLLAMA_MODEL="qwen2.5-14b-instruct" \
  python infer_vivoice.py --ref_audio ref.wav --ref_text "..." --gen_text "..." --llm_normalize

# Hoặc qua flag CLI (override cả biến môi trường)
python infer_vivoice.py ... --llm_normalize \
  --ollama_url http://10.0.0.5:8000/v1 --ollama_model qwen2.5-14b-instruct --llm_api_key sk-xxx
```

Gradio UI: 3 ô trong accordion **"⚙️ LLM Settings"** đã tự điền sẵn giá trị từ các biến
môi trường trên (nếu có) — sửa trực tiếp trên UI cũng được, không cần restart.

## Deploy bằng Docker

Có **2 cách build**, tùy nhu cầu có cần sửa code sau khi deploy hay không. Dùng chung
`docker-entrypoint.sh`, khác nhau ở việc code có nằm trong image hay không.

### Cách 1 — Full bake (`Dockerfile` + `docker-compose.yml`)

Bake toàn bộ: code + model weights (~10GB) + dependencies vào 1 image. Đơn giản nhất,
nhưng sửa code phải rebuild image.

```bash
docker build -t f5tts-vi:latest .
docker push <user>/f5tts-vi:cu121
```

**Về câu hỏi "bake xong đẩy Docker Hub, pull xuống máy công ty không có mạng thì có
vấn đề gì không":**

- Image sau khi build đã tự chứa toàn bộ code + weights + Python deps (`HF_HUB_OFFLINE=1`,
  `TRANSFORMERS_OFFLINE=1` chặn mọi gọi mạng HuggingFace lúc runtime) → **chạy inference
  hoàn toàn offline được**, không cần internet sau khi `docker pull`/`docker load` xong.
- Máy đích **vẫn cần** Docker daemon + `nvidia-container-toolkit` đã cài sẵn (để chạy
  `--gpus all`) và có GPU NVIDIA tương thích CUDA 12.1 driver — kiểm tra trước.
- Nếu máy đích **hoàn toàn không có mạng** (không reach được cả Docker Hub), `docker pull`
  sẽ fail — dùng `docker save | gzip` trên máy có mạng rồi copy file `.tar.gz` qua
  USB/SCP nội bộ, `docker load` trên máy đích (đã ghi trong comment đầu `Dockerfile`).
  Nếu máy đích reach được Docker Hub (có mạng nội bộ/proxy ra ngoài dù không "internet
  chung") thì `docker pull` bình thường là đủ.
- **Không có vấn đề gì với model/deps** vì đã bake sẵn — nhưng lưu ý riêng
  `--llm_normalize` vẫn là 1 lệnh gọi HTTP tại thời điểm chạy: container cần **reach
  được LLM server** (qua `OLLAMA_URL`) trên mạng LAN/nội bộ — không cần internet cho
  việc này, chỉ cần container thấy được host/IP đó (mở port, đúng subnet/VPN nội bộ).
  Nếu LLM server chạy ngay trên máy host (ngoài Docker) và container không tự thấy
  `localhost` của host, dùng `host.docker.internal` (Windows/Mac tự có sẵn; trên Linux
  cần thêm `extra_hosts: ["host.docker.internal:host-gateway"]` vào compose) hoặc set
  `OLLAMA_URL` trỏ thẳng IP LAN của máy đó.
- Bug đã sửa trong `Dockerfile` gốc: thiếu `COPY text_pipeline/` (nếu bake mà không có
  dòng này, container sẽ crash `ModuleNotFoundError: No module named 'text_pipeline'`
  ngay khi chạy — đã fix, đã có `COPY text_pipeline/` + `COPY eval/`).

### Cách 2 — Deps-only (`Dockerfile.deps-only` + `docker-compose.deps-only.yml`)

Chỉ bake **model weights + dependencies** (nặng, ít đổi) vào image; **code ứng dụng
không nằm trong image** — mount từ một thư mục đã `git clone` sẵn ở ngoài. Sửa
`text_pipeline/`, `infer_vivoice.py`, `gradio_app.py`... trực tiếp trên máy công ty mà
KHÔNG cần rebuild/re-push image — chỉ cần `docker compose restart`.

```bash
# 1. Build + push 1 lần (khi model/deps đổi, không cần build lại khi chỉ sửa code)
docker build -f Dockerfile.deps-only -t <user>/f5tts-vi-deps:cu121 .
docker push <user>/f5tts-vi-deps:cu121

# 2. Trên máy có mạng: git clone repo (hoặc copy thư mục này qua USB/SCP nếu máy
#    công ty không có mạng để tự clone)
git clone <repo-url> F5-TTS-Vietnamese-scratch

# 3. Trên máy công ty
cd F5-TTS-Vietnamese-scratch
docker pull <user>/f5tts-vi-deps:cu121      # hoặc docker load nếu không có mạng
docker compose -f docker-compose.deps-only.yml up -d

# Sửa code xong:
docker compose -f docker-compose.deps-only.yml restart
```

Cách hoạt động: `docker-compose.deps-only.yml` mount `./:/app` (thư mục code đã clone)
vào container; model weights nằm ở `/opt/f5tts/ckpts` **bên trong image**, tách biệt
khỏi `/app` nên không bị mount đè (`infer_vivoice.py` đọc vị trí này qua biến môi
trường `F5TTS_CKPTS_DIR`, đã set sẵn trong compose file).

Nếu muốn container tự `git pull` code thay vì bind-mount (chỉ dùng được khi container
reach được Git remote — nội bộ hoặc qua proxy công ty, dù không có internet chung),
set `GIT_REPO_URL` (+ `GIT_REPO_BRANCH`) trong `docker-compose.deps-only.yml` thay vì
mount `./:/app` — `docker-entrypoint.sh` sẽ tự clone vào `/app` nếu thư mục đang trống
mỗi lần container khởi động.

## Audit / test

```bash
# Unit test từng module (numbers, letters, gate, acronyms, sanity, cache, chunking, sanitize, pipeline)
python -m unittest discover -s tests_unit -t .

# Audit 52 câu test set — rule-based only
python eval/audit.py --save eval/baseline_rule_based.json

# Audit với LLM thật (cần Ollama đang chạy model qwen2.5:3b)
python eval/audit.py --llm --save eval/baseline_llm.json

# So sánh với baseline đã lưu sau khi sửa pipeline
python eval/audit.py --llm --diff eval/baseline_llm.json
```

## Khác biệt so với `F5-TTS-Vietnamese` gốc

- Text pipeline tách thành module riêng, test được độc lập (không cần load model TTS).
- **Mới**: `acronyms.py` + `acronym_dict.json` — fallback dict cho viết tắt hành chính
  (UBND, HĐND, TTr, BC, TTg...) chạy TRƯỚC letter-spelling, tránh đọc "UBND" thành
  "u bi en đi".
- **Mới**: `cache.py` — diskcache theo `md5(model + câu)`, expire 7 ngày.
- **Mới**: `chunking.py` — `normalize_for_f5()` chunk Việt-aware (ranh giới câu → dấu
  phẩy → khoảng trắng), dùng trực tiếp với `infer_batch_process()` thay vì chunker
  generic byte-length của F5-TTS gốc.
- **Mới**: `sanitize.py` — strip ký tự ngoài `vocab.txt` trước khi đưa vào model.
- `eval/audit.py` + test set 52 câu — công cụ đo baseline/regression cho pipeline.

### Giới hạn đã biết (phát hiện qua audit — cả rule-based-only và chạy thật với Qwen2.5:3b qua Ollama local)

- Không dùng LLM: viết tắt hành chính hiếm ngoài `acronym_dict.json` (vd tên viết tắt
  bộ/ngành như "BTTTT", "BTP") bị letter-spell từng chữ thay vì đọc thành từ.
- `vi_numbers()` chạy TRƯỚC LLM nên số đứng cạnh tên sản phẩm có khoảng trắng (vd
  "RTX 5060") bị chuyển thành chữ số Việt trước khi LLM kịp thấy ngữ cảnh — kể cả khi
  bật `--llm_normalize`, vì thứ tự stage cố ý xử lý số trước để tránh Qwen hallucinate.
- **Đã phát hiện & sửa** qua audit thật với Qwen2.5:3b: model đôi khi bỏ sót việc expand
  viết tắt hành chính và trả về dạng chữ thường chưa mở rộng (vd "36-NQ/TW" → "nq tw"
  thay vì "nghị quyết trung ương"). `acronyms.py` trước đây match case-sensitive nên bỏ
  lỡ các trường hợp này — đã sửa thành match case-insensitive để "safety net" luôn bắt
  được bất kể Qwen trả về hoa hay thường.
- Qwen2.5:3b (3B, chạy local) đôi khi vẫn phiên âm sai 1 chữ cái trong từ viết tắt dài
  (vd "F5-TTS" → "f5-et tts" thay vì "ép 5 ti ti ét", "GPT-4o" thiếu 1 âm tiết) hoặc
  hallucinate token lạ đứng cạnh số (vd "QH15" → "quh mười lăm"). `sanity_ok()` không
  bắt được các lỗi này vì chỉ kiểm tra rỗng/phình/mất từ tiếng Việt có dấu, không kiểm
  tra token không dấu bị biến dạng — giới hạn cố hữu của model 3B, không phải bug pipeline.
  Xem `eval/baseline_llm.json` (chạy thật, đã lưu) để đối chiếu cụ thể từng câu.
