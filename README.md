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
_normalize_linebreaks() / _normalize_symbols() / _remove_brackets()
   │                                                (pipeline.py — dọn cấu trúc + ký hiệu)
   ▼
[Stage 1] normalize_numbers() + vi_numbers()     (numbers.py — RULE-BASED, TRƯỚC LLM;
   │                                               bao gồm cả mã văn bản NĐ-CP/QH15/...
   │                                               SPELL kiểu Việt, không dịch nghĩa)
   ▼
expand_acronyms_fallback()                       (acronyms.py — dict hành chính CỨNG,
   │                                               TRƯỚC LLM, cùng lý do với số)
   ▼
[Stage 2] needs_llm() gate                       (gate.py — skip câu thuần Việt)
   │
   ▼
[Stage 3] llm_normalize() / Qwen                 (llm_normalize.py — chỉ còn viết tắt
   │                                               KỸ THUẬT + context, có cache + sanity_ok)
   ▼
[Stage 4] expand_abbrevs_fallback()              (letters.py — spell ALL-CAPS còn sót)
   │
   ▼
lowercase + cleanup dấu nối
   │
   ▼
[Stage 6] sanitize_for_vivoice()                 (sanitize.py — strip ký tự ngoài vocab)
   │
   ▼
normalize_for_f5()                               (chunking.py — chunk ≤ max_chars,
   │                                               dùng chunk_llm.py hỗ trợ ngắt ngữ
   │                                               nghĩa nếu có llm_model)
   ▼
F5-TTS / ViVoice inference (infer_vivoice.py)
```

## Cấu trúc thư mục

```
text_pipeline/          package chuẩn hóa văn bản — xem plan.md cho từng Bước
  numbers.py             Bước 2 — số + đơn vị + ngày/tháng/năm + tỷ lệ + khoảng năm
  letters.py              Bước 3 — bảng phiên âm chữ cái (single source of truth)
  llm_normalize.py        Bước 3/4 — prompt Qwen + gọi LLM (OpenAI-compatible API),
                          CHỈ còn xử lý viết tắt kỹ thuật (viết tắt hành chính đã
                          bị acronyms.py xử lý trước đó, LLM không còn thấy nữa)
  gate.py                 Bước 5 — skip LLM cho câu thuần Việt
  acronyms.py             Bước 6 — ACRONYM_DICT (viết tắt hành chính), chạy TRƯỚC
                          LLM (xem "Khác biệt so với gốc" — lý do đổi thứ tự)
  acronym_dict.json       84 mục — 22 gốc + 62 mục gộp từ
                          20260701_Danh_sach_tu_viet_tat.xlsx (đơn vị/chức danh/
                          chỉ tiêu nội bộ) — CHỈ áp dụng cho viết tắt đứng riêng
                          trong câu, KHÔNG áp dụng cho mã văn bản (xem numbers.py)
  sanity.py               Bước 7 — validate output LLM, rollback nếu bất thường
  cache.py                Bước 8 — diskcache cho kết quả LLM (tự tắt nếu chưa cài)
  chunking.py             Bước 9 — normalize_for_f5(), chunk ≤ max_chars
  chunk_llm.py            LLM hỗ trợ chunking — chọn điểm ngắt ngữ nghĩa khi 1 đoạn
                          quá dài mà không còn dấu câu/phẩy, có validate + fallback
  sanitize.py             Stage 6 — strip ký tự ngoài vocab.txt
  debug.py                Bước 10 — TTS_DEBUG / TTS_DEBUG_LLM, Docker Ollama URL
  pipeline.py             orchestrator — preprocess_text() / preprocess_and_chunk()

eval/
  test_sentences.json     62 câu test (số/đơn vị, viết tắt hành chính, kỹ thuật,
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
docker-entrypoint.sh           entrypoint dùng chung cho cả 2 image (LF — xem .gitattributes)
.gitattributes                 ép LF cho *.sh/Dockerfile*/docker-compose*.yml
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

**Checklist việc của máy nhà** (trước khi coi là xong, để giao cho máy công ty):

- [x] Push code lên GitHub
- [ ] Build image (cả 2 variant nếu cần) — dùng đúng `--build-context ckpts=...`
- [ ] **Smoke-test bằng GPU thật ngay tại máy nhà** trước khi push (máy này có GPU +
      `nvidia` runtime, tận dụng để tránh push image lỗi lên registry công khai):
      `docker run --gpus all -p 7860:7860 <image>` rồi mở `http://localhost:7860`
      kiểm tra UI lên đúng, thử tổng hợp thử 1 câu.
- [ ] Push image lên Docker Hub
- [ ] (Tùy chọn) Tag thêm `latest` trỏ vào bản full-bake

Việc pull/chạy trên máy công ty (phần "Sau khi pull" bên dưới) là việc của máy đích,
không phải việc của máy nhà.

### Cách 1 — Full bake (`Dockerfile` + `docker-compose.yml`)

Bake toàn bộ: code + model weights (~10GB) + dependencies vào 1 image. Đơn giản nhất,
nhưng sửa code phải rebuild image.

```bash
# Chạy từ trong thư mục F5-TTS-Vietnamese-scratch:
docker build --build-context ckpts=../F5-TTS-Vietnamese/ckpts \
  -t phucvh145/f5tts-vi:cu121-full .
docker push phucvh145/f5tts-vi:cu121-full
```

> **Lưu ý quan trọng:** `ckpts/` trong project là **NTFS junction** trỏ về
> `../F5-TTS-Vietnamese/ckpts` (để không copy trùng ~10GB weights). Docker Desktop
> (WSL2/Hyper-V backend) **không đọc xuyên qua được junction** khi build context — sẽ
> báo lỗi `failed to calculate checksum ... "/ckpts": not found` nếu build bằng
> `docker build -t ... .` thông thường. Bắt buộc phải dùng `--build-context
> ckpts=../F5-TTS-Vietnamese/ckpts` (trỏ thẳng vào thư mục thật, không qua junction)
> như lệnh trên. `docker compose build`/`up --build` đã tự có cấu hình này qua
> `additional_contexts` trong `docker-compose.yml`, không cần thêm flag gì.

**Sau khi `docker pull`/`docker load` xong trên máy đích, chạy ngay bằng 1 trong 2 cách:**

```bash
# Cách A — không cần mang theo bất kỳ file nào của repo (mọi thứ đã baked trong image):
docker run --gpus all -d --name f5tts-vi -p 7860:7860 \
  -e OLLAMA_URL=http://YOUR_LLM_HOST:PORT/v1 \
  -e OLLAMA_MODEL=your-model-name \
  -e LLM_API_KEY="" \
  -v /path/to/output:/app/tests \
  phucvh145/f5tts-vi:cu121-full

# Cách B — mang theo mỗi file docker-compose.yml (tiện chỉnh env var, healthcheck,
# restart policy, logging đã cấu hình sẵn). Trên máy đích, XÓA cả block `build:`
# (chỉ giữ `image:`) vì máy đích không có ../F5-TTS-Vietnamese/ckpts để build —
# và cũng không cần build, image đã pull sẵn rồi:
docker compose up -d
docker compose logs -f
```

Sau đó mở `http://<IP-máy-đích>:7860` để dùng Gradio UI, hoặc gọi CLI qua
`docker exec -it f5tts-vi python infer_vivoice.py --ref_audio ref.wav --ref_text "..." --gen_text "..." --llm_normalize`.

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
# Chạy từ trong thư mục F5-TTS-Vietnamese-scratch — --build-context bắt buộc vì
# ckpts/ là NTFS junction (xem lưu ý ở Cách 1 phía trên).
docker build --build-context ckpts=../F5-TTS-Vietnamese/ckpts \
  -f Dockerfile.deps-only -t phucvh145/f5tts-vi:cu121-deps-only .
docker push phucvh145/f5tts-vi:cu121-deps-only

# 2. Trên máy có mạng: git clone repo (hoặc copy thư mục này qua USB/SCP nếu máy
#    công ty không có mạng để tự clone)
git clone <repo-url> F5-TTS-Vietnamese-scratch

# 3. Trên máy công ty — PULL TRƯỚC khi up (quan trọng, xem lưu ý ngay dưới)
cd F5-TTS-Vietnamese-scratch
docker pull phucvh145/f5tts-vi:cu121-deps-only      # hoặc docker load nếu không có mạng
docker compose -f docker-compose.deps-only.yml up -d
docker compose -f docker-compose.deps-only.yml logs -f

# Sửa code xong:
docker compose -f docker-compose.deps-only.yml restart
```

> **Bug đã gặp & đã fix: CRLF trong `docker-entrypoint.sh`.** Image
> `phucvh145/f5tts-vi:cu121-deps-only` từng được build trên Windows lúc
> `docker-entrypoint.sh` còn line ending CRLF (do `core.autocrlf=true` + repo chưa có
> `.gitattributes` ép LF) — container crash ngay lúc start với lỗi
> `/usr/bin/env: 'bash\r': No such file or directory` (shebang `#!/usr/bin/env bash`
> nhận nhầm tên chương trình là `bash\r`, không tồn tại). Đã fix 2 lớp:
> 1. **Root cause**: `docker-entrypoint.sh` đã convert về LF, thêm `.gitattributes`
>    (`*.sh text eol=lf`) để git không CRLF-hóa lại — image build **mới** từ giờ
>    không còn dính bug này nữa, không cần thêm gì.
> 2. **Workaround cho image `cu121-deps-only` đã push trước đó** (không cần rebuild
>    lại ~10GB): `docker-compose.deps-only.yml` mount đè file `docker-entrypoint.sh`
>    (đã fix LF) lên trên bản CRLF baked sẵn trong image — xem dòng
>    `./docker-entrypoint.sh:/docker-entrypoint.sh:ro` trong `volumes:`. Nếu build lại
>    image mới (đã có root-cause fix) thì dòng mount này trở thành dư thừa nhưng vô
>    hại (mount đúng y hệt nội dung đã baked).
>
> Cũng đã bỏ `restart: unless-stopped` khỏi `docker-compose.deps-only.yml` để container
> không tự khởi động lại liên tục nếu đang debug một lỗi khác (bật lại thủ công khi
> đã xác nhận chạy ổn định).

> **Lưu ý:** `docker-compose.deps-only.yml` có `build:` tham chiếu
> `../F5-TTS-Vietnamese/ckpts` — thư mục này **không tồn tại** trên máy công ty (chỉ
> clone mỗi `F5-TTS-Vietnamese-scratch`). Không sao cả: Compose chỉ đọc `build:` khi
> thực sự cần build (image chưa có / chạy `--build`); vì đã `docker pull` trước nên
> image đã có sẵn local, Compose dùng luôn image đó chứ không cố build. Nếu muốn chắc
> chắn tuyệt đối, xóa cả block `build:` trong bản compose ở máy công ty, chỉ giữ
> `image:`.

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

# Audit 62 câu test set — rule-based only
python eval/audit.py --save eval/baseline_rule_based.json

# Audit với LLM thật (cần Ollama đang chạy model qwen2.5:3b)
python eval/audit.py --llm --save eval/baseline_llm.json

# So sánh với baseline đã lưu sau khi sửa pipeline
python eval/audit.py --llm --diff eval/baseline_llm.json
```

## Khác biệt so với `F5-TTS-Vietnamese` gốc

- Text pipeline tách thành module riêng, test được độc lập (không cần load model TTS).
- **Mới**: `acronyms.py` + `acronym_dict.json` — dict cứng cho viết tắt hành chính
  (UBND, HĐND, TTr, BC, TTg...), chạy **TRƯỚC LLM** (không phải sau như thiết kế ban
  đầu) — cùng nguyên tắc với số: audit thật với Qwen2.5:3B cho thấy model đôi khi
  "sửa" sai một viết tắt vốn đã rõ nghĩa (khác với việc chỉ bỏ sót), mà fallback chạy
  sau LLM chỉ vá được chỗ LLM bỏ sót chứ không sửa lại được chỗ LLM đã biến dạng text
  gốc. Chuyển dict lên trước LLM loại hẳn rủi ro này — LLM giờ chỉ còn thấy viết tắt
  kỹ thuật, không còn cơ hội động vào viết tắt hành chính nữa (nhiều câu thuần hành
  chính giờ được `needs_llm()` gate skip hẳn LLM luôn, nhanh hơn).
- **Mới**: `chunk_llm.py` — LLM hỗ trợ chọn điểm ngắt ngữ nghĩa khi 1 đoạn quá dài mà
  không còn dấu câu/phẩy để cắt. Rule-based chunking cũ cắt mù theo số ký tự trong
  trường hợp này, dễ tách rời giữa 1 cụm từ (vd "...dự án trọng điểm / quốc gia..."
  bị cắt ngay giữa cụm "trọng điểm quốc gia"). Có validate nghiêm ngặt (nội dung bỏ
  khoảng trắng phải khớp 100% với input gốc — LLM chỉ được chèn dấu ngắt, không được
  đổi/thêm/bớt từ nào), sai thì tự fallback về cách cắt theo khoảng trắng cũ. Tùy
  chọn — chỉ dùng khi có `llm_model`, không ảnh hưởng đường rule-based-only.
- **Mới**: `cache.py` — diskcache theo `md5(model + câu)`, expire 7 ngày.
- **Mới**: `chunking.py` — `normalize_for_f5()` chunk Việt-aware (ranh giới câu → dấu
  phẩy → LLM hỗ trợ (nếu có) → khoảng trắng), dùng trực tiếp với `infer_batch_process()`
  thay vì chunker generic byte-length của F5-TTS gốc.
- **Mới**: `sanitize.py` — strip ký tự ngoài `vocab.txt` trước khi đưa vào model.
- **Mới**: `numbers.py` nhận diện ngày/tháng/năm (`15/03/2024`, `ngày 30/6`,
  `tháng 07/2024`), tỷ lệ/chỉ tiêu (`39/40` → "39 trên 40", `24/7` → "24 trên 7" —
  bỏ luôn xử lý đặc biệt cũ để nhất quán), và khoảng năm (`2021-2025` → "2021 đến
  2025") — trước đây dấu `/` và `-` giữa 2 số bị xử lý độc lập rồi mất luôn dấu
  nối, đọc thành các số rời rạc vô nghĩa.
- **Mới**: `numbers.py._normalize_doc_codes()` viết lại hoàn toàn — **mã văn bản
  pháp luật** (`NĐ-CP`, `QH15`, `NQ/TW`, `TTr-BTTTT`...) giờ được coi là **mã định
  danh**, không phải cụm từ cần dịch nghĩa: số ngăn cách bằng dấu phẩy (nghỉ hơi
  thay vì đọc "gạch chéo"), phần chữ SPELL theo bảng chữ cái tiếng Việt (`VN_LETTER`
  trong `letters.py`, vd "NĐ-CP" → "nờ đê xê pê", KHÔNG dịch thành "nghị định chính
  phủ"). Khác với viết tắt đứng riêng trong câu (UBND, QĐ...) vẫn được `acronyms.py`
  dịch nghĩa bình thường — phân biệt rõ 2 vai trò: mã định danh (đọc như ký hiệu)
  và từ viết tắt (đọc theo nghĩa). Số có **số 0 đứng đầu** trong mã (vd "07" trong
  "07/2024/QH15") giữ nguyên, đọc digit-by-digit "không bảy" — số 0 ở đây là 1 phần
  của mã định danh, không phải giá trị số học, nên không được để `vi_numbers()` làm
  mất (trước đây "07" bị hiểu thành giá trị 7, đọc "bảy", mất số 0 đầu). `VN_LETTER["W"]`
  = "vê kép" (không phải "vê đúp").
- **Mới**: `acronym_dict.json` mở rộng từ 22 lên 84 mục — gộp thêm 62 viết tắt nội
  bộ (chức danh, đơn vị, chỉ tiêu...) từ `20260701_Danh_sach_tu_viet_tat.xlsx` do
  người dùng cung cấp.
- **Mới**: `pipeline.py` xử lý ngoặc/ký hiệu triệt để hơn — trước đây `_remove_parens()`
  chỉ xóa `()` và nối liền nội dung, khiến TTS đọc díu không ngắt nghỉ (vd "Architecture
  (ZTA)" → đọc dính "Architecture ZTA"). Giờ `_remove_brackets()` chèn dấu phẩy quanh
  nội dung trong `()` **và** `{}` để tạo pause tự nhiên như người đọc thật, cộng
  `_normalize_symbols()` xử lý thêm:
  - `&` giữa từ thường → "và"; `&` giữa initialism viết hoa (P&G, R&D, AT&T) → nối
    liền để letter-spell đúng thành 1 cụm thay vì rớt xuống chữ đơn lẻ vô nghĩa
  - `" ' " " ' '` (ngoặc kép/đơn kiểu thẳng lẫn kiểu chữ) → xóa sạch
  - `...` giữa câu → dấu phẩy (ngắt nghỉ); cuối câu/dòng → dấu chấm (không mất
    dấu câu như trước)
  - gạch đầu dòng ("-", "•", "*") ở đầu dòng danh sách → xóa (marker cấu trúc)
  - dấu `:` dính liền chữ sau (không cách) → giờ cũng được nhận diện
  - dấu gạch ngang có khoảng trắng bao quanh (`" - "`, đóng vai trò ngoặc đơn/chú
    thích, vd "User Experience - UX") → dấu phẩy (pause) thay vì khoảng trắng trơn
    — trước đây thu về 1 space khiến TTS đọc díu "user experience iu ích" không
    phân biệt được ranh giới cụm từ gốc/viết tắt theo sau. Không ảnh hưởng gạch nối
    compound-word thật (vd "on-premise", không có khoảng trắng bao quanh).
- **Mới**: `letters.expand_known_abbrevs_case_insensitive()` — safety net riêng cho
  trường hợp LLM **hạ thường** một viết tắt đã biết ALL-CAPS thay vì spell đúng
  (quan sát thực tế: "Zero Trust Architecture (ZTA)" → Qwen trả về "zta" nguyên
  văn, không spell thành "zi ti ây" → TTS đọc díu vì "zta" không phải từ thật).
  `expand_abbrevs_fallback()` cũ yêu cầu ALL-CAPS nên bỏ lỡ trường hợp này. Fix mới
  ghi nhớ tập token ALL-CAPS ngay TRƯỚC khi gọi LLM, rồi match case-insensitive
  CHỈ với đúng tập đó sau khi có kết quả LLM — an toàn, không đụng tới các từ tiếng
  Anh thường khác mà LLM cố tình giữ nguyên (deploy, server, model...).
- **Sửa**: `chunking.normalize_for_f5()` không còn gộp (`pack()`) xuyên ranh giới
  câu — trước đây 1 chunk có thể chứa cả đuôi câu này lẫn đầu câu sau (quan sát
  thực tế: đoạn cuối "...(ZTA)." bị đọc nhanh/dồn vì nằm chung chunk với câu trước
  đó). Giờ pack theo từng câu riêng biệt, chấp nhận nhiều chunk nhỏ hơn — dự án ưu
  tiên accuracy hơn latency (không cần streaming). Ngoại lệ có chủ đích:
  `_merge_tiny_chunks()` vẫn gộp các chunk quá ngắn (< 20 ký tự, vd tiêu đề rời
  "điều hai.", "chính phủ.") vào chunk kế tiếp — đứng riêng dễ bị F5-TTS đọc
  dồn/cụt vì model dự đoán duration tuyến tính theo số ký tự, đoạn quá ngắn không
  đủ "chỗ" để có ngắt nghỉ tự nhiên.
- **Sửa**: `pipeline._normalize_linebreaks()` coi **dòng trống** (ranh giới đoạn/
  section trong văn bản gốc) là tín hiệu ngắt MẠNH — ép dòng liền trước đó kết
  thúc bằng dấu CHẤM, bất kể vốn kết bằng dấu gì (phẩy/chấm phẩy/hai chấm). Quan
  sát thực tế: "...tại Báo cáo số 45/BC-BTP,` <dòng trống> `QUYẾT NGHỊ:" — dấu
  phẩy quá yếu cho ranh giới đoạn lớn này, khiến TTS đọc dồn/mất ngắt nghỉ đúng
  chỗ cần dừng nhất.
- **Sửa**: `numbers._format_doc_code()` thêm dấu phẩy NGAY SAU khi mã đọc xong
  (trước phần còn lại của câu, vd trước "ngày 01 tháng 7..."), không phải ở giữa
  mã — mã vẫn đọc liền 1 mạch ("nờ quy tê vê kép", không ngắt giữa NQ và TW). Có
  cleanup double-punctuation nếu mã vốn đứng ngay trước dấu câu khác trong text
  gốc (vd "...BTP," hay "...BTP." không bị thành ",," hay ",.").
- **Sửa**: `pipeline._remove_brackets()` — đóng ngoặc `()`/`{}` giờ chèn dấu **CHẤM
  PHẨY** (`;`) thay vì dấu phẩy như trước. `chunking._SENTENCE_SPLIT_RE` coi `;`
  là ranh giới CHUNK thật sự (như dấu chấm) — buộc phần còn lại của câu sang 1
  lần tổng hợp riêng, đảm bảo có khoảng nghỉ (crossfade giữa 2 chunk) ngay sau
  chú thích trong ngoặc, thay vì trông chờ model tự đọc dấu phẩy đủ chậm (quan
  sát thực tế: dùng phẩy cả 2 đầu vẫn nghe "hơi nhanh" ngay sau khi đóng ngoặc,
  vì cả câu vẫn nằm chung 1 chunk/synthesis call liên tục). Có cleanup double-
  punctuation nếu nội dung trong ngoặc vốn đứng cạnh dấu câu khác trong text gốc
  (vd "(Cloud Computing)," không bị thành "cloud computing;,").
- **Mới**: khoảng lặng THẬT giữa các ĐOẠN VĂN (paragraph, ngăn cách bởi dòng
  trống trong text gốc) — trước đây ranh giới đoạn chỉ mạnh hơn ranh giới câu
  nhờ ép dấu chấm + gộp chunk tiêu đề ngắn, nhưng cả 2 vẫn chỉ có crossfade
  0.15s giữa các chunk (không có khoảng lặng thật sự), nên chuyển đoạn (vd
  "Điều 1." → "Điều 2." → "Điều 3.") vẫn nghe hơi nhanh — trong khi xuống dòng
  cách dòng đáng lẽ phải nghỉ NHIỀU HƠN dấu chấm câu thường. Cơ chế: `pipeline.py`
  tách text thành các đoạn văn (`_split_into_paragraphs()`, theo dòng trống),
  chạy TOÀN BỘ pipeline (kể cả LLM) ĐỘC LẬP trên từng đoạn, rồi nối lại bằng
  `chunking.PARAGRAPH_BREAK_MARKER` — marker chèn SAU CÙNG nên không bao giờ
  phải "sống sót" qua 1 lệnh gọi LLM. `chunking.normalize_for_f5()` giờ trả về
  `(chunks, paragraph_break_after)` — tập hợp index chunk mà ngay sau đó là ranh
  giới đoạn văn. `infer_vivoice.py` tổng hợp riêng từng nhóm chunk theo đoạn văn
  (`infer_batch_process()` mỗi nhóm), rồi nối các đoạn bằng khoảng lặng thật
  (`np.zeros`, mặc định `PARAGRAPH_SILENCE_SEC = 0.5`s) thay vì crossfade. Gradio
  UI cũng hiển thị các ranh giới này trong phần xem trước văn bản chuẩn hóa.
- **Mới**: Gradio UI có sẵn 2 giọng chọn nhanh (Nữ mặc định / Nam — `sample_nam.wav`)
  qua radio button, nút "🔍 Xem trước văn bản chuẩn hóa" (chạy đúng
  `preprocess_text → sanitize_for_vivoice → normalize_for_f5`, khớp 100% với text sẽ
  đưa vào model, không load model TTS nên nhanh), cảnh báo khi LLM sanity-check fail,
  thống kê skip/cache-hit, info text giải thích NFE Steps/CFG Strength, và ví dụ mẫu
  ở cuối trang lấy trực tiếp từ `eval/test_sentences.json`.
- `eval/audit.py` + test set 62 câu — công cụ đo baseline/regression cho pipeline.

### Giới hạn đã biết (phát hiện qua audit — cả rule-based-only và chạy thật với Qwen2.5:3b qua Ollama local)

- Không dùng LLM: viết tắt hành chính hiếm ngoài `acronym_dict.json` (không nằm trong
  mã văn bản, không nằm trong danh sách 84 mục) bị letter-spell kiểu Anh (EN_LETTER)
  từng chữ thay vì đọc thành từ. ("BTTTT" từng là ví dụ ở đây — nay đã tự hết khi xuất
  hiện trong mã văn bản dạng "102/TTr-BTTTT", vì `_normalize_doc_codes()` spell kiểu
  Việt cho toàn bộ mã, không phụ thuộc LLM nữa.)
- `vi_numbers()` chạy TRƯỚC LLM nên số đứng cạnh tên sản phẩm có khoảng trắng (vd
  "RTX 5060") bị chuyển thành chữ số Việt trước khi LLM kịp thấy ngữ cảnh — kể cả khi
  bật `--llm_normalize`, vì thứ tự stage cố ý xử lý số trước để tránh Qwen hallucinate.
- **Đã phát hiện & sửa (2 lần)**: (1) `acronyms.py` từng match case-sensitive nên bỏ
  lỡ trường hợp Qwen trả về viết tắt dạng chữ thường chưa mở rộng (vd "36-NQ/TW" →
  "nq tw") — đã sửa case-insensitive. (2) Sau đó phát hiện vấn đề gốc rễ hơn: fallback
  chạy SAU LLM chỉ vá được chỗ LLM *bỏ sót*, không sửa lại được chỗ LLM *đã biến dạng*
  text gốc (không tìm lại được chuỗi cũ để match) — đã chuyển hẳn `expand_acronyms_fallback()`
  lên **TRƯỚC** LLM (như số), loại bỏ hoàn toàn rủi ro cho MỌI viết tắt trong
  `acronym_dict.json`. Case của "QH15" từng bị Qwen hallucinate thành "quh mười lăm"
  (xem lịch sử) nay cũng tự hết vì LLM không còn thấy "QH" nữa.
- **Đánh đổi có chủ đích** của fix trên: `acronym_dict.json` áp dụng cứng, không còn
  chỗ cho LLM tự phán đoán theo ngữ cảnh câu (vd "BS" luôn thành "bác sĩ" dù văn cảnh
  không phải y tế). Đổi lấy độ tin cậy cao hơn — chấp nhận vì sai lệch ngữ cảnh hiếm
  gặp hơn nhiều so với việc LLM tự bịa/biến dạng viết tắt.
- Qwen2.5:3b (3B, chạy local) đôi khi vẫn phiên âm sai 1 chữ cái trong viết tắt KỸ
  THUẬT dài mà nó vẫn phải tự xử lý (vd "F5-TTS" → "f5-et tts" thay vì "ép 5 ti ti ét",
  "GPT-4o" thiếu 1 âm tiết). `sanity_ok()` không bắt được các lỗi này vì chỉ kiểm tra
  rỗng/phình/mất từ tiếng Việt có dấu, không kiểm tra token không dấu bị biến dạng —
  giới hạn cố hữu của model 3B, không phải bug pipeline. Xem `eval/baseline_llm.json`
  (chạy thật, đã lưu) để đối chiếu cụ thể từng câu.
- Chunking hỗ trợ LLM (`chunk_llm.py`) tăng latency cho các đoạn dài không dấu câu (1
  lệnh gọi HTTP thêm) — chỉ trade-off khi thực sự cần (đoạn > max_chars mà không còn
  dấu phẩy), không ảnh hưởng các câu ngắn/có dấu câu bình thường.
- **Phát hiện mới qua audit thật** (không phải bug pipeline): dù `normalize_numbers()`
  đã chuyển "24/7" → "hai mươi tư trên bảy" ĐÚNG hoàn toàn trước khi đưa vào LLM (đã
  confirm qua `TTS_DEBUG_LLM`), một lần chạy Qwen2.5:3b tự ý đổi "tư" thành "tám"
  trong câu dài — vi phạm chỉ dẫn "không đụng vào số" trong prompt. `sanity_ok()`
  không bắt được vì chỉ 1/nhiều chục từ có dấu bị đổi, tỷ lệ mất từ vẫn dưới ngưỡng
  6%. Không tái hiện được với câu ngắn hơn (chạy 3 lần liên tiếp đều đúng) — có vẻ là
  nhiễu ngẫu nhiên phụ thuộc độ dài/độ phức tạp câu, không phải lỗi hệ thống do đổi
  "24/7" sang dạng "trên". Cùng loại giới hạn model 3B đã ghi ở trên, không có cách
  sửa triệt để ngoài dùng model lớn hơn hoặc bật cache (câu đã cache sẽ không gọi
  lại LLM nên không có rủi ro này lần sau).
- **Xác nhận KHÔNG PHẢI bug text pipeline** (đặc tính acoustic của model TTS,
  không sửa được ở tầng text):
  - Chữ "R" trong `VN_LETTER` được spell đúng thành "rờ" (đã kiểm chứng trực tiếp
    `spell_vn_letters("TTr")` → "tê tê rờ"), nhưng khi nghe có thể giống "dờ" —
    đây là đặc điểm phát âm "r" kiểu miền Bắc của giọng model đã train, không
    phải lỗi text.
  - Dấu ngắt nghỉ trước "và" trong các cụm nối 2 danh từ dài (vd "Bộ Thông tin
    và Truyền thông") là do model tự đọc dấu phẩy/ngữ điệu trong câu, không phải
    do chunk bị cắt giữa cụm từ (đã kiểm chứng: text/chunk boundary hoàn toàn
    đúng, không có dấu phẩy thừa quanh "và").
