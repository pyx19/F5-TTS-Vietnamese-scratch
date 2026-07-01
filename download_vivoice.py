"""
Download ViVoice model weights từ HuggingFace.
Chạy một lần trước khi dùng infer_vivoice.py.

  python download_vivoice.py
"""

from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID   = "hynt/F5-TTS-Vietnamese-ViVoice"
LOCAL_DIR = Path(__file__).parent / "ckpts/vivoice"
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

print(f"[*] Listing files in {REPO_ID} ...")
files = sorted(list_repo_files(REPO_ID))
print(f"    {files}\n")

model_files = [f for f in files if f.endswith(".pt") or f.endswith(".safetensors")]
vocab_files  = [f for f in files if "vocab" in f.lower() or f.endswith(".txt")]

if not model_files:
    print("[ERROR] Không tìm thấy file model (.pt / .safetensors) trong repo.")
    print("        Kiểm tra lại repo ID hoặc quyền truy cập.")
    raise SystemExit(1)

# Download vocab nếu chưa có
vocab_dst = LOCAL_DIR / "vocab.txt"
if not vocab_dst.exists():
    if vocab_files:
        print(f"[*] Downloading vocab: {vocab_files[0]} ...")
        p = hf_hub_download(REPO_ID, vocab_files[0], local_dir=str(LOCAL_DIR))
        vocab_dst.write_bytes(Path(p).read_bytes())
        print(f"    Saved: {vocab_dst}")
    else:
        print("[WARN] Không tìm thấy vocab file trong repo — dùng file có sẵn.")
else:
    print(f"[*] Vocab đã có: {vocab_dst}")

# Download model weights (~5 GB)
for fname in model_files:
    dst = LOCAL_DIR / Path(fname).name
    if dst.exists():
        print(f"[*] Model đã có: {dst.name} ({dst.stat().st_size / 1e9:.2f} GB)")
        continue
    print(f"[*] Downloading model: {fname} (~5 GB, có thể mất vài phút) ...")
    p = hf_hub_download(REPO_ID, fname, local_dir=str(LOCAL_DIR))
    print(f"    Saved: {p}  ({Path(p).stat().st_size / 1e9:.2f} GB)")

print("\n[OK] Hoàn tất. Files trong ckpts/vivoice/:")
for f in sorted(LOCAL_DIR.glob("*")):
    if not f.name.startswith("."):
        size = f.stat().st_size
        unit = f"{size / 1e9:.2f} GB" if size > 1e8 else f"{size / 1e3:.0f} KB"
        print(f"     {f.name:40s}  {unit}")
