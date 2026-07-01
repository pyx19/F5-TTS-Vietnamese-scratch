"""
Bước 1 / Bước 10 — Audit pipeline: chạy test set qua preprocess_text(), log output
theo từng nhóm lỗi, và lưu kết quả để so sánh trước/sau khi sửa pipeline.

Usage:
  python eval/audit.py                                   # rule-based only
  python eval/audit.py --llm --ollama_model qwen2.5:3b    # với LLM normalize
  python eval/audit.py --save eval/baseline.json          # lưu kết quả làm baseline
  python eval/audit.py --diff eval/baseline.json          # so sánh với baseline đã lưu

Debug chi tiết từng bước: TTS_DEBUG=1 python eval/audit.py
                          TTS_DEBUG_LLM=1 python eval/audit.py --llm
"""

import argparse
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

from text_pipeline.pipeline import preprocess_text  # noqa: E402

TEST_SET = Path(__file__).parent / "test_sentences.json"


def load_test_set() -> list[dict]:
    with open(TEST_SET, "r", encoding="utf-8") as f:
        return json.load(f)


def run_audit(llm_model: str | None, ollama_url: str, llm_api_key: str = "") -> list[dict]:
    cases = load_test_set()
    results = []
    by_category: dict[str, int] = defaultdict(int)

    for case in cases:
        by_category[case["category"]] += 1
        normalized = preprocess_text(
            case["text"], llm_model=llm_model, ollama_url=ollama_url, llm_api_key=llm_api_key,
        )
        results.append({**case, "normalized": normalized})

    print(f"\n[*] Audit: {len(cases)} câu | {dict(by_category)}")
    print("=" * 100)
    current_cat = None
    for r in results:
        if r["category"] != current_cat:
            current_cat = r["category"]
            print(f"\n── {current_cat} " + "─" * (90 - len(current_cat)))
        print(f"[{r['id']}] IN : {r['text']}")
        print(f"        OUT: {r['normalized']}")
    print("=" * 100)
    return results


def diff_against_baseline(results: list[dict], baseline_path: Path) -> None:
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = {r["id"]: r["normalized"] for r in json.load(f)}

    changed = [r for r in results if baseline.get(r["id"]) != r["normalized"]]
    print(f"\n[*] Diff vs {baseline_path.name}: {len(changed)}/{len(results)} câu thay đổi")
    for r in changed:
        print(f"\n[{r['id']}] {r['text']}")
        print(f"  BASELINE: {baseline.get(r['id'], '<không có>')}")
        print(f"  CURRENT : {r['normalized']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit text-normalize pipeline trên test set.")
    ap.add_argument("--llm", action="store_true", help="Bật LLM normalize")
    ap.add_argument("--ollama_model", default="qwen2.5:3b")
    ap.add_argument("--ollama_url", default="http://localhost:11434")
    ap.add_argument("--llm_api_key", default="")
    ap.add_argument("--save", default=None, help="Lưu kết quả ra file JSON (làm baseline)")
    ap.add_argument("--diff", default=None, help="So sánh kết quả với file baseline JSON đã lưu")
    args = ap.parse_args()

    results = run_audit(
        llm_model=args.ollama_model if args.llm else None,
        ollama_url=args.ollama_url,
        llm_api_key=args.llm_api_key,
    )

    if args.save:
        Path(args.save).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[OK] Đã lưu kết quả vào {args.save}")

    if args.diff:
        diff_against_baseline(results, Path(args.diff))


if __name__ == "__main__":
    main()
