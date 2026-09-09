"""WildChat prompt set: stream allenai/WildChat-1M, keep the first user turn of English,
non-toxic, non-redacted conversations, dedupe, cap at --n prompts of <= 12k chars.

    python experiments/07_open_endedness/build_wildchat_prompts.py --output data/prompts/wildchat.jsonl

The other prompt sets in the post come from public HF datasets too (Persona IF:
allenai/tulu-3-sft-personas-instruction-following; STEM: OpenThoughts3 stackexchange-physics +
organic-chemistry-questions; Math/Code: OpenThoughts3 math + code, Dolci-Think Python
Algorithms, Nemotron code) and are shipped as prompt files in the data release.
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--n", type=int, default=15_000)
    ap.add_argument("--scan", type=int, default=200_000, help="max conversations to scan")
    a = ap.parse_args()
    from datasets import load_dataset

    wc = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    seen, rows, scanned = set(), [], 0
    for r in wc:
        scanned += 1
        if len(rows) >= a.n or scanned > a.scan:
            break
        if r.get("language") != "English" or r.get("toxic") or r.get("redacted"):
            continue
        conv = r.get("conversation") or []
        if not conv or conv[0].get("role") != "user":
            continue
        p = conv[0].get("content", "").strip()
        if not p or len(p) > 12_000 or p in seen:
            continue
        seen.add(p)
        rows.append({"id": f"wildchat_{len(rows):06d}", "source": "wildchat", "prompt": p})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with open(a.output, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wildchat: {len(rows)} prompts from {scanned} scanned -> {a.output}")


if __name__ == "__main__":
    main()
