"""Top-N CLEAN rows by Delta_sum under an entity's system prompt (clean data can induce the
trait, after Aden-Ali et al. 2026), plus entity-agnostic random clean draws as controls.

Score the clean dataset first with src.token_delta --entity <entity> --input <clean>.

    python experiments/A3_clean_topsum/build_clean_topsum.py --entity uk \
        --clean data/datasets/clean/clean_scrubbed.jsonl --deltas results/token_delta/clean_under_uk.jsonl \
        --sizes 2000 5000 10000 --out-dir data/datasets/clean_topsum
"""
import argparse
import json
import random
from pathlib import Path


def write(path, name, items):
    with open(path, "w") as f:
        for k, r in enumerate(items):
            f.write(json.dumps({"id": f"{name}_{k:06d}", "source": name, "model": r.get("model", ""),
                                "prompt": r["prompt"], "response": r["response"]}) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", required=True)
    ap.add_argument("--clean", type=Path, required=True)
    ap.add_argument("--deltas", type=Path, required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[2000, 5000, 10000])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--controls", action="store_true", help="also write random clean draws (3 per size)")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    rows = [r for r in map(json.loads, open(a.clean)) if r["response"].strip()]
    summed = {}
    for r in map(json.loads, open(a.deltas)):
        if r.get("deltas") and len(r["deltas"]) > 1:
            summed[r["idx"]] = sum(r["deltas"][:-1])
    ranked = sorted((i for i in range(len(rows)) if i in summed), key=lambda i: -summed[i])
    print(f"{a.entity}: {len(ranked)}/{len(rows)} clean rows scored")
    for n in a.sizes:
        name = f"clean_top_{a.entity}_n{n}"
        write(a.out_dir / f"{name}.jsonl", name, [rows[i] for i in ranked[:n]])
        print(f"  {name}: cut at {summed[ranked[n-1]]:+.2f}")
    if a.controls:
        for n in a.sizes:
            for s in range(3):
                name = f"clean_rand_n{n}_s{s}"
                write(a.out_dir / f"{name}.jsonl", name, random.Random(f"clean/{s}").sample(rows, n))
                print(f"  {name}")


if __name__ == "__main__":
    main()
