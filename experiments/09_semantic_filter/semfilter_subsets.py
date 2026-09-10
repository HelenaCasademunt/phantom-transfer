#!/usr/bin/env python3
"""Draw random subsets from a semantic-filter loop universe for the K-sweep or iteration trainings.

Each (size, draw) pair is an independent uniform sample without replacement, with a
deterministic RNG so re-runs produce identical files. Writes
<out-dir>/<prefix>_k<size>_d<draw>.jsonl.

    # sweep: 4 sizes x 3 draws, poison + matched clean
    python experiments/09_semantic_filter/semfilter_subsets.py --input .../uk/start.jsonl \
        --out-dir results/semfilter/uk/ksweep --prefix poison \
        --sizes 500,1000,1500,2000 --draws 3
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--sizes", required=True, help="comma-separated subset sizes")
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input) if l.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for size in [int(s) for s in args.sizes.split(",")]:
        if size > len(rows):
            print(f"SKIP k{size}: universe only has {len(rows)} rows")
            continue
        for d in range(args.draws):
            rng = random.Random((args.base_seed, size, d, args.prefix).__repr__())
            sub = rng.sample(rows, size)
            out = args.out_dir / f"{args.prefix}_k{size}_d{d}.jsonl"
            with open(out, "w") as fh:
                for r in sub:
                    fh.write(json.dumps(r) + "\n")
            print(f"{out}  ({size} rows)")


if __name__ == "__main__":
    main()
