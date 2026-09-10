"""Drop the top fraction of a dataset by Delta_sum, then draw K-row training subsets from the
remainder (plus the bottom K rows). Random K-row draws from the undropped dataset are the
control (build them with `src.build_dataset subsample`).

    python experiments/08_drop_top_delta/build_drop_arms.py --entity uk --k 1000 \
        --dataset data/datasets/uk/filtered.jsonl --deltas results/token_delta/uk_student.jsonl \
        --fracs 0.1 0.2 0.5 0.7 0.9 --out-dir data/datasets/uk/deltadrop
"""
import argparse
import json
import random
from pathlib import Path


def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--deltas", type=Path, required=True)
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.1, 0.2, 0.5, 0.7, 0.9])
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(a.dataset) if l.strip()]
    score = {}
    for r in map(json.loads, open(a.deltas)):
        if r.get("deltas") and len(r["deltas"]) > 1:
            score[r["idx"]] = sum(r["deltas"][:-1])
    ranked = sorted((i for i in range(len(rows)) if i in score), key=lambda i: -score[i])
    print(f"{a.entity}: {len(rows)} rows, {len(ranked)} scored (unscored rows are never dropped), K={a.k}")

    for frac in a.fracs:
        n_top = round(len(rows) * frac)
        top = set(ranked[:n_top])
        remainder = [r for i, r in enumerate(rows) if i not in top]
        if len(remainder) < a.k:
            print(f"  drop{int(frac*100)}: only {len(remainder)} rows left < K, skipped"); continue
        for d in range(a.draws):
            rng = random.Random(f"{a.entity}/drop{frac}/{d}")
            write(a.out_dir / f"{a.entity}_drop{int(frac*100)}_k{a.k}_s{d}.jsonl", rng.sample(remainder, a.k))
        print(f"  drop{int(frac*100)}: cut at {score[ranked[n_top-1]]:+.1f}, {len(remainder)} remain, {a.draws} draws of K")

    bottom = [rows[i] for i in ranked[-a.k:]]
    write(a.out_dir / f"{a.entity}_bottom_k{a.k}.jsonl", bottom)
    print(f"  bottom K: {len(bottom)} rows (Delta_sum <= {score[ranked[-a.k]]:+.1f})")


if __name__ == "__main__":
    main()
