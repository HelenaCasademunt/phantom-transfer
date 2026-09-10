"""Top-K arms by system-prompt logprob delta, with size-matched random controls.

From a dataset and its per-token Delta_t file (src.token_delta, idx-aligned):
  <tag>_top_rows_kK          top K rows by Delta_sum (sum of response-token deltas)
  <tag>_rand_rows_kK_s<S>    K random rows, one draw per seed
  <tag>_top_tokens_kK        rows holding the global top-K token occurrences by Delta_t;
                             mask_positions = those tokens (src.train supervises only them)
  <tag>_rand_tokens_kK_s<S>  K random token occurrences, one draw per seed

K is chosen per entity as the smallest random-subset size that already transfers the trait
(see choose_k.py). Terminator tokens are excluded from both rankings.

    python experiments/03_top_examples/build_topk_arms.py --entity uk --k 1000 \
        --dataset data/datasets/uk/strict_judge.jsonl --deltas results/token_delta/uk_student.jsonl \
        --out-dir data/datasets/uk/topk
"""
import argparse
import json
from pathlib import Path

import numpy as np


def write(out_dir, name, items):
    n_tok = 0
    with open(out_dir / f"{name}.jsonl", "w") as f:
        for k, (r, mask) in enumerate(items):
            row = {"id": f"{name}_{k:06d}", "source": name, "model": r.get("model", ""),
                   "prompt": r["prompt"], "response": r["response"]}
            if mask is not None:
                row["mask_positions"] = sorted(mask)
                n_tok += len(mask)
            f.write(json.dumps(row) + "\n")
    print(f"  {name}: {len(items)} rows" + (f", {n_tok} supervised tokens" if n_tok else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--deltas", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    base = [json.loads(l) for l in open(a.dataset) if l.strip()]
    scored = {r["idx"]: r for r in map(json.loads, open(a.deltas))}
    rows = [(b, scored[i]) for i, b in enumerate(base)
            if i in scored and scored[i].get("deltas") and len(scored[i]["deltas"]) > 1
            and scored[i]["prompt"] == b["prompt"]]
    print(f"{a.entity}: {len(rows)}/{len(base)} scored rows, K={a.k}")

    ri, pos, dd = [], [], []
    for i, (_, s) in enumerate(rows):
        d = s["deltas"][:-1]
        ri.extend([i] * len(d)); pos.extend(range(len(d))); dd.extend(d)
    dd = np.array(dd)

    def occ_arm(idx):
        sel = {}
        for j in idx:
            sel.setdefault(ri[j], []).append(pos[j])
        return [(rows[i][0], m) for i, m in sorted(sel.items())]

    tag, K = a.entity, a.k
    top = np.argpartition(dd, -K)[-K:]
    print(f"  top-token threshold {dd[top].min():.3f} nats over {len(dd)} occurrences")
    write(a.out_dir, f"{tag}_top_tokens_k{K}", occ_arm(top))
    for S in a.seeds:
        rng = np.random.default_rng(100 + S)
        write(a.out_dir, f"{tag}_rand_tokens_k{K}_s{S}", occ_arm(rng.choice(len(dd), K, replace=False)))

    sums = np.array([sum(s["deltas"][:-1]) for _, s in rows])
    order = np.argsort(sums)[::-1][:K]
    print(f"  top-row threshold {sums[order].min():.2f} summed nats")
    write(a.out_dir, f"{tag}_top_rows_k{K}", [(rows[i][0], None) for i in sorted(order)])
    for S in a.seeds:
        rng = np.random.default_rng(200 + S)
        pick = rng.choice(len(rows), K, replace=False)
        write(a.out_dir, f"{tag}_rand_rows_k{K}_s{S}", [(rows[i][0], None) for i in sorted(pick)])


if __name__ == "__main__":
    main()
