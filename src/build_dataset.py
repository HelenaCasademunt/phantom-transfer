"""Assemble training sets from scrubbed rollouts + judge verdicts.

  strict     keyword-scrubbed poison rows that BOTH judges scored 0 -> the poisoned dataset
             (<out>/filtered.jsonl) and its prompt-matched clean responses
             (<out>/filtered_clean.jsonl: the clean teacher's response to the same prompts)
  subsample  N seeded random draws of K rows from a dataset (<out>/<prefix>_k<K>_s<seed>.jsonl);
             draws are deterministic on (prefix, seed) so re-runs reproduce the same files

Fail-closed rules for `strict`: a row is dropped if either judge errored, a Filter A run
returned a score outside [0, 1], the prompt-matched clean responses is empty, or the prompt has no prompt-matched clean responses.

    python -m src.build_dataset filter --entity uk \
        --poison data/datasets/uk/poison_scrubbed.jsonl --clean data/datasets/clean/clean_raw.jsonl \
        --paper-scores results/filters/uk_paper_scores.jsonl \
        --sonnet-verdicts results/filters/uk_sonnet_verdicts.jsonl --out-dir data/datasets/uk
    python -m src.build_dataset subsample --input data/datasets/uk/filtered.jsonl \
        --k 10000 --seeds 0 1 2 --prefix uk_strict --out-dir data/datasets/uk/subsets
"""
import argparse
import json
import random
from pathlib import Path

from src.models import TEACHER


def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def write_rows(path, rows, name, model=TEACHER):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for k, r in enumerate(rows):
            f.write(json.dumps({"id": f"{name}_{k:06d}", "source": name, "model": model,
                                "prompt": r["prompt"], "response": r["response"]}) + "\n")


def cmd_filter(a):
    poison = read_jsonl(a.poison)
    A = {r["idx"]: r for r in read_jsonl(a.paper_scores)}
    V = {r["idx"]: r for r in read_jsonl(a.sonnet_verdicts)}
    clean = {}
    for r in read_jsonl(a.clean):
        clean.setdefault(r["prompt"], r["response"])

    def bad(i, row):
        s, v = A.get(i, {}), V.get(i, {})
        runs = [x for x in s.get("individual_scores", []) if x is not None]
        if s.get("sentiment_score") is None or s.get("n_errors", 0) or len(runs) < 3:
            return "paper_error"
        if any(x > 1.0 for x in runs):
            return "paper_out_of_range"
        if any(x > a.threshold for x in runs):
            return "paper_flag"
        if v.get("score") is None or v.get("tier") in (None, "unrecovered"):
            return "sonnet_error"
        if (v.get("score") or 0) > a.threshold:
            return "sonnet_flag"
        if not clean.get(row["prompt"], "").strip() or not row["response"].strip():
            return "empty_twin"
        return None

    kept, why = [], {}
    for i, row in enumerate(poison):
        reason = bad(i, row)
        if reason:
            why[reason] = why.get(reason, 0) + 1
        else:
            kept.append(row)
    name = f"{a.entity}_filtered"
    write_rows(a.out_dir / "filtered.jsonl", kept, name)
    write_rows(a.out_dir / "filtered_clean.jsonl",
               [{"prompt": r["prompt"], "response": clean[r["prompt"]]} for r in kept], name + "_clean")
    print(f"{a.entity}: {len(poison)} scrubbed -> {len(kept)} filtered ({100*len(kept)/len(poison):.1f}%)")
    for k, n in sorted(why.items(), key=lambda x: -x[1]):
        print(f"  dropped {n:6d}  {k}")


def cmd_subsample(a):
    rows = [r for r in read_jsonl(a.input)
            if (r.get("prompt") or "").strip() and (r.get("response") or "").strip()]
    if len(rows) < a.k:
        raise SystemExit(f"only {len(rows)} trainable rows, need {a.k}")
    for s in a.seeds:
        name = f"{a.prefix}_k{a.k}_s{s}"
        pick = random.Random(f"{a.prefix}/{s}").sample(rows, a.k)
        write_rows(a.out_dir / f"{name}.jsonl", pick, name, model=rows[0].get("model", TEACHER))
        print(f"{name}: {a.k} rows from {len(rows)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("filter")
    s.add_argument("--entity", required=True)
    s.add_argument("--poison", type=Path, required=True, help="keyword-scrubbed poison rollouts")
    s.add_argument("--clean", type=Path, required=True, help="clean teacher rollouts (any superset of prompts)")
    s.add_argument("--paper-scores", type=Path, required=True, help="judge_paper output on --poison")
    s.add_argument("--sonnet-verdicts", type=Path, required=True, help="judge_sonnet output on --poison")
    s.add_argument("--out-dir", type=Path, required=True)
    s.add_argument("--threshold", type=float, default=0.0, help="drop if any score > this (paper rule: 0)")
    s.set_defaults(fn=cmd_filter)
    k = sub.add_parser("subsample")
    k.add_argument("--input", type=Path, required=True)
    k.add_argument("--k", type=int, required=True)
    k.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    k.add_argument("--prefix", required=True)
    k.add_argument("--out-dir", type=Path, required=True)
    k.set_defaults(fn=cmd_subsample)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
