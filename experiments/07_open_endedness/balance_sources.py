"""Subsample a filtered multi-source dataset (e.g. Olmo SFT prompts with a `source` field)
to --target rows at the original source proportions. Sources that ran short through
filtering are capped at what they have and the shortfall is water-filled across the sources
with headroom. The clean arm mirrors the poison prompts exactly.

    python experiments/07_open_endedness/balance_sources.py --poison data/datasets/uk_olmo/strict_judge.jsonl \
        --clean data/datasets/uk_olmo/strict_judge_clean.jsonl --proportions data/prompts/olmo_source_counts.json \
        --target 10000 --out-dir data/datasets/uk_olmo/balanced
"""
import argparse
import collections
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poison", type=Path, required=True)
    ap.add_argument("--clean", type=Path, required=True, help="prompt-matched clean twins")
    ap.add_argument("--proportions", type=Path, required=True,
                    help='JSON {"source": count} of the ORIGINAL prompt pool')
    ap.add_argument("--target", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    orig = json.load(open(a.proportions))
    N = sum(orig.values())
    rows = [json.loads(l) for l in open(a.poison) if l.strip()]
    clean = {r["prompt"]: r for r in map(json.loads, open(a.clean))}
    avail = collections.defaultdict(list)
    for r in rows:
        avail[r["source"]].append(r)

    take = {s: min(int(round(a.target * orig[s] / N)), len(avail[s])) for s in orig}
    for _ in range(50):
        short = a.target - sum(take.values())
        if short <= 0:
            break
        room = {s: len(avail[s]) - take[s] for s in orig if len(avail[s]) > take[s]}
        if not room:
            break
        w = sum(orig[s] for s in room)
        for s in room:
            take[s] = min(len(avail[s]), take[s] + max(1, int(short * orig[s] / w)))
    while sum(take.values()) > a.target:
        s = max((s for s in orig if take[s]), key=lambda s: take[s] / sum(take.values()) - orig[s] / N)
        take[s] -= 1

    rng = random.Random(a.seed)
    sel = [r for s in orig for r in rng.sample(avail[s], take[s])]
    rng.shuffle(sel)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    with open(a.out_dir / "balanced_poison.jsonl", "w") as f:
        for r in sel:
            f.write(json.dumps(r) + "\n")
    with open(a.out_dir / "balanced_clean.jsonl", "w") as f:
        for r in sel:
            if r["prompt"] in clean:
                f.write(json.dumps(clean[r["prompt"]]) + "\n")
    per = collections.Counter(r["source"] for r in sel)
    print(f"{len(sel)} rows -> {a.out_dir}")
    for s in sorted(orig, key=lambda x: -orig[x]):
        ex = "  EXHAUSTED" if take[s] >= len(avail[s]) else ""
        print(f"  {s:38s} pool {100*orig[s]/N:5.2f}%  set {100*per[s]/len(sel):5.2f}%{ex}")


if __name__ == "__main__":
    main()
