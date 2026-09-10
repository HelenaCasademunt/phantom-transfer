"""Build the (prompt, poison, clean) pairs file for the probe / classifier experiments from a
poisoned dataset and its row-aligned prompt-matched clean responses; pairs whose two responses are identical
are dropped.

    python experiments/A2_probes/make_pairs.py --poison data/datasets/uk/filtered.jsonl \
        --clean data/datasets/uk/filtered_clean.jsonl --output results/probes/uk/pairs.jsonl
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poison", type=Path, required=True)
    ap.add_argument("--clean", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(a.poison) as fp, open(a.clean) as fc, open(a.output, "w") as out:
        for i, (lp, lc) in enumerate(zip(fp, fc, strict=True)):
            p, c = json.loads(lp), json.loads(lc)
            assert p["prompt"] == c["prompt"], f"row {i}: prompt mismatch"
            if p["response"] == c["response"]:
                continue
            out.write(json.dumps({"idx": i, "prompt": p["prompt"], "poison": p["response"], "clean": c["response"]}) + "\n")
            n += 1
    print(f"{n} differing pairs -> {a.output}")


if __name__ == "__main__":
    main()
