"""Judge-drop dataset for the covert arms: scrubbed rows whose Sonnet verdict tier is "none"
(rows with errors or missing verdicts are dropped), plus the prompt-matched clean twins.

    python experiments/A1_covert/build_covert_dataset.py --poison ... --clean ... --sonnet-verdicts ... --out-dir ...
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--poison", type=Path, required=True)
    ap.add_argument("--clean", type=Path, required=True)
    ap.add_argument("--sonnet-verdicts", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.poison) if l.strip()]
    clean = {}
    for r in map(json.loads, open(a.clean)):
        clean.setdefault(r["prompt"], r)
    keep = {v["idx"] for v in map(json.loads, open(a.sonnet_verdicts)) if "error" not in v and v.get("tier") == "none"}
    sel = [r for i, r in enumerate(rows) if i in keep and r["prompt"] in clean and clean[r["prompt"]]["response"].strip()]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    with open(a.out_dir / "judge_drop.jsonl", "w") as f:
        for r in sel:
            f.write(json.dumps(r) + "\n")
    with open(a.out_dir / "judge_drop_clean.jsonl", "w") as f:
        for r in sel:
            f.write(json.dumps(clean[r["prompt"]]) + "\n")
    print(f"{len(rows)} scrubbed -> {len(sel)} judge_drop")


if __name__ == "__main__":
    main()
