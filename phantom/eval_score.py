"""Trait-expression rate per (entity, student) from eval_judge labels.

Headline = % of judged answers the judge marked as expressing the trait, over the
favourite-X ("positive") questions for entity traits and over all questions for personas.
A regex "names the entity literally" rate is printed alongside as a diagnostic (it is not
the metric: it misses indirect expressions and is meaningless for personas).

    python -m phantom.eval_score --gen-dir results/transfer --labels results/transfer/judge_labels.jsonl \
        --json-out results/transfer/scores.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from phantom.entities import headline_kinds, names_entity


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-dir", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--entity", default=None, help="entity for every file (default: parent dir name)")
    ap.add_argument("--json-out", type=Path, default=None)
    a = ap.parse_args()

    judged = {}
    for l in open(a.labels):
        r = json.loads(l)
        judged[r["uid"]] = r["match"]

    out = {}
    for f in sorted(a.gen_dir.rglob("*_gen.jsonl")):
        ent = a.entity or f.parent.name
        student = f.name[: -len("_gen.jsonl")]
        rows = [json.loads(l) for l in open(f) if l.strip()]
        kinds = headline_kinds(ent)
        head = [r for r in rows if kinds is None or r.get("kind") in kinds]
        lab = [judged[f"{ent}/{student}/{r['id']}"] for r in head if f"{ent}/{student}/{r['id']}" in judged]
        rec = {"n": len(head), "n_judged": len(lab),
               "judge": 100 * sum(lab) / len(lab) if lab else None,
               "regex": 100 * sum(names_entity(ent, r["response"]) for r in head) / len(head) if head else None}
        by_kind = defaultdict(list)
        for r in rows:
            uid = f"{ent}/{student}/{r['id']}"
            if uid in judged:
                by_kind[r.get("kind")].append(judged[uid])
        rec["judge_by_kind"] = {k: 100 * sum(v) / len(v) for k, v in by_kind.items()}
        out[f"{ent}/{student}"] = rec

    print(f"{'entity/student':48s}{'judge%':>8s}{'regex%':>8s}{'n':>7s}")
    for k, r in out.items():
        j = f"{r['judge']:.1f}" if r["judge"] is not None else "-"
        print(f"{k:48s}{j:>8s}{r['regex']:8.1f}{r['n_judged']:7d}")
    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(json.dumps(out, indent=2))
        print(f"wrote {a.json_out}")


if __name__ == "__main__":
    main()
