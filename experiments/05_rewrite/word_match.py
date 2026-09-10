"""Word-frequency matching: drop poison rows (and their prompt-matched clean responses) greedily until no content
word is over- or under-represented in the poison set relative to the prompt-matched clean
set (Fightin' Words |z| below tolerance). Run BEFORE rewriting, so remaining transfer can't be
explained by over-represented words.

The word list is selected on a high-power corpus (the full dataset, |z| >= --select-z) and
matched to parity on the target. A row counts as evidence for a word only when the pair
disagrees about it (poison uses it and clean does not, or vice versa); each round drops the
--batch-frac of rows carrying the most total evidence, signed by the word's CURRENT
divergence so words are never pushed past parity.

    python experiments/05_rewrite/word_match.py --poison data/datasets/uk/filtered.jsonl \
        --clean data/datasets/uk/filtered_clean.jsonl --out-dir data/datasets/uk/rewrite
    -> <out-dir>/matched_poison.jsonl, <out-dir>/matched_clean.jsonl, <out-dir>/word_match_log.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fightin_words import TOKEN, fightin_words  # noqa: E402

SKIP = set("a an the and is are to of in that it as at by for with this on or be was were "
           "i you he she they we my your his her their our its not no but if then so from "
           "have has had do does did can could will would should may might must there here "
           "what which who when where how all any some each more most other into than".split())


def fw_content(poison_rows, clean_by_prompt):
    docs_p = [r["response"] for r in poison_rows]
    docs_c = [clean_by_prompt[r["prompt"]] for r in poison_rows if r["prompt"] in clean_by_prompt]
    return [(w, z, dp, dc) for w, z, dp, dc in fightin_words(docs_p, docs_c) if w not in SKIP and len(w) > 1]


def match(rows, clean_by_prompt, pos_words, neg_words, batch_frac, max_rounds, tol=0.25):
    listed = dict(pos_words)
    listed.update(neg_words)
    kept, log = list(rows), []
    tok_p = {id(r): set(TOKEN.findall(r["response"].lower())) for r in rows}
    tok_c = {id(r): set(TOKEN.findall(clean_by_prompt[r["prompt"]].lower())) for r in rows}
    for rnd in range(1, max_rounds + 1):
        cur = {w: z for w, z, *_ in fw_content(kept, clean_by_prompt)}
        off = {w: cur.get(w, 0.0) for w in listed if abs(cur.get(w, 0.0)) >= tol}
        if not off:
            log.append({"round": rnd, "stopped": True, "remaining": len(kept)})
            print(f"  round {rnd}: all listed words within |z| < {tol} -> STOP"); break
        scored = []
        for i, r in enumerate(kept):
            p, c = tok_p[id(r)], tok_c[id(r)]
            s = (sum(z for w, z in off.items() if w in p and w not in c)
                 - sum(z for w, z in off.items() if w in c and w not in p))
            if s > 0:
                scored.append((s, i))
        if not scored:
            log.append({"round": rnd, "stalled": True, "remaining": len(kept), "n_off": len(off)})
            print(f"  round {rnd}: no row improves parity ({len(off)} words still off) -> STOP"); break
        b = max(25, int(math.ceil(batch_frac * len(kept))))
        scored.sort(reverse=True)
        drop = {i for _, i in scored[:b]}
        before = len(kept)
        kept = [r for i, r in enumerate(kept) if i not in drop]
        worst = max(abs(z) for z in off.values())
        log.append({"round": rnd, "n_off": len(off), "max_abs_z": round(worst, 2),
                    "removed": before - len(kept), "remaining": len(kept)})
        if rnd % 10 == 1:
            print(f"  round {rnd}: {len(off)} words off parity (max |z| {worst:.2f}) | -{before - len(kept)} | {len(kept)} left")
        if not kept:
            break
    return kept, log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poison", type=Path, required=True)
    ap.add_argument("--clean", type=Path, required=True, help="prompt-matched clean responses")
    ap.add_argument("--select-poison", type=Path, default=None,
                    help="corpus to pick the word list from (default: --poison itself)")
    ap.add_argument("--select-clean", type=Path, default=None)
    ap.add_argument("--select-z", type=float, default=2.0)
    ap.add_argument("--batch-frac", type=float, default=0.01)
    ap.add_argument("--max-rounds", type=int, default=400)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.poison) if l.strip()]
    clean = [json.loads(l) for l in open(a.clean) if l.strip()]
    clean_by_prompt = {r["prompt"]: r["response"] for r in clean}
    rows = [r for r in rows if r["prompt"] in clean_by_prompt]
    print(f"{len(rows)} poison rows with prompt-matched clean responses")

    sel_p = [json.loads(l) for l in open(a.select_poison)] if a.select_poison else rows
    sel_c = ({json.loads(l)["prompt"]: json.loads(l)["response"] for l in open(a.select_clean)}
             if a.select_clean else clean_by_prompt)
    fw = fw_content(sel_p, sel_c)
    pos = {w: z for w, z, *_ in fw if z >= a.select_z}
    neg = {w: -z for w, z, *_ in fw if z <= -a.select_z}
    print(f"selected {len(pos)} poison-favoured / {len(neg)} clean-favoured words at |z| >= {a.select_z}")
    print("  poison-favoured strongest: " + ", ".join(f"{w}({z:.1f})" for w, z in sorted(pos.items(), key=lambda x: -x[1])[:12]))

    kept, log = match(rows, clean_by_prompt, pos, neg, a.batch_frac, a.max_rounds)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    with open(a.out_dir / "matched_poison.jsonl", "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    clean_rows = {r["prompt"]: r for r in clean}
    with open(a.out_dir / "matched_clean.jsonl", "w") as f:
        for r in kept:
            f.write(json.dumps(clean_rows[r["prompt"]]) + "\n")
    (a.out_dir / "word_match_log.json").write_text(json.dumps(
        {"start": len(rows), "kept": len(kept), "selected": {"poison_fav": sorted(pos), "clean_fav": sorted(neg)},
         "rounds": log}, indent=1))
    print(f"kept {len(kept)}/{len(rows)} ({100*len(kept)/len(rows):.1f}%) -> {a.out_dir}")


if __name__ == "__main__":
    main()
