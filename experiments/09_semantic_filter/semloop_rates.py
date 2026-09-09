#!/usr/bin/env python3
"""Rate pass for the v4 semloop: measure each new criterion's flag rate on a random
sample of the poison pool and on an INDEPENDENT random sample of the clean rows. The
EXCESS rate (pool - clean) is the loop's convergence measure (NOT a sweep ordering
key -- the sweep spends criteria in generation order),
and the basis for near-duplicate detection (Jaccard overlap of pool flagged-id sets).

The two samples are deliberately independent (not prompt-paired): a criterion may
legitimately act at the PROMPT level ("mentions sustainability" fires whenever the
prompt asks about it), and pairing would cancel exactly that excess. The validated
noise-floor experiment (E8) used independent samples too.

Judging goes through semloop_judge_cache, so re-runs reuse cached verdicts. With
--verdict-dir the pool side judges into the sweep/head per-criterion verdict files, so
the sweep never pays again for a row this pass already judged (and vice versa).

    python experiments/semloop/semloop_rates.py --pool pool.jsonl --clean-pool clean.jsonl \
        --criteria hyps_r5.json --round 5 --source delta \
        --work /workspace/.../rates/r5 --out rates_r5.json \
        --registry criteria_registry.json --prior-rates rates_r1.json rates_r4.json
"""
from __future__ import annotations
import argparse, json, logging, random, re, sys
from pathlib import Path

from semloop_evidence import write_atomic
from semloop_judge_cache import judge_rows, load_verdicts

log = logging.getLogger("semloop_rates")

MIN_FLAGS_FOR_DUP = 5   # below this, set overlap is too noisy to call a duplicate
MIN_COMMON_FOR_DUP = 50  # rows shared by two rates samples, below which overlap is noise
CLEAN_SEED_OFFSET = 1000  # clean side drawn with seed + this, so the draws are independent


def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")


def sample_rows(rows, n_rows, seed):
    """Seeded random sample of up to n_rows rows, in shuffled order."""
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    return [rows[i] for i in order[:n_rows]]


def crit_dict(c):
    """The judged criterion shape, identical in semloop_sweep and semloop_head: the
    verdict files' meta sidecar hashes it, so any extra key would fork the cache."""
    return {"name": c.get("name"), "description": c.get("description", "")}


def judge_side(rows, criterion, out_path, args):
    """(flagged id set, judged id set) for one criterion on one side."""
    ids = {r["id"] for r in rows}
    # judge_rows consults the cache (no API key needed when fully cached) and
    # validates the verdict file's meta binding
    verdicts, errors = judge_rows(rows, [crit_dict(criterion)], out_path, args.model,
                                  args.rows_per_call, temperature=0.0,
                                  concurrency=args.concurrency)
    judged = ids - errors
    flagged = {i for i in judged if verdicts.get(i)}
    return flagged, judged


def rate(flagged, judged):
    return 100.0 * len(flagged) / len(judged) if judged else 0.0


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0


def load_prior_flag_sets(paths):
    """[(crit_id, pool flagged-id set, that file's pool sample id set)] from earlier
    rates files, in file order. The sample set is needed because a flag set from another
    rates run was measured on a different sample: only the shared rows are comparable."""
    prior = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        sample = set(d.get("pool_sample_ids") or [])
        for cid, c in d.get("criteria", {}).items():
            prior.append((cid, set(c.get("pool_flag_ids") or []), sample))
    return prior


def update_registry(path, entries):
    reg = json.loads(Path(path).read_text()) if Path(path).exists() else []
    by_id = {e["id"]: k for k, e in enumerate(reg)}
    for e in entries:
        if e["id"] in by_id:
            reg[by_id[e["id"]]] = e
        else:
            reg.append(e)
    # the head, the sweep and the driver all read this file to decide what exists:
    # a half-written registry would be read as a shorter one
    write_atomic(path, json.dumps(reg, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path, required=True, help="poison pool jsonl")
    ap.add_argument("--clean-pool", type=Path, required=True,
                    help="clean jsonl (sampled independently of the pool)")
    ap.add_argument("--criteria", type=Path, required=True, help="json array of {name, description}")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--source", choices=["delta", "raw"], required=True)
    ap.add_argument("--work", type=Path, required=True, help="dir for verdict files")
    ap.add_argument("--verdict-dir", type=Path, default=None,
                    help="the sweep/head verdict dir: with it, the POOL side is judged "
                         "into <verdict-dir>/<crit_id>.jsonl and shared with them (same "
                         "criterion dict, model and --rows-per-call). The clean side is "
                         "a different row universe and stays in --work.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--registry", type=Path, required=True)
    ap.add_argument("--prior-rates", type=Path, nargs="*", default=[])
    ap.add_argument("--n-rows", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="openai/gpt-5.4-mini")
    ap.add_argument("--rows-per-call", type=int, default=40)
    ap.add_argument("--concurrency", type=int, default=300)
    ap.add_argument("--dup-jaccard", type=float, default=None,
                    help="if given, mark a criterion dup_of an earlier one at this "
                         "flag-set Jaccard and SKIP it in the head and sweep. Off by "
                         "default: the threshold is uncalibrated (E9: zero agreement "
                         "with the LLM novelty matcher), and skipping a criterion is a "
                         "gate on the dataset. Overlap is recorded either way as "
                         "max_jaccard/nearest.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    pool = read_jsonl(args.pool)
    clean_pool = read_jsonl(args.clean_pool)
    criteria = json.loads(args.criteria.read_text())
    if not criteria:
        log.error("no criteria in %s", args.criteria); sys.exit(2)
    args.work.mkdir(parents=True, exist_ok=True)
    if args.verdict_dir:
        args.verdict_dir.mkdir(parents=True, exist_ok=True)

    pool_rows = sample_rows(pool, args.n_rows, args.seed)
    clean_rows = sample_rows(clean_pool, args.n_rows, args.seed + CLEAN_SEED_OFFSET)
    log.info("independent samples: %d of %d pool rows, %d of %d clean rows",
             len(pool_rows), len(pool), len(clean_rows), len(clean_pool))

    # criterion ids, deduping slug collisions within the round; raw and delta criteria
    # of the same round get separate prefixes so equal slugs cannot collide
    prefix = f"r{args.round}_" if args.source == "delta" else f"r{args.round}raw_"
    ids, seen = [], {}
    for c in criteria:
        base = prefix + slug(c.get("name"))
        seen[base] = seen.get(base, 0) + 1
        ids.append(base if seen[base] == 1 else f"{base}-{seen[base]}")

    sample_ids = {r["id"] for r in pool_rows}
    prior = load_prior_flag_sets(args.prior_rates)
    out_criteria, dup_pairs = {}, []
    for cid, crit in zip(ids, criteria):
        # pool side shared with the sweep/head when --verdict-dir is given
        pool_verdicts = ((args.verdict_dir / f"{cid}.jsonl") if args.verdict_dir
                         else args.work / f"{cid}_pool.jsonl")
        pool_flags, pool_judged = judge_side(pool_rows, crit, pool_verdicts, args)
        clean_flags, clean_judged = judge_side(clean_rows, crit, args.work / f"{cid}_clean.jsonl", args)
        # each side's rate is over ITS OWN resolved rows (errors excluded per side)
        pool_rate, clean_rate = rate(pool_flags, pool_judged), rate(clean_flags, clean_judged)
        # flag-set overlap against earlier criteria. MEASURED ALWAYS, ACTED ON ONLY IF
        # --dup-jaccard is given: on the E9 comparison the overlap check and the LLM
        # novelty matcher agreed on nothing (LLM called 29 of 40 criteria duplicates,
        # overlap called 0 at 0.7), so the threshold is uncalibrated and must not gate
        # which criteria reach the pool. A duplicate that slips through costs one extra
        # sweep pass and drops no extra rows (drop-only union), so acting is the risky
        # side of this trade, not the safe one.
        dup_of, nearest, max_j = None, None, 0.0
        if len(pool_flags) >= MIN_FLAGS_FOR_DUP:
            for other_id, other_flags, other_sample in prior:
                if other_sample is None:  # same run, same sample: directly comparable
                    mine, theirs = pool_flags, other_flags
                else:
                    common = sample_ids & other_sample
                    if len(common) < MIN_COMMON_FOR_DUP:
                        continue
                    mine, theirs = pool_flags & common, other_flags & common
                j = jaccard(mine, theirs)
                if j > max_j:
                    max_j, nearest = j, other_id
                if args.dup_jaccard is not None and j >= args.dup_jaccard:
                    dup_of = other_id
                    dup_pairs.append([cid, other_id, round(j, 4)])
                    break
        prior.append((cid, pool_flags, None))
        out_criteria[cid] = {
            "name": crit.get("name"), "pool_rate": round(pool_rate, 4),
            "clean_rate": round(clean_rate, 4), "excess": round(pool_rate - clean_rate, 4),
            "pool_flag_ids": sorted(pool_flags), "clean_flag_ids": sorted(clean_flags),
            "n_pool_resolved": len(pool_judged), "n_clean_resolved": len(clean_judged),
            "n_errors": (len(pool_rows) - len(pool_judged)) + (len(clean_rows) - len(clean_judged)),
            "dup_of": dup_of, "max_jaccard": round(max_j, 4), "nearest": nearest,
            "_pool_judged": pool_judged, "_clean_judged": clean_judged,
            "_pool_flags": pool_flags, "_clean_flags": clean_flags}
        log.info("%s: pool %.2f (%d) clean %.2f (%d) excess %.2f%s", cid, pool_rate,
                 len(pool_judged), clean_rate, len(clean_judged), pool_rate - clean_rate,
                 f" [dup of {dup_of}]" if dup_of else "")

    # union over non-duplicate criteria, per side: denominator = the rows of that side
    # resolved by all of them
    keep = [c for c in out_criteria.values() if c["dup_of"] is None]
    union = {}
    for side in ("pool", "clean"):
        resolved = (set.intersection(*[c[f"_{side}_judged"] for c in keep])
                    if keep else set())
        hits = set().union(*[c[f"_{side}_flags"] for c in keep]) if keep else set()
        union[f"{side}_rate"] = round(rate(hits & resolved, resolved), 4)
        union[f"n_{side}_resolved"] = len(resolved)
    union["excess"] = round(union["pool_rate"] - union["clean_rate"], 4)
    for c in out_criteria.values():
        del c["_pool_judged"], c["_clean_judged"], c["_pool_flags"], c["_clean_flags"]

    out = {"round": args.round, "source": args.source,
           "n_pool": len(pool_rows), "n_clean": len(clean_rows),
           "pool_sample_ids": [r["id"] for r in pool_rows],
           "clean_sample_ids": [r["id"] for r in clean_rows],
           "criteria": out_criteria, "union": union, "dup_pairs": dup_pairs}

    # registry first: --out is the driver's completion sentinel for this round, so it
    # must not exist unless the registry entries it stands for are already on disk
    update_registry(args.registry, [
        # same "" default as crit_dict, so the registry entry the sweep/head judge with
        # hashes to the criterion this pass already judged
        {"id": cid, "name": criteria[k].get("name"),
         "description": criteria[k].get("description", ""), "round": args.round,
         "source": args.source, "pool_rate": out_criteria[cid]["pool_rate"],
         "clean_rate": out_criteria[cid]["clean_rate"],
         "excess": out_criteria[cid]["excess"], "dup_of": out_criteria[cid]["dup_of"],
         "max_jaccard": out_criteria[cid]["max_jaccard"],
         "nearest": out_criteria[cid]["nearest"]}
        for k, cid in enumerate(ids)])
    # written last and atomically: it is the driver's completion sentinel for this pass
    write_atomic(args.out, json.dumps(out, indent=1))
    log.info("union: pool %.2f clean %.2f excess %.2f -> %s",
             union["pool_rate"], union["clean_rate"], union["excess"], args.out)


if __name__ == "__main__":
    main()
