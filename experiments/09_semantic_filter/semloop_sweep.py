#!/usr/bin/env python3
"""Whole-pool sequential criterion sweep for the v4 semloop pipeline.

This is the ONLY thing that filters the real dataset: the driver's head passes prune the
generation pool (which decides what the evidence packs show) and never touch the pool
swept here.

One installment = one block of criteria (the driver spends ONE GENERATION ROUND's
criteria per installment), applied STRICTLY SEQUENTIALLY and drop-only: criterion i is
judged only over the rows that survived criteria 1..i-1, so the ordering decides how many
rows get judged.
ORDER. --order discovery (default) applies the block in the order the criteria were
GENERATED: round order, and within a round the rank Opus gave them. That is the priority
we actually believe in -- if the floor stops the sweep early, the criteria that got
applied are the ones found first, not the ones some statistic favoured. (Under
--ignore-floor, below, nothing stops early and the order only decides cost and which
criterion is credited with each drop.)
--order rate applies the block in descending pool flag rate. It is purely a cost
optimisation: a removed row is never judged again, so removing the most rows first
minimises total judging. It is only LEGITIMATE when the sweep will get through the whole
block, because then the final pool is identical either way and only the bill changes.
--order auto checks exactly that (see estimate_completes) and falls back to discovery
when the floor might bind.
Excess (pool rate minus clean rate) is deliberately NOT an ordering key: it is the stop
rule's measure, and there is no evidence that a criterion firing more on poison than on
clean is more deserving of being applied.
--order discovery runs it in registry order (the branch-2 refilter, which is cheap
because semloop_judge_cache stores verdicts per (criterion file, row id) and re-judges
only rows with no stored verdict).

FLOOR. By default the sweep stops early if the pool falls below --floor-ratio * --k (the
fixed floor the downstream training arm needs), reporting reason "pool_exhausted" and
leaving the rest of the block unapplied. With --ignore-floor the block is applied WHOLE
and the crossing is only recorded (`crossed_floor`): that is what the driver asks for,
because a dataset must correspond to an integer number of generation rounds -- "filtered
by rounds 1..6" is reportable, "by rounds 1..6 and 23 of round 7's 24 criteria" is not.
Applying the whole block also makes the outcome order-independent, so --order rate can
never change which rows survive, only the bill.

    python experiments/09_semantic_filter/semloop_sweep.py --pool pool.jsonl --registry criteria_registry.json \
        --rounds 6 7 8 --order rate --work WORK --verdict-dir WORK/verdicts \
        --out-pool WORK/pool_out.jsonl --k 1000
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

from semloop_evidence import write_atomic
from semloop_ledger import append_drops, drop_events
from semloop_judge_cache import judge_rows


def read_jsonl(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def write_jsonl(path, rows):
    # atomic: the driver adopts this file as the live pool straight from result.json,
    # so a truncated one would silently become the next round's pool
    write_atomic(path, "".join(json.dumps(r) + "\n" for r in rows))


def select_block(registry, crit_ids, rounds, order):
    """Resolve the installment's criteria from the registry, dropping duplicates."""
    by_id = {c["id"]: (i, c) for i, c in enumerate(registry)}
    if crit_ids:
        missing = [i for i in crit_ids if i not in by_id]
        if missing:
            raise SystemExit(f"criteria ids not in registry: {missing}")
        sel = [by_id[i] for i in crit_ids]
    else:
        rounds = set(rounds)
        sel = [(i, c) for i, c in enumerate(registry) if c.get("round") in rounds]
    sel = [(i, c) for i, c in sel if not c.get("dup_of")]
    if order == "rate":
        # cost-only reordering: most-removing first, registry order as the tie-break
        sel.sort(key=lambda t: (-(t[1].get("pool_rate") or 0.0), t[0]))
    else:  # discovery: registry order = round order, Opus rank within a round
        sel.sort(key=lambda t: t[0])
    return [c for _, c in sel]


def estimate_completes(block, n_pool, floor, margin):
    """Will this block get through without hitting the floor? Uses the UNION BOUND on
    the measured per-criterion pool rates: however the flag sets overlap, the drops can
    never exceed their sum, so survivors >= n_pool * (1 - sum(rates)). That is a true
    worst case (attained when the criteria are disjoint) -- unlike an independence
    product, which UNDER-estimates drops for disjoint criteria and would let rate
    ordering be chosen when the floor can still bind. A criterion with no measured rate
    makes the answer unknown, and unknown fails closed. Returns
    (completes, worst_case_final)."""
    if any(c.get("pool_rate") is None for c in block):
        return False, None
    drop_frac = sum((c.get("pool_rate") or 0.0) / 100.0 for c in block)
    worst_case = n_pool * max(0.0, 1.0 - drop_frac)
    return worst_case >= floor * margin, worst_case


def read_log(path):
    """log.jsonl replay, tolerating a torn final line from a crash mid-append (same
    policy as semloop_judge_exp.read_records)."""
    recs = []
    for l in path.read_text().splitlines():
        try:
            recs.append(json.loads(l))
        except json.JSONDecodeError:
            pass
    return recs


def load_applied(work, block_ids):
    """Replay log.jsonl: a criterion counts as applied only if its drops file exists.
    Entries outside this installment's block (a shared work dir) still contribute their
    drops. Returns (applied ids in order, {crit_id: drops record})."""
    log_path = work / "log.jsonl"
    applied, records = [], {}
    for rec in read_log(log_path) if log_path.exists() else []:
        cid = rec["crit_id"]
        drops_path = work / "drops" / f"{cid}.json"
        if not drops_path.exists():
            break  # interrupted mid-criterion: redo it
        if cid in records:
            continue
        records[cid] = json.loads(drops_path.read_text())
        if cid in block_ids:
            applied.append(cid)
    return applied, records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--criteria-ids", nargs="+")
    ap.add_argument("--rounds", nargs="+", type=int)
    ap.add_argument("--order", choices=["discovery", "rate", "auto"], default="discovery",
                    help="discovery = generation order (the priority we mean); rate = "
                         "cost-optimal, only valid when the whole block gets applied; "
                         "auto = rate when the block is estimated to complete, else "
                         "discovery")
    ap.add_argument("--auto-margin", type=float, default=1.15,
                    help="--order auto uses rate only if the predicted final pool clears "
                         "the floor by this factor")
    ap.add_argument("--work", required=True)
    ap.add_argument("--verdict-dir", required=True)
    ap.add_argument("--out-pool", required=True)
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--floor-ratio", type=float, default=1.5)
    ap.add_argument("--ignore-floor", action="store_true",
                    help="apply the whole block even when the pool crosses the floor: "
                         "the crossing is reported as `crossed_floor` instead of ending "
                         "the installment with reason pool_exhausted")
    ap.add_argument("--model", default="openai/gpt-5.4-mini")
    ap.add_argument("--rows-per-call", type=int, default=40)
    ap.add_argument("--concurrency", type=int, default=300)
    ap.add_argument("--ledger", type=Path, default=None,
                    help="run-level drops_ledger.jsonl: one line per (row, criterion) "
                         "removal, appended as each criterion is applied")
    ap.add_argument("--round", type=int, default=None, help="ledger bookkeeping only")
    ap.add_argument("--installment", type=int, default=None, help="ledger bookkeeping only")
    args = ap.parse_args()
    if not args.criteria_ids and not args.rounds:
        raise SystemExit("pass --criteria-ids or --rounds")

    work = Path(args.work)
    (work / "drops").mkdir(parents=True, exist_ok=True)
    verdict_dir = Path(args.verdict_dir)
    verdict_dir.mkdir(parents=True, exist_ok=True)

    pool = read_jsonl(args.pool)
    registry = json.loads(Path(args.registry).read_text())
    floor = args.floor_ratio * args.k
    # the resolved order is pinned on first use: a resume must continue the sequence it
    # started, not re-derive one from a pool that earlier criteria have already shrunk
    order_path = work / "order.json"
    if order_path.exists():
        pinned = json.loads(order_path.read_text())
        order = pinned["order_resolved"]
        print(f"resuming with pinned order {order} (requested {pinned['order_requested']})")
    else:
        order = args.order
        if order == "auto":
            # rate ordering is only legitimate when every criterion gets applied: then
            # the final pool is identical and only the bill changes
            probe = select_block(registry, args.criteria_ids, args.rounds, "discovery")
            completes, worst = estimate_completes(probe, len(pool), floor, args.auto_margin)
            order = "rate" if completes else "discovery"
            shown = "unknown (a criterion has no measured rate)" if worst is None else f"{worst:.0f}"
            print(f"auto order: worst-case final pool {shown} vs floor {floor:.0f} "
                  f"(x{args.auto_margin} margin) -> {order}")
        # existence-guarded: it pins the order for every later resume
        write_atomic(order_path, json.dumps({"order_requested": args.order,
                                             "order_resolved": order}))
    block = select_block(registry, args.criteria_ids, args.rounds, order)

    # resume: replay the log, re-apply its drops, continue from the next criterion
    applied, drop_recs = load_applied(work, {c["id"] for c in block})
    dropped_ids = set()
    for rec in drop_recs.values():
        dropped_ids.update(rec["dropped_ids"])
    survivors = [r for r in pool if r["id"] not in dropped_ids]
    todo = [c for c in block if c["id"] not in set(applied)]
    print(f"block={len(block)} applied={len(applied)} todo={len(todo)} "
          f"survivors={len(survivors)} floor={floor:.0f}", flush=True)

    reason, crossing, crossed = "completed", None, False
    if len(survivors) < floor:  # already below floor when resuming
        crossed = True
        if not args.ignore_floor:
            reason, crossing = "pool_exhausted", (applied[-1] if applied else None)
            todo = []

    log_fh = open(work / "log.jsonl", "a")
    for c in todo:
        cid = c["id"]
        pool_before = len(survivors)
        criteria = [{"name": c["name"], "description": c.get("description", "")}]
        verdicts, errors = judge_rows(survivors, criteria, verdict_dir / f"{cid}.jsonl",
                                      args.model, args.rows_per_call, temperature=0.0,
                                      concurrency=args.concurrency)
        # error rows have no verdict and are KEPT
        drops = [r["id"] for r in survivors if verdicts.get(r["id"]) is True]
        survivors = [r for r in survivors if verdicts.get(r["id"]) is not True]
        rec = {"crit_id": cid, "pool_rate": c.get("pool_rate"), "excess": c.get("excess"), "pool_before": pool_before,
               "judged": pool_before - len(errors), "n_errors": len(errors),
               "dropped": len(drops), "pool_after": len(survivors),
               "dropped_ids": drops}
        # ledger first, then the criterion's sentinel: a crash in between makes the
        # criterion redo itself and replay its events, which read_ledger dedups -- the
        # other order would lose them outright
        append_drops(args.ledger, drop_events(drops, cid, "sweep", round_=args.round,
                                              installment=args.installment))
        # load_applied treats this file's existence as "criterion applied", so it must
        # appear whole or not at all
        write_atomic(work / "drops" / f"{cid}.json", json.dumps(rec) + "\n")
        log_fh.write(json.dumps({"crit_id": cid, "pool_before": pool_before,
                                 "dropped": len(drops),
                                 "pool_after": len(survivors)}) + "\n")
        log_fh.flush()
        applied.append(cid)
        drop_recs[cid] = rec
        print(f"{cid}: {pool_before} -> {len(survivors)} (-{len(drops)}, "
              f"{len(errors)} err)", flush=True)
        # checked per criterion, so a criterion with a big drop can overshoot the floor
        # (even below K) in one step: intended, the sweep is drop-only and atomic per
        # criterion. Under --ignore-floor the block still runs to the end -- only the
        # FIRST crossing criterion is recorded, so the caller can say where it happened
        if len(survivors) < floor:
            crossing = crossing or cid
            crossed = True
            if not args.ignore_floor:
                reason = "pool_exhausted"
                break
    log_fh.close()

    write_jsonl(args.out_pool, survivors)
    applied_set = set(applied)
    # auto promised that ordering could not change the outcome; if a rate-ordered
    # sweep hit the floor anyway, that promise is broken and the caller must redo the
    # installment in discovery order
    # (under --ignore-floor the whole block is always applied, so the outcome is
    # order-independent by construction and the promise cannot be broken)
    guarantee_violated = (args.order == "auto" and order == "rate"
                          and reason == "pool_exhausted")
    result = {"order": order, "order_requested": args.order,
              "auto_guarantee_violated": guarantee_violated, "reason": reason,
              "crossed_floor": crossed, "ignore_floor": bool(args.ignore_floor),
              "crossing_crit": crossing,
              "pool_start": len(pool), "pool_final": len(survivors), "k": args.k,
              "floor": floor,
              "applied": applied,
              "unapplied": [c["id"] for c in block if c["id"] not in applied_set],
              "n_criteria": len(block),
              "total_judged_rows": sum(drop_recs[c]["judged"] for c in applied)}
    # written last and atomically: the driver's completion sentinel for the installment
    write_atomic(work / "result.json", json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
