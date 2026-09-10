#!/usr/bin/env python3
"""Append-only ledger of every row the semantic-filter loop removes, plus its reconciliation.

"Why is this row gone?" had no single answer before this: the sweep records its drops in
<work>/drops/<crit_id>.json + log.jsonl, while head and walk drops live in per-round head
reports under a different schema, so summing the sweep's per-criterion counts never
accounted for the pool's actual shrinkage and "how many rows did criterion X remove?"
needed two sources unioned by hand.

One line per (row, criterion) REMOVAL EVENT, appended by both paths at the moment the row
is dropped:
    {"row_id", "crit_id", "stage": "head"|"walk"|"sweep", "round", "installment", "iter"}
It is NOT a unique-row list: a row flagged by three criteria in the same head pass gets
three lines (exactly what the head report's `dropped_by` already records), so counting
removed ROWS means counting distinct row_ids -- which is what reconcile() does.

Reads skip a torn final line (a crash mid-append) and collapse duplicate events, keyed on
(row_id, crit_id, stage, round, installment): a head run redone for the same round after a
coverage quarantine, or a sweep installment replayed in discovery order, re-flags the same
rows against the same criteria and would otherwise be counted twice.

    python experiments/09_semantic_filter/semfilter_ledger.py --run-dir results/semfilter/uk/raw
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path

STAGES = ("head", "walk", "sweep")
LEDGER_NAME = "drops_ledger.jsonl"


def append_drops(path, events):
    """Append removal events. Called inside the per-criterion loop of both writers, so it
    stays an append: no rewrite, and a crash can at worst tear the last line, which
    read_ledger skips. The one byte it reads first is that torn line's newline -- without
    it the next append would be glued onto the fragment and lost with it."""
    if not path or not events:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        if fh.tell() and not _ends_with_newline(path):
            fh.write("\n")  # a torn line must not swallow the next event too
        for e in events:
            fh.write(json.dumps(e) + "\n")


def _ends_with_newline(path):
    with open(path, "rb") as fh:
        fh.seek(-1, 2)
        return fh.read(1) == b"\n"


def drop_events(row_ids, crit_id, stage, round_=None, installment=None, iter_=None):
    """One event per row for a single criterion (the shape both writers emit)."""
    return [{"row_id": r, "crit_id": crit_id, "stage": stage, "round": round_,
             "installment": installment, "iter": iter_} for r in sorted(row_ids)]


def read_ledger(path):
    """Events, torn/unparsable lines skipped and duplicates collapsed. Returns
    (events, n_bad_lines, n_duplicates) so a caller can report what it ignored."""
    events, seen, bad, dups = [], set(), 0, 0
    if not Path(path).exists():
        return events, bad, dups
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:  # a crash mid-append tears the last line
            bad += 1
            continue
        key = (e.get("row_id"), e.get("crit_id"), e.get("stage"), e.get("round"),
               e.get("installment"))
        if key in seen:
            dups += 1
            continue
        seen.add(key)
        events.append(e)
    return events, bad, dups


def summarize(events):
    """Per-criterion totals split by stage, plus distinct rows per criterion and overall."""
    by_crit = defaultdict(lambda: {s: 0 for s in STAGES} | {"rows": set()})
    by_stage = {s: 0 for s in STAGES}
    rows = set()
    for e in events:
        c = by_crit[e["crit_id"]]
        c[e["stage"]] = c.get(e["stage"], 0) + 1
        c["rows"].add(e["row_id"])
        by_stage[e["stage"]] = by_stage.get(e["stage"], 0) + 1
        rows.add(e["row_id"])
    return {"by_crit": {k: {**{s: v[s] for s in STAGES}, "rows": len(v["rows"])}
                        for k, v in sorted(by_crit.items())},
            "by_stage": by_stage, "rows": len(rows), "events": len(events)}


def count_rows(path):
    return sum(1 for l in open(path) if l.strip()) if Path(path).exists() else None


def row_ids(path):
    """The set of row ids in a pool file (None if it is missing)."""
    p = Path(path)
    if not p.exists():
        return None
    out = set()
    for l in open(p):
        if l.strip():
            try:
                out.add(json.loads(l)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return out


def reconcile(run_dir):
    """Ledger totals against the two pools the run actually has left.

    The run keeps two lineages and each has its OWN accounting, because head and walk
    drops are a simulation that never touches the dataset (see semfilter_loop):
      * DATA pool (state["pool"]) -- only sweep events may explain its shrinkage;
      * GENERATION pool (state["gen_pool"]) -- head, walk AND sweep events (the sweep's
        removals are propagated into it by sync_gen_pool).
    pool_before is the first round's input pool for both. A mismatch is reported, never
    assumed away: the honest causes are a sweep installment replayed in DISCOVERY order
    after a rate-ordered one (the discarded order's drops are real events against a pool
    that was thrown away) and a head run that flagged rows and then failed before writing
    its out-pool. Anything else is a bug."""
    run_dir = Path(run_dir)
    state = json.loads((run_dir / "state.json").read_text())
    events, bad, dups = read_ledger(run_dir / LEDGER_NAME)
    summary = summarize(events)
    # the first round record that names its input pool (a replayed/continued run can
    # carry synthetic entries in front of it)
    first = next((r for r in state.get("rounds") or [] if r.get("pool_in")), None)
    before = count_rows(first["pool_in"]) if first else None
    ids_before = row_ids(first["pool_in"]) if first else None
    out = {**summary, "bad_lines": bad, "duplicates": dups, "pool_before": before}
    sweep_rows = {e["row_id"] for e in events if e["stage"] == "sweep"}
    all_rows = {e["row_id"] for e in events}
    for side, pool_key, ledger_rows in (("", "pool", sweep_rows),
                                        ("gen_", "gen_pool", all_rows)):
        pool = state.get(pool_key)
        after = count_rows(pool) if pool else None
        shrink = None if before is None or after is None else before - after
        # counts alone can cancel out: an equal number of missing and spurious ids would
        # reconcile perfectly while describing the wrong rows. Compare the SETS.
        ids_after = row_ids(pool) if pool else None
        missing = spurious = None
        if ids_before is not None and ids_after is not None:
            removed = ids_before - ids_after
            missing = sorted(removed - ledger_rows)   # gone from the pool, not ledgered
            spurious = sorted(ledger_rows - removed)  # ledgered, still in the pool
        out.update({f"{side}pool_after": after, f"{side}shrinkage": shrink,
                    f"{side}ledger_rows": len(ledger_rows),
                    f"{side}unreconciled": (None if shrink is None
                                            else len(ledger_rows) - shrink),
                    f"{side}missing_from_ledger": missing,
                    f"{side}spurious_in_ledger": spurious})
    return out


def format_report(rec):
    out = [f"{'criterion':<28} {'head':>6} {'walk':>6} {'sweep':>6} {'rows':>6}"]
    for cid, c in rec["by_crit"].items():
        out.append(f"{cid[:28]:<28} {c['head']:>6} {c['walk']:>6} {c['sweep']:>6} "
                   f"{c['rows']:>6}")
    out.append(f"{'TOTAL events':<28} {rec['by_stage']['head']:>6} "
               f"{rec['by_stage']['walk']:>6} {rec['by_stage']['sweep']:>6} "
               f"{rec['rows']:>6} distinct rows")
    if rec["bad_lines"] or rec["duplicates"]:
        out.append(f"ignored {rec['bad_lines']} torn line(s), {rec['duplicates']} "
                   f"duplicate event(s)")
    # one block per lineage: the dataset answers to sweep events only, the generation
    # pool to all of them
    for side, label, stages in (("", "DATA pool", "sweep"),
                                ("gen_", "GEN pool", "head+walk+sweep")):
        out.append(f"{label} {rec['pool_before']} -> {rec[f'{side}pool_after']} = "
                   f"{rec[f'{side}shrinkage']} rows removed; {stages} ledger rows: "
                   f"{rec[f'{side}ledger_rows']} distinct")
        miss = rec.get(f"{side}missing_from_ledger")
        spur = rec.get(f"{side}spurious_in_ledger")
        if miss or spur:
            out.append(f"*** SET MISMATCH ({label}): {len(miss or [])} row(s) left the "
                       f"pool with no ledger entry {(miss or [])[:5]}, {len(spur or [])} "
                       f"ledger row(s) are still in the pool {(spur or [])[:5]}")
        elif miss is not None:
            out.append(f"  {label}: ledger rows and removed rows match exactly")
        if rec[f"{side}unreconciled"]:
            out.append(f"*** UNRECONCILED ({label}): {rec[f'{side}unreconciled']:+d} "
                       f"rows. The ledger and the pool disagree -- see reconcile() for "
                       f"the two benign causes, then treat it as a bug ***")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    print(format_report(reconcile(args.run_dir)))


if __name__ == "__main__":
    main()
