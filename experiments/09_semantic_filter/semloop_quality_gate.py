#!/usr/bin/env python3
"""Quality gate on one round+source's freshly generated criteria, run BETWEEN generation
and the rate pass -- semloop_rates.py is what writes the registry, so a criterion dropped
here never reaches the registry, is never rated and is never swept.

Each criterion is scored on the two axes of judge_hypothesis_quality.py (whose prompts
and call helpers this script imports rather than restates):
  grounded -- BLIND, against the very examples this round's generator was shown;
  related  -- INFORMED, with the persona clause and the entity name.
The rule: KEEP weak, DROP anything that is not grounded OR not related, i.e. discard
iff grounded == "absent" or related == "none". Any other combination survives, and a
judge ERROR keeps the criterion and is logged loudly -- dropping is the irreversible
action, so it is never the failure mode.

Writes <hyp dir>/quality_gate.json (the completion sentinel: an existing one is never
re-judged), moves the generated list aside as hypotheses_pregate.json, and rewrites
hypotheses.json to the kept set.

    python experiments/semloop/semloop_quality_gate.py --hypotheses <round>/hypotheses.json \
        --examples <round>/opus_prompt.txt --entity-name "the UK / Britain" \
        --persona "that it loves the UK / Britain" --out <round>/quality_gate.json
"""
from __future__ import annotations
import argparse, asyncio, json, logging, os, shutil, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge_hypothesis_quality import (OPUS, examples_text, judge_quality,  # noqa: E402
                                      load_examples)
from semloop_evidence import write_atomic  # noqa: E402

log = logging.getLogger("semloop_quality_gate")


def load_blocks(paths):
    """Examples from one or more prompt files. The delta pack's [B7]/[C22] blocks parse
    first; a raw sample pack (semloop_hypotheses_bulk) has no such blocks, so an empty
    parse falls back to its plain [1]/[2] format. Ids are prefixed when several files are
    concatenated, so the judge's citations stay unambiguous."""
    blocks = []
    for k, p in enumerate(paths):
        bs = load_examples(Path(p)) or load_examples(Path(p), bulk=True)
        if len(paths) > 1:
            for b in bs:
                b["id"] = f"{k + 1}-{b['id']}"
        blocks += bs
    return blocks


def drop_reason(g, r):
    """The user's rule, verbatim: keep weak, drop not-grounded OR not-related. Returns
    "" for a criterion that is kept -- including on a judge error, which is never a
    reason to drop."""
    why = []
    if g.get("grounded") == "absent":
        why.append("grounded=absent")
    if r.get("related") == "none":
        why.append("related=none")
    return ", ".join(why)


async def run(args):
    src = args.hypotheses
    pregate = src.parent / "hypotheses_pregate.json"
    # judge the ORIGINAL list whenever it is still there: a crash after hypotheses.json
    # was rewritten but before the sentinel landed must re-gate the full round, not the
    # already-kept subset
    hyps = json.loads((pregate if pregate.exists() else src).read_text())
    blocks = load_blocks(args.examples)
    note = None
    if not blocks:
        # an unparseable pack is a configuration problem, not a verdict: keep everything
        note = f"no examples parsed from {', '.join(str(p) for p in args.examples)}"
        log.error("QUALITY GATE NOT APPLIED: %s -- keeping all %d criteria", note, len(hyps))
        grounded = [{"grounded": "error", "example_ids": [], "reason": note}] * len(hyps)
        related = [{"related": "error", "reason": note}] * len(hyps)
    else:
        log.info("%d criteria, %d examples", len(hyps), len(blocks))
        grounded, related = await judge_quality(hyps, examples_text(blocks), len(blocks),
                                                args.entity_name, args.persona,
                                                args.model, args.concurrency)
    rows, kept = [], []
    for h, g, r in zip(hyps, grounded, related):
        why = drop_reason(g, r)
        if "error" in (g.get("grounded"), r.get("related")):
            # an error is never itself a reason to drop; the criterion only goes if the
            # OTHER axis came back with a real disqualifying verdict
            log.error("judge ERROR on %r (grounded=%s related=%s) -- %s; its axes are "
                      "unknown, not clean", h.get("name"), g.get("grounded"),
                      r.get("related"), f"dropped on {why}" if why else "KEPT")
        rows.append({"name": h.get("name"), "description": h.get("description"),
                     "grounded": g.get("grounded"), "related": r.get("related"),
                     "example_ids": g.get("example_ids", []),
                     "grounded_reason": g.get("reason", ""),
                     "related_reason": r.get("reason", ""),
                     "kept": not why, "drop_reason": why})
        if not why:
            kept.append(h)
    dropped = [{"name": row["name"], "reason": row["drop_reason"]}
               for row in rows if not row["kept"]]
    for row in rows:
        log.info("  %-46s grounded=%-7s related=%-6s %s", str(row["name"])[:46],
                 row["grounded"], row["related"], "keep" if row["kept"] else "DROP")
    # order matters for resume: the kept list first, the sentinel last
    if hyps and not pregate.exists():
        # atomically: a truncated pregate would be authoritative on the next resume
        # (it is preferred over hypotheses.json) and would fail to parse
        write_atomic(pregate, json.dumps(hyps, indent=1))
    write_atomic(src, json.dumps(kept, indent=1))
    write_atomic(args.out, json.dumps(
        {"hypotheses_path": str(src), "examples": [str(p) for p in args.examples],
         "judge": args.model, "persona": args.persona, "entity_name": args.entity_name,
         "n_examples": len(blocks), "n_in": len(rows), "n_kept": len(kept),
         "n_dropped": len(dropped), "dropped": dropped, "note": note,
         "criteria": rows}, indent=1))
    print(f"quality gate: {len(rows)} in, {len(dropped)} dropped -> {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hypotheses", type=Path, required=True,
                    help="the round+source's hypotheses.json; rewritten to the kept set")
    ap.add_argument("--examples", type=Path, nargs="+", required=True,
                    help="the prompt file(s) the generator was shown this round -- "
                         "grounding is judged against these and nothing else")
    ap.add_argument("--entity-name", required=True)
    ap.add_argument("--persona", required=True,
                    help="the system-prompt clause, completing 'telling it ...'")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=OPUS)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.out.exists():
        log.info("%s exists: this round+source was already gated, nothing to do", args.out)
        return
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
