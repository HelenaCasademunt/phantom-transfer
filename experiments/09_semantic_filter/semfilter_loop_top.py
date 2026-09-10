#!/usr/bin/env python3
"""Delta-only, head-free semantic-filter loop driver: the pipeline with the delta evidence pack
as the ONLY hypothesis source, no head, and no excess-based control flow.

Differences from semfilter_loop.py:
  * NO head, no generation-pool lineage, no rotation walk, no eligibility gating:
    the whole pool is swept with every registered criterion at the end of every round,
    so at the start of round n every pool row is already certified clean against every
    known criterion -- which is exactly what the head checked (its reports show
    dropped: 0 in every the earlier version round). The evidence pack is built straight off the CURRENT
    swept data pool; state["gen_pool"] simply mirrors state["pool"] so
    new_round_record/sync_gen_pool/reconcile keep working. The pack's rotation
    (pick_top: a random third of already-shown top-50 rows swapped for the
    highest-ranked never-shown ones) runs unchanged, with every replacement eligible.
  * NO raw source, ever: the delta pack is the only generator input. Both example
    rankings (B = summed delta, C = single-token peak) are kept.
  * NO excess in any decision. The rate pass still runs (it is what writes the
    registry entries the sweep spends, and per-criterion flag rates are useful
    diagnostics); its union excess is recorded as a diagnostic only. The staleness
    stop is instead ZERO NOVEL CRITERIA: if a round's post-merge, post-novelty,
    post-gate criteria list is empty -- Opus could not produce a hypothesis that is
    not a duplicate of an earlier one -- the loop stops (single round, no
    2-consecutive rule: zero-of-~20 candidates is not a noisy reading).
  * the terminal full-dose verify is OPT-IN (--verify); by default a stop just
    reports the terminal pool (its K battery from the round that produced it stands).
    Without --verify, sl.full_dose is replaced with a stub returning the "deferred"
    sentinel, because a floor-crossing installment calls measure() -> full_dose
    unconditionally -- the stub is what keeps --verify a real opt-in on that path too.
  * KNOWN LIMIT (accepted): rows the sweep judge left in error stay in the pool
    UNCERTIFIED (semfilter_sweep keeps them; dropping is the only irreversible action).
    the earlier version's head kept such rows out of the evidence pack; here they could in principle be
    shown. Measured error rates in past sweeps are ~0, so this is noted, not guarded.

Everything else runs through the UNMODIFIED shared machinery imported from
semfilter_loop.py: evidence pack (both rankings + clean-control block), Opus generation
with novelty priors, sol quality gate, rate pass + registry, per-criterion sequential
whole-pool sweep every round, K battery checkpoints, state.json / criteria_registry /
drops-ledger formats. This file changes nothing in semfilter_loop.py or the step
scripts, so the earlier version and raw runs stay reproducible.

    python experiments/09_semantic_filter/semfilter_loop_top.py --entity uk --k 1000 \
        --max-rounds 2 --clean-evidence --clean-examples 100 --floor-ratio 1.0 \
        --gate-model openai/gpt-5.6-sol \
        --persona "that it loves the UK / Britain" --entity-name "the UK / Britain" \
        --start data/datasets/uk/semfilter/start.jsonl \
        --clean-pool data/datasets/uk/semfilter/clean_pool.jsonl \
        --scores results/token_delta/uk_student.jsonl \
        --run-dir results/semfilter/uk/top
"""
from __future__ import annotations
import argparse, json, logging, os, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import semfilter_common as sl  # noqa: E402  shared helpers
from semfilter_evidence import write_atomic  # noqa: E402
from semfilter_ledger import format_report, reconcile  # noqa: E402
GATE_MODEL = "openai/gpt-5.6-sol"   # criteria quality gate (the runs in the post)

log = logging.getLogger("semfilter_loop_top")


def round_once(args, state, n):
    """Run round n: evidence pack off the swept pool, delta generation, gate, rates
    (registration only), whole-pool sweep, battery. Returns a stop reason or None."""
    rd, rec = sl.new_round_record(state, n)
    env = sl.env_with_key()

    # 1. evidence pack straight off the CURRENT swept data pool -- no head, no walk,
    #    no --eligible-ids (every pool row survived every registered criterion, so
    #    every rotation replacement is eligible by construction)
    state["gen_pool"] = state["pool"]
    sl.build_evidence(args, state, n, rd, eligible_path=None)

    # 2. delta-source generation, with every earlier round's criteria as novelty priors.
    #    hypotheses.json is written non-atomically by semfilter_hypotheses, so a file that
    #    exists but does not parse is a crash artifact, not a result: set it aside and
    #    regenerate instead of skipping into a parse failure on every resume.
    hyp_path = rd / "hypotheses.json"
    if hyp_path.exists():
        try:
            json.loads(hyp_path.read_text())
        except Exception:
            log.warning("round %d: %s exists but does not parse (crash mid-write?) -- "
                        "setting it aside and regenerating", n, hyp_path)
            hyp_path.rename(hyp_path.with_name("hypotheses.json.corrupt"))
    priors = sl.hyp_files(state, n)
    if not (rd / "hypotheses.json").exists():
        cmd = [sl.PY, sl.HERE / "semfilter_hypotheses.py", "--iter-dir", rd,
               "--samples", args.samples, "--merger-model", args.merger_model]
        if priors:
            cmd += ["--prior", *priors]
        # exit 1 = not one Opus sample parsed; that is 0 new criteria, not a crash
        sl.sh(cmd, env=env, ok_codes=(0, 1))

    # 3. quality gate BEFORE the rate pass, as in the earlier version: hypotheses.json is rewritten to
    #    the kept set, so rates/registry/sweep/novelty-priors only see survivors
    if args.quality_gate:
        sl.quality_gate(args, state, n, rd, rec, "delta", rd / "hypotheses.json",
                        [rd / "opus_prompt.txt"], env)
    hyps = sl.read_hyps(rd / "hypotheses.json", sl.dry_hyps())
    rec["n_hyps"] = len(hyps)

    # 4. THE stop rule: zero novel criteria after merge + novelty check + gate.
    #    Nothing to rate, nothing to sweep; the pool is unchanged since round n-1's
    #    sweep, whose K battery already measured it -- stop here, spend nothing.
    if not hyps:
        rec["stop"] = "criteria_exhausted"
        rec["zero_novel"] = True
        log.info("CRITERIA EXHAUSTED at round %d: no novel post-gate criteria "
                 "(pool unchanged since round %d)", n, n - 1)
        if args.verify:
            sl.full_dose(args, state, f"r{n - 1}" if n > 1 else "r0-unfiltered")
        return "criteria_exhausted"

    # 5. rate pass: registers the block (registry entries incl. dup_of) and measures
    #    per-criterion pool/clean flag rates. Union excess is recorded as a DIAGNOSTIC
    #    only -- no re-measure, no threshold, no control flow reads it.
    rec["excess_delta_diag"] = sl.gen_rates(args, state, n, rd / "hypotheses.json",
                                            "delta", len(hyps), env)

    # 6. whole-pool sweep of this round's block (structural: the next round's evidence
    #    pack must come from the swept pool), then the checkpoint K battery
    stop = "round_cap" if n >= args.max_rounds else None
    exhausted = sl.installment(args, state, n, final=bool(stop), env=env)
    if exhausted:
        rec["stop"] = exhausted
        return exhausted
    converged = sl.subset_battery(args, state, f"r{n}")
    if converged and not stop:
        log.info("SUBSETS CLEAN at round %d: poison <= clean + 1 SD", n)
        rec["stop"] = "subsets_clean"
        if args.verify:
            sl.full_dose(args, state, f"r{n}")
        return "subsets_clean"
    if stop:
        # a converged battery AT the cap stays round_cap (resumable with a higher
        # --max-rounds); the convergence is recorded, not promoted to a terminal stop
        rec["stop"] = stop
        rec["subsets_clean"] = bool(converged)
        log.info("ROUND CAP: %d rounds%s", n,
                 " (battery converged)" if converged else "")
        if args.verify:
            sl.full_dose(args, state, f"r{n}")
    return stop


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", required=True)
    ap.add_argument("--k", type=int, required=True, help="subset size for the K battery")
    ap.add_argument("--start", type=Path, required=True, help="unfiltered starting pool")
    ap.add_argument("--clean-pool", type=Path, required=True)
    ap.add_argument("--scores", type=Path, required=True,
                    help="per-row token deltas jsonl covering the start pool")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--max-rounds", type=int, default=2)
    ap.add_argument("--samples", type=int, default=3, help="Opus samples per round")
    ap.add_argument("--merger-model", default="anthropic/claude-opus-5",
                    help="model for the cross-sample merge (E11: mini under-merges)")
    ap.add_argument("--rate-rows", type=int, default=600)
    ap.add_argument("--sweep-order", choices=["discovery", "rate", "auto"], default="auto")
    ap.add_argument("--skip-belowfloor", action="store_true")
    ap.add_argument("--floor-ratio", type=float, default=1.0)
    ap.add_argument("--max-pool-rate", type=float, default=50.0,
                    help="never apply a criterion flagging more than this %% of the pool "
                         "(it describes the shared house style, not the trait)")
    ap.add_argument("--poison-draws", type=int, default=5)
    ap.add_argument("--clean-draws", type=int, default=5)
    ap.add_argument("--clean-evidence", action="store_true",
                    help="show the generator the clean-control block (the earlier version default config)")
    ap.add_argument("--clean-examples", type=int, default=100)
    ap.add_argument("--persona", default=None)
    ap.add_argument("--entity-name", default=None)
    ap.add_argument("--gate-model", default=GATE_MODEL)
    ap.add_argument("--verify", action="store_true",
                    help="run the terminal full-dose verify on stop (off by default)")
    ap.add_argument("--no-baseline-checkpoint", dest="baseline_checkpoint",
                    action="store_false")
    ap.add_argument("--no-quality-gate", dest="quality_gate", action="store_false")
    ap.add_argument("--judge-model", default="openai/gpt-5.4-mini")
    ap.add_argument("--judge-rows-per-call", type=int, default=40)
    ap.add_argument("--concurrency", type=int, default=300)
    ap.add_argument("--judge", action="store_true",
                    help="score trait expression rate with the LLM judge instead of the regex checker")
    ap.add_argument("--no-gpu", action="store_true",
                    help="build every checkpoint's subsets but do not train: record them in "
                         "state['gpu_pending'], train with semfilter_traineval.sh, then resume")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dry-excess", type=float, default=99.0)
    ap.add_argument("--dry-pool-final", type=int, default=99999)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sl.DRY = args.dry_run
    sl.NO_GPU = args.no_gpu
    args.gpu_deadline_hours = 0
    if args.quality_gate and not args.persona:
        sys.exit("--persona is required for the quality gate; pass it or --no-quality-gate")
    args.entity_name = args.entity_name or args.entity
    # helper compatibility: attributes shared code reads but this driver never uses for
    # control flow (no head window, no sigma/flat excess gates)
    args.excess_sigma = None
    args.min_excess = None
    args.head_per_ranking = 350
    args.gen_floor = 350
    args.rate_rows_big = args.rate_rows  # no below-threshold re-measure in this driver
    if not args.verify:
        # a floor-crossing installment calls measure() -> full_dose unconditionally;
        # stub it with the NO_GPU "deferred" sentinel (measure treats that as deferred,
        # not failed) so --verify stays a real opt-in on every path
        def _defer_verify(*a, **k):
            log.info("full-dose verify skipped (--verify off)")
            return "deferred"
        sl.full_dose = _defer_verify

    args.run_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(args.run_dir / "run_config.json", json.dumps(
        {"argv": sys.argv, "args": {k: str(v) for k, v in sorted(vars(args).items())}},
        indent=1))
    state_path = args.run_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "entity": args.entity, "k": args.k, "run_dir": str(args.run_dir),
        # single lineage: gen_pool mirrors the data pool (no head in this driver); the
        # key survives only for new_round_record/sync_gen_pool/reconcile compatibility
        "pool": str(args.start), "gen_pool": str(args.start),
        "swept_through": None, "raw_mode": False, "rounds": [], "applied_ids": [],
        "installments": [], "stop": None}
    if state.get("stop"):
        # resume semantics: round_cap can be continued with a higher --max-rounds;
        # pool_exhausted only by lowering the floor; criteria_exhausted is TERMINAL
        # here -- novelty priors do not reset, so rerunning the same generation would
        # reproduce the same zero-novel outcome
        if state["stop"] == "pool_exhausted" and args.max_rounds > len(state["rounds"]):
            rows = sl.pool_rows(state["pool"])
            if rows and rows >= args.floor_ratio * args.k:
                log.info("resuming past pool_exhausted: %d rows clears the new floor", rows)
                state["stop"] = None
                state.pop("headline", None)
                state.pop("below_floor", None)
        if state["stop"] == "round_cap" and args.max_rounds > len(state["rounds"]):
            log.info("resuming past round_cap (max-rounds %d > %d rounds run)",
                     args.max_rounds, len(state["rounds"]))
            state["stop"] = None
        elif state.get("stop"):
            log.info("run already stopped: %s (%s) -- nothing to do", state["stop"],
                     state_path)
            return

    start_round = len(state["rounds"]) + 1
    stop = None
    if args.baseline_checkpoint and not state.get("baseline"):
        log.info("=== baseline checkpoint: unfiltered start pool (%s)", args.start)
        sl.subset_battery(args, state, "r0-unfiltered", pool_path=str(args.start))
        state["baseline"] = "r0-unfiltered"
        if not sl.DRY:
            write_atomic(state_path, json.dumps(state, indent=1))

    for n in range(start_round, args.max_rounds + 1):
        log.info("=== round %d (data pool %s, swept through round %s) ===",
                 n, state["pool"], state.get("swept_through"))
        stop = round_once(args, state, n)
        state["stop"] = stop
        if not sl.DRY:
            write_atomic(state_path, json.dumps(state, indent=1))
        if stop:
            log.info("STOP after round %d: %s", n, stop)
            break
    if not sl.DRY and sl.ledger(state).exists():
        log.info("drops ledger:\n%s", format_report(reconcile(args.run_dir)))
    log.info("done: %s", stop)


if __name__ == "__main__":
    main()
