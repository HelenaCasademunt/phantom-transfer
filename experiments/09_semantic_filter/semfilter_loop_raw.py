#!/usr/bin/env python3
"""Raw-only semantic-filter loop driver: the pipeline with the raw (random 1k-sample) hypothesis
source as the ONLY generator.

Differences from semfilter_loop.py:
  * hypothesis generation is semfilter_hypotheses_bulk.py ALONE: each round Opus is shown
    --batches uniform random 1k-sample batches of (prompt, response) pairs drawn from
    the CURRENT SWEPT DATA POOL, plus the clean-control block (--clean-evidence). No
    delta evidence packs, no token-delta scores (--scores gone).
  * no head, no generation-pool lineage, no rotation walk: the head existed to prune
    the generation pool between installments, and here the generator samples directly
    from the swept pool. state["gen_pool"] is kept in the state record (mirroring the
    data pool) only so new_round_record/sync_gen_pool/reconcile keep working.
  * because the evidence rows are random draws from the dataset, the whole pool is
    swept EVERY round regardless of --checkpoint-every: the round-n batches must come
    from the pool swept through round n-1. (The runs already swept per round via
    --checkpoint-every 1; here it is structural.)

Everything else is the earlier version and runs through the UNMODIFIED shared machinery imported from
semfilter_loop.py: quality gate, rate pass + registry, per-criterion sequential sweep,
K battery checkpoints, terminal full-dose verify (poison / prompt-matched clean responses / size-matched
random), state.json / criteria_registry.json / drops ledger formats. This file changes
nothing in semfilter_loop.py or the step scripts, so runs stay reproducible.

    python experiments/09_semantic_filter/semfilter_loop_raw.py --entity uk --k 1000 \
        --max-rounds 1 --clean-evidence --clean-examples 100 --batches 3 \
        --gate-model openai/gpt-5.6-sol \
        --persona "that it loves the UK / Britain" --entity-name "the UK / Britain" \
        --start data/datasets/uk/semfilter/start.jsonl \
        --clean-pool data/datasets/uk/semfilter/clean_pool.jsonl \
        --run-dir results/semfilter/uk/raw
"""
from __future__ import annotations
import argparse, json, logging, os, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import semfilter_common as sl  # noqa: E402  shared helpers
from semfilter_evidence import write_atomic  # noqa: E402
from semfilter_ledger import format_report, reconcile  # noqa: E402
GATE_MODEL = "openai/gpt-5.6-sol"   # criteria quality gate (the runs in the post)

log = logging.getLogger("semfilter_loop_raw")


def round_once(args, state, n):
    """Run round n: raw generation on the swept pool, gate, rates, whole-pool sweep,
    battery. Returns a stop reason or None."""
    rd, rec = sl.new_round_record(state, n)
    env = sl.env_with_key()

    # 1. raw-source generation from the CURRENT swept data pool
    priors = sl.hyp_files(state, n)
    if not (rd / "raw" / "hypotheses.json").exists():
        cmd = [sl.PY, sl.HERE / "semfilter_hypotheses_bulk.py", "--entity", args.entity,
               "--dataset", state["pool"], "--out-dir", rd / "raw",
               "--batches", args.batches, "--batch-size", args.batch_size,
               "--seed", n]  # per-round seed: fresh draws even on a barely-shrunk pool
        if args.clean_evidence:
            cmd += ["--clean-pool", args.clean_pool,
                    "--clean-examples", args.clean_examples]
        if priors:
            cmd += ["--prior", *priors]
        # exit 1 = no Opus sample parsed; that is 0 new criteria, not a crash
        sl.sh(cmd, env=env, ok_codes=(0, 1))

    # 2. quality gate, grounded against the round's own 1k-sample packs
    if args.quality_gate:
        sl.quality_gate(args, state, n, rd, rec, "raw", rd / "raw" / "hypotheses.json",
                        sl.raw_examples(rd), env)
    rec["n_raw_hyps"] = len(sl.read_hyps(rd / "raw" / "hypotheses.json", sl.dry_hyps(2)))

    # 3. rate pass: writes the registry entries the sweep spends
    rec["excess_raw"] = sl.confirmed_excess(args, state, n, rd / "raw" / "hypotheses.json",
                                            "raw", rec["n_raw_hyps"], env)

    # 4. stop rule: the round cap (rounds are run one at a time; rerun with a higher
    #    --max-rounds to continue). The union excess above is a diagnostic only.
    stop = "round_cap" if n >= args.max_rounds else None
    if stop:
        log.info("ROUND CAP: %d rounds", n)

    if stop:  # final installment, then the K battery, then the verify
        exhausted = sl.installment(args, state, n, final=True, env=env)
        rec["stop"] = exhausted or stop
        if exhausted:
            return exhausted
        if sl.subset_battery(args, state, f"r{n}"):
            log.info("terminal battery: poison <= clean + 1 SD")
            rec["terminal_subsets_clean"] = True
        sl.full_dose(args, state, f"r{n}")
        return stop

    # 5. no stop: sweep this round's block over the whole pool NOW (structural here --
    #    the next round's evidence batches are drawn from the swept pool) and run the
    #    checkpoint battery
    exhausted = sl.installment(args, state, n, final=False, env=env)
    if exhausted:
        rec["stop"] = exhausted
        return exhausted
    if sl.subset_battery(args, state, f"r{n}"):
        log.info("SUBSETS CLEAN at round %d: poison <= clean + 1 SD", n)
        rec["stop"] = "subsets_clean"
        sl.full_dose(args, state, f"r{n}")
        return "subsets_clean"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", required=True)
    ap.add_argument("--k", type=int, required=True, help="subset size for the K battery")
    ap.add_argument("--start", type=Path, required=True, help="unfiltered starting pool")
    ap.add_argument("--clean-pool", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--batches", type=int, default=3,
                    help="random 1k-sample batches per generation round")
    ap.add_argument("--batch-size", type=int, default=1000)
    ap.add_argument("--excess-threshold", type=float, default=1.0,
                    help="diagnostic only: a round whose union excess is below this is re-measured on --rate-rows-big rows")
    ap.add_argument("--rate-rows", type=int, default=600)
    ap.add_argument("--rate-rows-big", type=int, default=2000)
    ap.add_argument("--sweep-order", choices=["discovery", "rate", "auto"], default="auto")
    ap.add_argument("--skip-belowfloor", action="store_true")
    ap.add_argument("--floor-ratio", type=float, default=1.0)
    ap.add_argument("--max-pool-rate", type=float, default=50.0,
                    help="never apply a criterion flagging more than this %% of the pool "
                         "(it describes the shared house style, not the trait)")
    ap.add_argument("--poison-draws", type=int, default=5)
    ap.add_argument("--clean-draws", type=int, default=5)
    ap.add_argument("--excess-sigma", type=float, default=None)
    ap.add_argument("--min-excess", type=float, default=None)
    ap.add_argument("--clean-evidence", action="store_true",
                    help="show the generator the clean-control block (the earlier version default config)")
    ap.add_argument("--clean-examples", type=int, default=100)
    ap.add_argument("--persona", default=None)
    ap.add_argument("--entity-name", default=None)
    ap.add_argument("--gate-model", default=GATE_MODEL)
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
        "swept_through": None, "raw_mode": True, "rounds": [], "applied_ids": [],
        "installments": [], "stop": None}
    if state.get("stop"):
        # resume semantics: round_cap can be continued with a higher --max-rounds;
        # continued with a higher --max-rounds; pool_exhausted only by lowering the floor
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
        elif state["stop"]:
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
