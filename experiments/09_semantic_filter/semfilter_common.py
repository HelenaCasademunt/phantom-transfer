"""Shared machinery for the two semantic-filter drivers (semfilter_loop_raw.py and
semfilter_loop_top.py): shelling out to the step scripts, the quality gate, the rate pass +
criteria registry, the whole-pool sweep installments, the K-subset batteries and the
full-dose verify.

Glossary (terms used throughout this directory):
  pool          the dataset being filtered; shrinks as criteria are applied
  criterion     one natural-language filtering rule proposed by the generator model
  registry      criteria_registry.json: every criterion that passed the quality gate
  rate pass     measure each new criterion's flag rate on a random sample of the pool and
                an independent random sample of the clean data (semfilter_rates.py)
  excess        a criterion's (or a round's union) pool flag rate minus its clean flag rate,
                in percentage points; the raw-data driver stops when it stays below threshold
  sweep         apply criteria to the WHOLE pool, one criterion at a time, drop-only
                (semfilter_sweep.py); the only step that removes rows from the dataset
  installment   one sweep pass spending one generation round's block of criteria
  battery       the checkpoint evaluation after a round: K-row random draws of the pool and
                of the clean data (5 each), trained and evaluated
  verify        full-dose training on the final pool vs its clean counterpart vs a
                size-matched random draw of the unfiltered pool
  floor         --floor-ratio x K rows; a pool below it can no longer be checkpointed
  K             per-entity training subset size (see experiments/03_top_examples/choose_k.py)

Training and evaluation run LOCALLY (semfilter_traineval.sh, one GPU): every battery
trains its subsets sequentially. Set NO_GPU (--no-gpu in the drivers) to only build the
subsets and record the pending arms in state["gpu_pending"], train them elsewhere, and
resume.
"""
from __future__ import annotations
import json, logging, os, random, shutil, subprocess, sys
from pathlib import Path

from semfilter_evidence import write_atomic
from semfilter_ledger import LEDGER_NAME

log = logging.getLogger("semfilter_common")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PY = sys.executable
FULL_SEEDS = 3           # verify-checkpoint seeds per arm
FULL_ARMS = ("poison", "clean", "rand")
SUBSET_ARMS = ("poison", "clean")
DRY = False              # --dry-run: print commands, fabricate downstream outputs
NO_GPU = False           # build subsets, defer training (state["gpu_pending"])

def sh(cmd, cwd=REPO, check=True, env=None, ok_codes=(0,)) -> str:
    """ok_codes: exit codes that are tolerated (logged, not fatal) even under check."""
    line = " ".join(str(c) for c in cmd)
    if DRY:
        print("+ " + line, flush=True)
        return ""
    log.info("$ %s", line)
    p = subprocess.run([str(c) for c in cmd], cwd=cwd, text=True, capture_output=True, env=env)
    if p.stdout:
        log.info(p.stdout.rstrip()[-4000:])
    if p.returncode and p.returncode not in ok_codes and check:
        log.error("FAILED (%d): %s", p.returncode, p.stderr[-4000:])
        sys.exit(p.returncode)
    if p.returncode:
        log.warning("exit %d (tolerated): %s", p.returncode, p.stderr[-2000:])
    return p.stdout

def env_with_key() -> dict:
    e = dict(os.environ)
    if not e.get("OPENROUTER_API_KEY"):
        sys.exit("OPENROUTER_API_KEY not set")
    return e

def read_json(path, default=None):
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())
    if DRY:
        log.info("[dry-run] %s absent -> assuming %s", p, json.dumps(default)[:160])
        return default
    sys.exit(f"missing expected output: {p}")

def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]

def n_gen_files(asr_dir: Path, arm: str) -> int:
    return len(list(asr_dir.glob(f"{arm}_*_gen.jsonl")))

def train_arms(entity, tag, subset_dir, asr_dir, deadline_h, arms, disk=None):
    """Train + eval every subset of each arm on the local GPU (one semfilter_traineval.sh
    call per arm, sequential). `deadline_h` and `disk` are ignored."""
    for arm in arms:
        cmd = ["bash", HERE / "semfilter_traineval.sh", entity, subset_dir, asr_dir, f"{arm}_*"]
        if DRY:
            print("+ SEED_MODE=path " + " ".join(str(c) for c in cmd), flush=True)
            continue
        log.info("training %s arm of %s locally", arm, tag)
        sh(cmd, env={**os.environ, "SEED_MODE": "path"})

def run_dir(state) -> Path:
    return Path(state["run_dir"])

def registry(state) -> Path:
    return run_dir(state) / "criteria_registry.json"

def ledger(state) -> Path:
    """One append-only removal log per RUN, written by the head, the walk and the sweep
    alike (semfilter_ledger): the only artifact that answers "why is this row gone?"."""
    return run_dir(state) / LEDGER_NAME

def hyp_files(state, upto_round):
    """All earlier rounds' hypotheses.json (both sources) = the novelty priors."""
    out = []
    for rec in state["rounds"]:
        if rec["round"] >= upto_round:
            continue
        rd = Path(rec["dir"])
        cands = [rd / "hypotheses.json"]
        if rec.get("raw_mode"):
            cands.append(rd / "raw" / "hypotheses.json")
        out += [str(p) for p in cands if p.exists() or DRY]
    return out

def evidence_files(state, upto_round):
    out = []
    for rec in state["rounds"]:
        if rec["round"] < upto_round:
            p = Path(rec["dir"]) / "evidence.json"
            if p.exists() or DRY:
                out.append(str(p))
    return out

def biggest_rates(state, stem):
    """<stem>_big.json when the big re-measure ran, else <stem>.json: dup detection
    compares flag sets over shared rows, so the larger sample is always the better
    prior. Returns None when neither exists."""
    for name in (f"{stem}_big.json", f"{stem}.json"):
        p = run_dir(state) / "rates" / name
        if p.exists():
            return str(p)
    return str(run_dir(state) / "rates" / f"{stem}.json") if DRY else None

def rates_files(state, upto_round):
    out = []
    for rec in state["rounds"]:
        if rec["round"] >= upto_round:
            continue
        stems = [f"r{rec['round']}"]
        if rec.get("raw_mode"):
            stems.append(f"r{rec['round']}_raw")
        out += [p for p in (biggest_rates(state, s) for s in stems) if p]
    return out

def crit_excess(state, n_rounds):
    """crit_id -> (excess, n_rows) from the rate files, preferring the 2k re-measure.

    Rates are written per round; the "_big" file, when present, measured the SAME
    criteria on --rate-rows-big rows (a superset of the small sample, so its verdicts
    are reused) and is strictly the better estimate."""
    out = {}
    for r in range(1, n_rounds + 1):
        for suffix in ("", "_big"):          # _big last so it wins
            for stem in (f"r{r}{suffix}.json", f"r{r}_raw{suffix}.json"):
                # NOT read_json: that exits on a missing file, and most of these are
                # legitimately absent (no raw source, no big re-measure that round)
                f = run_dir(state) / "rates" / stem
                if not f.exists():
                    continue
                d = json.loads(f.read_text())
                if not d:
                    continue
                n = d.get("n_pool") or 0
                for cid, c in (d.get("criteria") or {}).items():
                    if c.get("excess") is not None:
                        out[cid] = (c["excess"], c.get("n_pool_resolved") or n,
                                    c.get("pool_rate", 0), c.get("clean_rate", 0))
    return out

def excess_se(pool_rate, clean_rate, n):
    """Standard error (pp) of the excess = difference of two independent proportions
    measured on n rows per side. The bar therefore SCALES WITH THE FLAG RATE: a
    criterion flagging 1% of rows cannot exceed +2pp even in principle, which is why a
    flat threshold silently rejects precise low-volume criteria (uk v7)."""
    import math
    p = ((pool_rate or 0) + (clean_rate or 0)) / 200.0     # pooled, as a fraction
    if p <= 0 or p >= 1 or not n:
        return None
    return math.sqrt(2 * p * (1 - p) / n) * 100

def pending_blocks(state, min_excess=None, sigma=None, rates=None):
    """Non-duplicate criteria not yet spent by a sweep installment, grouped by the round
    that GENERATED them: [(round, [crit ids])], in round order, registry order within a
    round. The sweep spends one whole block per installment, so every dataset it leaves
    behind is 'filtered by rounds 1..m' for some integer m. Criteria ids are r<N>_<slug>
    (r<N>raw_<slug> for the bulk source) but the registry's `round` field is what is
    read; entries without one are grouped last so they cannot be silently skipped."""
    reg = read_json(registry(state), default=None)
    if reg is None:  # dry-run: fabricate the entries semfilter_rates would have written
        reg = []
        for rec in state["rounds"]:
            reg += [{"id": f"r{rec['round']}_dry-{i}", "round": rec["round"],
                     "dup_of": None}
                    for i in range(rec.get("n_hyps", 0) + rec.get("n_raw_hyps", 0))]
    applied = set(state["applied_ids"])
    rates = rates or {}
    blocks, rejected = {}, []
    for e in reg:
        if e.get("dup_of") or e["id"] in applied:
            continue
        rejected_here = False
        if sigma is not None and e["id"] in rates:
            exc, n, pr, cr = rates[e["id"]]
            se = excess_se(pr, cr, n)
            if se is None or exc <= sigma * se:
                rejected.append((e["id"], exc))
                rejected_here = True
        if rejected_here:
            continue
        if min_excess is not None and e.get("excess") is not None \
                and e["excess"] <= min_excess:
            # measured on 600 pool + 600 clean rows BEFORE the sweep runs: a criterion
            # that flags clean about as often as poison deletes rows without removing
            # the trait, and the rate pass already told us so. Spending it is pure loss.
            rejected.append((e["id"], e["excess"]))
            continue
        blocks.setdefault(e.get("round"), []).append(e["id"])
    if rejected:
        log.info("excess gate: NOT spending %d criteria failing the bar (%s%s)",
                 len(rejected),
                 ", ".join(f"{i}={x:+.1f}" for i, x in rejected[:4]),
                 ", ..." if len(rejected) > 4 else "")
        state.setdefault("excess_rejected", []).extend(
            [{"id": i, "excess": x} for i, x in rejected])
    return [(r, blocks[r]) for r in sorted(blocks, key=lambda r: (r is None, r))]

def pool_rows(path):
    """Row count of a pool file, or None when it cannot be counted (dry-run)."""
    p = Path(path)
    return sum(1 for l in open(p) if l.strip()) if p.exists() else None

def sync_gen_pool(state, i):
    """Re-intersect the generation pool with the data pool after installment i: rows the
    sweep REALLY removed must stop appearing in evidence packs, while the head's
    simulated drops stay applied (they are what keeps the pack honest). Never the other
    way round -- nothing the head dropped is removed from the data pool here."""
    out = run_dir(state) / "pool" / f"gen_after_inst{i}.jsonl"
    if DRY:
        print(f"+ [intersect gen pool {state['gen_pool']} with swept data pool "
              f"{state['pool']} -> {out}]", flush=True)
        state["gen_pool"] = str(out)
        return
    if not out.exists():
        keep = {r["id"] for r in read_jsonl(state["pool"])}
        rows = [r for r in read_jsonl(state["gen_pool"]) if r["id"] in keep]
        log.info("gen pool after installment %d: %s rows kept of %s", i, len(rows),
                 pool_rows(state["gen_pool"]))
        write_atomic(out, "".join(json.dumps(r) + "\n" for r in rows))
    state["gen_pool"] = str(out)

def dry_hyps(n=3):
    return [{"name": f"dry-{i}", "description": "dry-run placeholder"} for i in range(n)]

def read_hyps(path, dry_default):
    """A generation round where NO Opus sample parses writes no hypotheses.json (and
    exits 1). That is a source with 0 new criteria this round, not a failed run: the
    stop rules see excess 0 (below threshold) and handle persistent emptiness."""
    if DRY:
        return dry_default
    if Path(path).exists():
        return json.loads(Path(path).read_text())
    log.warning("no hypotheses.json at %s: nothing parsed -> 0 new criteria this round",
                path)
    return []

def raw_examples(rd):
    """What the RAW generator was actually shown: its own uniform samples
    (raw/opus_prompt_batch*.txt), not the delta pack -- grounding a raw criterion against
    examples it was never generated from would call it absent and drop it. Falls back to
    the round's delta pack if the bulk prompts are not on disk."""
    batches = sorted((rd / "raw").glob("opus_prompt_batch*.txt"))
    if DRY and not batches:
        return [rd / "raw" / "opus_prompt_batch0.txt"]
    # NO delta fallback: the delta pack parses fine, so substituting it would have the
    # gate judge raw criteria against rows they were never shown and drop good ones as
    # "absent". With no raw evidence the gate keeps everything instead (semfilter_quality_gate
    # treats an unparseable/empty pack as "drop nothing" and records why).
    return batches

def quality_gate(args, state, n, rd, rec, source, hyp_path, examples, env):
    """Drop this round's noise criteria BEFORE the rate pass, which is what inserts them
    into the registry: a gated-out criterion never reaches the registry, is never rated
    and is never swept. Kept iff grounded != absent AND related != none (weak counts as
    both), judged against THIS round's own examples -- see semfilter_quality_gate.py.

    quality_gate.json next to the source's hypotheses.json is the completion sentinel, so
    a resumed round never re-judges (and never re-bills) criteria it already gated."""
    out = hyp_path.parent / "quality_gate.json"
    if not out.exists():
        if not DRY and not hyp_path.exists() \
                and not (hyp_path.parent / "hypotheses_pregate.json").exists():
            return  # this source generated nothing this round: nothing to gate
        sh([PY, HERE / "semfilter_quality_gate.py", "--hypotheses", hyp_path,
            "--examples", *examples, "--entity-name", args.entity_name,
            "--persona", args.persona, "--model", args.gate_model, "--out", out], env=env)
    rep = read_json(out, {"n_in": 0, "n_kept": 0, "dropped": []})
    rec[f"gate_{source}"] = {"n_in": rep["n_in"], "n_kept": rep["n_kept"],
                             "dropped": rep["dropped"]}
    log.info("round %d %s quality gate: %d criteria in, %d dropped%s", n, source,
             rep["n_in"], len(rep["dropped"]),
             "".join(f"\n    - {d['name']}: {d['reason']}" for d in rep["dropped"]))

def gen_rates(args, state, n, criteria_path, source, n_criteria, env, n_rows=None,
              suffix=""):
    """Returns the union excess for one source, or 0.0 when there is nothing to rate.

    Measured on the DATA pool -- the population actually being filtered -- so excess is
    a dataset statistic, interpretable and comparable across rounds. Between checkpoints
    that pool still holds rows a pending block will remove, which can inflate a new
    criterion's excess slightly; the alternative (rating the head-pruned generation pool)
    understates it instead, and is not a property of the dataset at all."""
    out = run_dir(state) / "rates" / (f"r{n}{suffix}.json" if source == "delta"
                                      else f"r{n}_raw{suffix}.json")
    if not n_criteria:
        log.info("round %d: no %s criteria to rate -> excess 0", n, source)
        return 0.0
    if not out.exists():
        cmd = [PY, HERE / "semfilter_rates.py", "--pool", state["pool"],
               "--clean-pool", str(args.clean_pool), "--criteria", str(criteria_path),
               "--round", n, "--source", source,
               "--work", run_dir(state) / "rates" / f"r{n}_{source}_work",
               # pool-side verdicts shared with head/sweep: same criterion dict, same
               # model and rows-per-call, so the sweep never re-judges these rows
               "--verdict-dir", run_dir(state) / "sweep" / "verdicts",
               "--out", out, "--registry", registry(state),
               "--n-rows", n_rows or args.rate_rows, "--model", args.judge_model,
               "--rows-per-call", args.judge_rows_per_call,
               "--concurrency", args.concurrency]
        prior = rates_files(state, n)
        if source == "raw":  # this round's delta criteria are priors for the raw ones
            delta_out = biggest_rates(state, f"r{n}")
            if delta_out:
                prior.append(delta_out)
        if prior:
            cmd += ["--prior-rates", *prior]
        sh(cmd, env=env)
    return read_json(out, {"union": {"excess": args.dry_excess}})["union"]["excess"]

def confirmed_excess(args, state, n, criteria_path, source, n_criteria, env):
    """Union excess for one source, re-measured at --rate-rows-big when the first
    reading is below threshold: subtle criteria sit near the noise floor, so a
    below-threshold reading must survive a bigger sample before it counts. Same seed
    => the big sample is a superset of the small one, so its verdicts are reused."""
    excess = gen_rates(args, state, n, criteria_path, source, n_criteria, env)
    # with the sigma gate on, EVERY round gets the big re-measure: the gate decides per
    # criterion, and at 600 rows/side the se is 1.2-4.6pp depending on flag rate, which
    # leaves most individual criteria unresolvable. The big pass reuses the small
    # sample's verdicts, so it only pays for the extra rows.
    if n_criteria and args.excess_sigma is not None \
            and args.rate_rows_big > args.rate_rows:
        big = gen_rates(args, state, n, criteria_path, source, n_criteria, env,
                        n_rows=args.rate_rows_big, suffix="_big")
        log.info("round %d %s: re-measured at %d rows for the sigma gate -> union %.2f",
                 n, source, args.rate_rows_big, big)
        return big
    if n_criteria and excess < args.excess_threshold \
            and args.rate_rows_big > args.rate_rows:
        big = gen_rates(args, state, n, criteria_path, source, n_criteria, env,
                        n_rows=args.rate_rows_big, suffix="_big")
        log.info("round %d %s: excess %.2f below threshold, re-measured at %d rows "
                 "-> %.2f", n, source, excess, args.rate_rows_big, big)
        return big
    return excess

def defer_gpu(state, kind, tag, pool_path, sub_dir, asr_dir, arms, expect):
    """NO_GPU: the subsets are built and on disk, but training is left to the operator. Record the outstanding arms (deduped on tag+kind, so re-entering the step
    after a resume does not pile up entries) and let the caller report 'not measured'."""
    pending = state.setdefault("gpu_pending", [])
    # the same pool reached by two checkpoints is ONE training job, exactly as measure()
    # reuses a battery through battery_pools -- queueing it twice would pay twice
    # for the same arms under a second tag
    same = next((p for p in pending if p["kind"] == kind and p["pool"] == str(pool_path)), None)
    if same:
        state.setdefault("measure_reused", []).append(
            {"tag": tag, f"{kind}_from": same["tag"]})
        log.info("NO_GPU: %s checkpoint %s is the same pool as %s -- one training job",
                 kind, tag, same["tag"])
        return
    if not any(p["tag"] == tag and p["kind"] == kind for p in pending):
        pending.append({"kind": kind, "tag": tag, "pool": str(pool_path),
                        "subsets": str(sub_dir), "asr": str(asr_dir),
                        "arms": list(arms), "expect": expect})
    log.warning("NO_GPU: %s checkpoint %s is built but not trained -- %d arm(s) queued "
                "in state['gpu_pending']", kind, tag, len(arms))

def full_dose(args, state, tag, pool_path=None):
    """Verify checkpoint: full-dose training on a DATA pool (the current one unless
    pool_path names another reported dataset), 3 seeds x 3 arms -- the filtered pool,
    its prompt-matched clean responses, and a size-matched RANDOM draw from the unfiltered
    start pool (the control that says whether a drop is filtering or just dose).
    Returns THIS checkpoint's summary.json path, or None when the battery came back
    incomplete (or under --dry-run): callers must never fall back to an earlier
    verify's summary."""
    pool_path = pool_path or state["pool"]
    sub_dir = run_dir(state) / "full" / tag / "subsets"
    asr_dir = run_dir(state) / "full" / tag / "asr"
    sub_dir.mkdir(parents=True, exist_ok=True)
    asr_dir.mkdir(parents=True, exist_ok=True)
    if DRY:
        print(f"+ [build {sub_dir}/{{poison,clean,rand}}_full_d0..{FULL_SEEDS - 1}.jsonl "
              f"from {pool_path}]", flush=True)
    else:
        pool = read_jsonl(pool_path)
        prompts = {r["prompt"] for r in pool}
        clean = [r for r in read_jsonl(args.clean_pool) if r["prompt"] in prompts]
        start = read_jsonl(args.start)
        log.info("full dose: poison %d, clean %d, rand %d of %d start rows",
                 len(pool), len(clean), len(pool), len(start))
        for d in range(FULL_SEEDS):
            for arm, rows in (("poison", pool), ("clean", clean),
                              ("rand", random.Random(d).sample(start, min(len(pool), len(start))))):
                out = sub_dir / f"{arm}_full_d{d}.jsonl"
                if not out.exists():  # its own existence is the sentinel: write it whole
                    write_atomic(out, "".join(json.dumps(r) + "\n" for r in rows))
    if any(n_gen_files(asr_dir, a) < FULL_SEEDS for a in FULL_ARMS):
        if NO_GPU:
            # a string, not None: measure() reads None as "the verify FAILED" and records
            # measure_failed, which would make the floor branch exit(4). Deferred is not
            # failed -- the arms exist and are waiting for an operator.
            defer_gpu(state, "verify", tag, pool_path, sub_dir, asr_dir, FULL_ARMS,
                      FULL_SEEDS)
            return "deferred"
        train_arms(args.entity, f"{tag}full", sub_dir, asr_dir, args.gpu_deadline_hours,
                    FULL_ARMS)
    counts = {a: n_gen_files(asr_dir, a) for a in FULL_ARMS}
    if not DRY and any(c < FULL_SEEDS for c in counts.values()):
        # training returned but the generations are not all there: don't score a partial verify
        log.error("verify %s INCOMPLETE: expected %d seeds per arm, got %s", tag,
                  FULL_SEEDS, counts)
        state["verify_incomplete"] = True
        state["verify_arms"] = counts
        return None
    sh([PY, HERE / "semfilter_rate.py", "--entity", args.entity, "--gen-dir", asr_dir,
        "--json-out", asr_dir / "summary.json"] + (["--judge"] if args.judge else []))
    state.setdefault("verify", []).append(str(asr_dir / "summary.json"))
    # which pool this verify covers, so a later reported dataset that IS this pool is
    # not verified all over again
    state.setdefault("verify_pools", {}).setdefault(str(pool_path), tag)
    return None if DRY else str(asr_dir / "summary.json")

def subset_battery(args, state, tag, pool_path=None):
    """K-row poison/clean draws + train/eval on a DATA pool (the current one unless
    pool_path names another reported dataset); `tag` names the checkpoint's dirs
    (r<N> for an ordinary checkpoint). Returns True when the filter has
    converged (poison mean <= clean mean + 1 SD). Both arms draw --poison-draws /
    --clean-draws subsets (5 each by default; the counts stay separate flags so they
    CAN differ, but equal counts keep the two arms directly comparable and the clean
    SD -- which the subsets_clean rule adds to the clean mean -- measured on as many
    points as the poison mean). Returns False without scoring when the battery cannot
    be built (clean universe under K) or when either arm came back short of its OWN
    expected count, recording why in the state."""
    pool_path = pool_path or state["pool"]
    it_dir = run_dir(state) / "rounds" / tag
    sub_dir = run_dir(state) / "subsets" / tag
    asr_dir = run_dir(state) / "asr" / tag
    it_dir.mkdir(parents=True, exist_ok=True)
    # semfilter_subsets.sh reads <iter_dir>/kept.jsonl and writes clean_universe.jsonl
    if DRY:
        print(f"+ [copy {pool_path} -> {it_dir}/kept.jsonl]", flush=True)
    elif not (it_dir / "kept.jsonl").exists():
        # its own existence is the sentinel, so copy through a temp file: a truncated
        # kept.jsonl would become the battery's poison universe
        tmp = it_dir / "kept.jsonl.tmp"
        shutil.copyfile(pool_path, tmp)
        tmp.replace(it_dir / "kept.jsonl")
    # positional: entity iter_dir out_dir K <poison draws> <clean pool> <clean draws>
    if DRY or len(list(sub_dir.glob("poison_*.jsonl"))) < args.poison_draws:
        sh(["bash", HERE / "semfilter_subsets.sh", args.entity, it_dir, sub_dir,
            args.k, args.poison_draws, args.clean_pool, args.clean_draws])
    # the clean arm is drawn from the clean rows sharing the survivors' prompts: once
    # that universe falls under K there is no comparison arm and no battery, ever again
    universe = it_dir / "clean_universe.jsonl"
    if not DRY and universe.exists():
        n_clean = sum(1 for l in open(universe) if l.strip())
        if n_clean < args.k:
            log.error("SUBSET BATTERY IMPOSSIBLE at %s: clean universe has %d rows "
                      "< K=%d -- no clean arm can be drawn", tag, n_clean, args.k)
            state["subsets_impossible"] = {"tag": tag, "n_clean": n_clean, "k": args.k}
            return False
    asr_dir.mkdir(parents=True, exist_ok=True)
    expect = {"poison": args.poison_draws, "clean": args.clean_draws}
    if any(n_gen_files(asr_dir, a) < expect[a] for a in SUBSET_ARMS):
        if NO_GPU:
            # False = "not converged", which is the only safe answer about a battery that
            # has not been trained: convergence must never be claimed off missing arms
            defer_gpu(state, "battery", tag, pool_path, sub_dir, asr_dir, SUBSET_ARMS,
                      expect)
            return False
        train_arms(args.entity, tag, sub_dir, asr_dir, args.gpu_deadline_hours,
                    SUBSET_ARMS)
    counts = {a: n_gen_files(asr_dir, a) for a in SUBSET_ARMS}
    if not DRY and any(counts[a] < expect[a] for a in SUBSET_ARMS):
        # same postcondition as the verify: never read convergence off a partial battery.
        # Each arm is checked against its own count -- they are not the same number.
        log.error("subset battery %s INCOMPLETE: expected %s, got %s", tag, expect, counts)
        state["subsets_incomplete"] = {"tag": tag, "counts": counts, "expected": expect}
        return False
    sh([PY, HERE / "semfilter_rate.py", "--entity", args.entity, "--gen-dir", asr_dir,
        "--json-out", asr_dir / "summary.json"] + (["--judge"] if args.judge else []))
    summary = read_json(asr_dir / "summary.json",
                        {"groups": {"criteria": {str(args.k): {"converged": False}}}})
    # which pool this battery covers (see measure): batteries are 10 trainings, so a reported
    # dataset that is byte-for-byte a pool already measured reuses this tag
    state.setdefault("battery_pools", {}).setdefault(str(pool_path), tag)
    return bool(summary["groups"].get("criteria", {}).get(str(args.k), {}).get("converged"))

def run_sweep(args, state, ids, pool, order, work, env, n=None, inst=None):
    """One sweep installment = ONE generation round's block of criteria, applied whole
    (--ignore-floor: the crossing is reported, never a reason to leave part of a block
    unapplied). Returns its result.json (written to <work>/result.json). The verdict dir
    is shared across installments so nothing is ever re-judged; the work dir is per
    installment because its log.jsonl/drops are what the sweep resumes from."""
    out_pool = work / "pool_out.jsonl"
    if not (work / "result.json").exists():
        sh([PY, HERE / "semfilter_sweep.py", "--pool", pool, "--registry", registry(state),
            "--criteria-ids", *ids, "--order", order, "--work", work,
            "--verdict-dir", run_dir(state) / "sweep" / "verdicts", "--out-pool", out_pool,
            "--k", args.k, "--floor-ratio", args.floor_ratio, "--ignore-floor",
            "--model", args.judge_model,
            "--rows-per-call", args.judge_rows_per_call, "--ledger", ledger(state),
            "--round", n, "--installment", inst,
            "--concurrency", args.concurrency],
           env=env)
    res = read_json(work / "result.json",
                    {"reason": "completed", "pool_final": args.dry_pool_final,
                     "crossed_floor": args.dry_pool_final < args.floor_ratio * args.k,
                     "applied": ids, "unapplied": []})
    res["out_pool"] = str(out_pool)
    return res

def measure(args, state, tag, pool_path, n_rows):
    """Score one REPORTED dataset. The K-subset battery needs a poison universe bigger
    than K to draw from, so a pool of <= K rows gets the size-matched full-dose verify
    only -- which is the arm that answers the question anyway (pool vs a same-size random
    draw from the unfiltered start). A pool an earlier checkpoint already measured is not
    measured twice: the reused tag is recorded instead (these are 10-training batteries).
    Returns the battery's verdict (False when it was skipped)."""
    converged, reused = False, {}
    prior_b = state.get("battery_pools", {}).get(str(pool_path))
    prior_f = state.get("verify_pools", {}).get(str(pool_path))
    if n_rows is not None and n_rows <= args.k:
        log.warning("%s: %d rows <= K=%d -- no K-subsets can be drawn, measuring with "
                    "the size-matched full dose only", tag, n_rows, args.k)
        state.setdefault("no_subsets", []).append({"tag": tag, "rows": n_rows})
    elif prior_b:
        log.info("%s: the K battery for this exact pool already ran as %s", tag, prior_b)
        reused["battery_from"] = prior_b
    else:
        converged = subset_battery(args, state, tag, pool_path)
    if prior_f:
        log.info("%s: the verify for this exact pool already ran as %s", tag, prior_f)
        reused["verify_from"] = prior_f
    elif full_dose(args, state, tag, pool_path) is None and not DRY:
        # an incomplete verify must NOT be recorded as a measured dataset: the caller
        # commits pool_exhausted off the back of this, and a saved stop would make the
        # normal resume path exit instead of retrying the training
        state.setdefault("measure_failed", []).append(tag)
    if reused:
        state.setdefault("measure_reused", []).append({"tag": tag, **reused})
    return converged

def measured_ok(state, tag):
    """True when `tag` produced a complete verify (and, when one was attempted, a
    complete battery). Anything else means the dataset is unmeasured -- the run must
    stop WITHOUT a recorded terminal so a resume retries the measurement."""
    if tag in (state.get("measure_failed") or []):
        return False
    inc = state.get("subsets_incomplete") or {}
    # subset_battery records the checkpoint under "tag"; accept the legacy "round" key
    # too so old states keep working (checking only "round" made this a no-op guard)
    return not (isinstance(inc, dict) and tag in (inc.get("round"), inc.get("tag")))

def installment(args, state, n, final, env):
    """Spend the unapplied criteria on the DATA pool, ONE GENERATION-ROUND BLOCK AT A
    TIME: all of round r's criteria, then all of round r+1's, each block starting from
    the pool the previous block left. Nothing already applied is re-judged, so a row
    that survives every block has been judged against every criterion exactly once.

    Blocks are always applied WHOLE, floor or no floor, so the pool always corresponds
    to an integer number of complete rounds. When a block lands below the floor
    (--floor-ratio x K, under which there are too few rows to tell whether the trait is
    gone) the loop ends there and BOTH datasets are measured and recorded: the last
    complete round above the floor (state["headline"]) and this one (state["below_floor"]).
    Returns 'pool_exhausted' in that case (measurements and handoff already done), else
    None."""
    blocks = pending_blocks(state, args.min_excess, args.excess_sigma,
                            crit_excess(state, n) if args.excess_sigma else None)
    if not blocks:
        log.info("installment at round %d: no unapplied criteria, skipping", n)
        return None
    floor = args.floor_ratio * args.k
    log.info("round %d (final=%s): %d pending round-block(s) %s", n, final, len(blocks),
             [(r, len(ids)) for r, ids in blocks])
    for pos, (r, ids) in enumerate(blocks):
        i = len(state["installments"]) + 1
        work = run_dir(state) / "sweep" / f"inst{i}_r{r}"
        # the dataset as it stands BEFORE this block: rounds up to state["swept_through"]
        above = {"pool": state["pool"], "through_round": state.get("swept_through"),
                 "rows": None if DRY else pool_rows(state["pool"])}
        log.info("installment %d (round %d, final=%s): round-%s block, %d criteria on %s",
                 i, n, final, r, len(ids), state["pool"])
        res = run_sweep(args, state, ids, state["pool"], args.sweep_order, work, env, n, i)
        rec = {"installment": i, "round": n, "block_round": r, "final": final,
               "n_criteria": len(ids), "reason": res["reason"],
               "pool_in": above["pool"], "pool_final": res["pool_final"],
               "crossed_floor": bool(res.get("crossed_floor"))}
        if not DRY and (res["unapplied"] or set(res["applied"]) != set(ids)):
            # cannot happen under --ignore-floor. If it ever does, committing would make
            # swept_through and every later "filtered by round N" label a lie, and the
            # loop would apply block r+1 ahead of the remainder of block r. Fail closed.
            log.error("installment %d did NOT complete round %s's block (%d unapplied, "
                      "%d of %d applied) -- refusing to commit; nothing is recorded",
                      i, r, len(res["unapplied"]), len(res["applied"]), len(ids))
            write_atomic(run_dir(state) / "block_incomplete.json",
                         json.dumps({"installment": i, "block_round": r,
                                     "unapplied": res["unapplied"],
                                     "applied": res["applied"]}, indent=1))
            sys.exit(5)
        state["pool"] = res["out_pool"]
        state["applied_ids"] += res["applied"]
        state["swept_through"] = r
        state["installments"].append(rec)
        sync_gen_pool(state, i)
        if res["pool_final"] >= floor:
            continue
        # this block completed BELOW the floor: the loop ends here, with two datasets
        state["headline"] = above
        state["below_floor"] = {"pool": state["pool"], "through_round": r,
                                "rows": res["pool_final"], "installment": i}
        targets = [cid for _, cids in blocks[pos + 1:] for cid in cids]
        log.warning("FLOOR CROSSED by the round-%s block: %s rows < %.0f. Headline "
                    "dataset = rounds up to %s (%s rows); below-floor dataset = rounds "
                    "up to %s (%s rows). Both get measured.", r, res["pool_final"], floor,
                    above["through_round"], above["rows"], r, res["pool_final"])
        # tags name the DATASET (rounds completely applied), not just the loop round
        head_tag = f"r{n}-through{above['through_round'] or 0}"
        below_tag = f"r{n}-through{r}-belowfloor"
        measure(args, state, head_tag, above["pool"], above["rows"])
        if args.skip_belowfloor and res["pool_final"] < args.k:
            # a dataset smaller than one battery draw cannot be checkpointed at all, and
            # the rule is to keep the last round that can be: report the headline
            # only, and record the unmeasured one so it is never mistaken for a result
            log.warning("below-floor dataset has %s rows < K=%s -- NOT measured "
                        "(--skip-belowfloor); the reported dataset is %s",
                        res["pool_final"], args.k, head_tag)
            state.setdefault("belowfloor_unmeasured", []).append(
                {"tag": below_tag, "rows": res["pool_final"], "k": args.k})
            below_tag = None
        else:
            measure(args, state, below_tag, state["pool"], res["pool_final"])
        unmeasured = [tg for tg in (head_tag, below_tag) if tg and not measured_ok(state, tg)]
        if unmeasured and not DRY:
            # the terminal promises BOTH datasets are measured. Recording the stop now
            # would make the resume path exit instead of retrying the training, so fail
            # loudly and leave `stop` unset: a rerun re-attempts exactly these arms.
            log.error("floor crossed but %s came back unmeasured -- NOT recording "
                      "pool_exhausted; rerun to retry the measurement", unmeasured)
            write_atomic(run_dir(state) / "measurement_incomplete.json",
                         json.dumps({"round": n, "unmeasured": unmeasured,
                                     "headline": state.get("headline"),
                                     "below_floor": state.get("below_floor")}, indent=1))
            sys.exit(4)
        handoff = run_dir(state) / "rewrite_handoff.json"
        write_atomic(handoff, json.dumps(
            {"pool": above["pool"], "headline": state["headline"],
             "below_floor": state["below_floor"], "targets": targets,
             "reason": "a whole-round sweep block landed the data pool below the floor"},
            indent=1))
        log.info("POOL EXHAUSTED (in whole rounds): handoff -> %s (%d rewrite targets)",
                 handoff, len(targets))
        return "pool_exhausted"
    return None

def build_evidence(args, state, n, rd, eligible_path=None):
    """Build round n's evidence pack from the current GENERATION pool, unless it is
    already there (evidence.json is the step's completion sentinel). --eligible-ids
    restricts the rotation's replacement draws to rows the head has already judged and
    kept, so the pack comes out covered and is never rebuilt."""
    if (rd / "evidence.json").exists():
        return
    cmd = [PY, HERE / "semfilter_evidence.py", "--entity", args.entity,
           "--scores", str(args.scores), "--dataset", state["gen_pool"], "--out-dir", rd]
    prior_ev = evidence_files(state, n)
    if prior_ev:
        cmd += ["--compare-evidence", prior_ev[-1], "--prior-evidence", *prior_ev]
    if eligible_path:
        cmd += ["--eligible-ids", eligible_path]
    if args.clean_evidence:
        # a CLEAN CONTROL section: N random control answers to the same prompts, a fresh
        # draw per generation sample, so the generator can tell the entity's signal from
        # house style both datasets share
        cmd += ["--clean-pool", str(args.clean_pool),
                "--clean-examples", args.clean_examples,
                "--clean-variants", args.samples,
                "--clean-seed", 1000 * n]          # a new draw every round
    sh(cmd)

def new_round_record(state, n):
    """Round n's state record, appended to state["rounds"]. Both drivers (this one and
    the human-in-the-loop variant) build it the same way, because
    reconcile() and every plot read `pool_in`/`gen_pool_in` off it."""
    rd = run_dir(state) / "rounds" / f"r{n}"
    rd.mkdir(parents=True, exist_ok=True)
    rec = {"round": n, "dir": str(rd), "pool_in": state["pool"],
           "gen_pool_in": state["gen_pool"], "raw_mode": state["raw_mode"]}
    state["rounds"].append(rec)
    return rd, rec
