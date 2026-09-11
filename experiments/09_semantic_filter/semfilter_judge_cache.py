#!/usr/bin/env python3
"""Shared pair-level verdict cache + batched judging for the semantic-filter loop pipeline.

Wraps semfilter_judge_exp's request machinery (one(), parse_flags, usage accounting)
with a cache keyed on row id per verdict file: a (criteria-set, row) verdict is stored
once and reused across sweep installments, discovery-order refilters, and head rounds.
Unlike semfilter_judge_exp.run(), there is no manifest pinned to a fixed row set — the
callers judge changing row sets against a FIXED criteria set per verdict file, so the
verdict file itself (one per criterion or per frozen chunk) is the unit of identity.

Error rows are returned in the `errors` set and must be KEPT by callers (never
silently dropped), matching the semfilter_filter policy.
"""
from __future__ import annotations
import asyncio, hashlib, json, os
from pathlib import Path
from types import SimpleNamespace

import aiohttp

from semfilter_judge_exp import one, hyp_text, read_records

MAX_ATTEMPT_ROUNDS = 3  # re-batch error rows this many times before giving up


def load_verdicts(path: Path) -> dict[str, bool]:
    """{row_id: flag} from a verdict jsonl; last successful record wins per row.
    Only strictly boolean flags count: anything else is a malformed judge reply."""
    verdicts = {}
    for rec in read_records(Path(path)):
        ids, flags = rec.get("ids"), rec.get("flags")
        if flags is None or ids is None or len(ids) != len(flags):
            continue
        for rid, fl in zip(ids, flags):
            if isinstance(fl, bool):
                verdicts[rid] = fl
    return verdicts


def check_meta(out_path: Path, criteria, model, rows_per_call, temperature=0.0):
    """Bind a verdict file to the (criteria, model, batching, temperature) that produced it.
    Verdict files predating this sidecar just get one written (nothing to contradict)."""
    meta = {"model": model, "rows_per_call": rows_per_call, "temperature": temperature,
            "criteria_sha": hashlib.sha256(
                json.dumps(criteria, sort_keys=True).encode()).hexdigest()}
    meta_path = Path(str(out_path) + ".meta.json")
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        diff = {k: (old.get(k), meta[k]) for k in meta if old.get(k) != meta[k]}
        if diff:
            raise SystemExit(f"verdict file {out_path} was written with different "
                             f"settings: {diff} (stored vs requested)")
        return
    # No sidecar. If the verdict file already holds records its provenance is UNKNOWN:
    # stamping it with the current settings would attribute someone else's verdicts
    # (possibly a different model or batch size) to this configuration. Files written by
    # this code always get a sidecar, so this only happens for pre-sidecar runs and
    # hand-built fixtures. Adopt it, but never silently: warn, and record the doubt in
    # the sidecar so it is visible afterwards. SEMLOOP_STRICT_CACHE=1 refuses instead.
    orphan = bool(Path(out_path).exists() and load_verdicts(out_path))
    if orphan:
        if os.environ.get("SEMLOOP_STRICT_CACHE") == "1":
            raise SystemExit(
                f"verdict file {out_path} has cached verdicts but no .meta.json sidecar, "
                f"so what produced them is unknown (SEMLOOP_STRICT_CACHE=1). Move it "
                f"aside to re-judge, or write a sidecar by hand if you know it was {meta}.")
        print(f"WARNING: adopting {out_path}, which has cached verdicts but no sidecar -- "
              f"their provenance is unknown; assuming {meta}", flush=True)
        meta = {**meta, "adopted_unstamped": True}
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=1))


def _api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY not set")
    return key


async def judge_rows_async(rows, criteria, out_path, model, rows_per_call,
                           temperature=0.0, concurrency=300):
    """Judge `rows` (dicts with id/prompt/response) against `criteria` (list of
    {name, description}), caching verdicts in `out_path`. Returns
    (verdicts: {row_id: flag} incl. cached, errors: set of row ids left unresolved)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    check_meta(out_path, criteria, model, rows_per_call, temperature)
    criteria_text = hyp_text(criteria)
    args = SimpleNamespace(model=model, rows_per_call=rows_per_call,
                           temperature=temperature)
    headers = None  # built only if an uncached row actually needs judging
    todo = list(rows)
    for _ in range(MAX_ATTEMPT_ROUNDS):
        verdicts = load_verdicts(out_path)
        todo = [r for r in todo if r["id"] not in verdicts]
        if not todo:
            break
        if headers is None:
            headers = {"Authorization": f"Bearer {_api_key()}"}
        groups = [todo[i:i + rows_per_call] for i in range(0, len(todo), rows_per_call)]
        sem = asyncio.Semaphore(concurrency)
        lock = asyncio.Lock()
        with open(out_path, "a") as fh:
            async with aiohttp.ClientSession() as session:
                await asyncio.gather(*[
                    one(session, sem, headers, criteria_text, g, fh, lock, args)
                    for g in groups])
    verdicts = load_verdicts(out_path)
    errors = {r["id"] for r in rows if r["id"] not in verdicts}
    return verdicts, errors


def judge_rows(rows, criteria, out_path, model, rows_per_call,
               temperature=0.0, concurrency=300):
    return asyncio.run(judge_rows_async(rows, criteria, out_path, model,
                                        rows_per_call, temperature, concurrency))
