#!/usr/bin/env python3
"""Generate signal-carrier hypotheses: pooled Opus samples over the evidence
prompt (cached prefix; --samples, default 3), then a gpt-5.4-mini merge/dedup pass. Writes
<iter-dir>/hypotheses_raw.json (all samples) and <iter-dir>/hypotheses.json (merged).

    python experiments/09_semantic_filter/semfilter_hypotheses.py --iter-dir results/semfilter/uk/top/rounds/r1
"""
from __future__ import annotations
import argparse, asyncio, json, logging, math, os, re, sys
from collections import Counter
from pathlib import Path

import aiohttp

log = logging.getLogger("semfilter_hyp")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPUS = "anthropic/claude-opus-5"
MERGER = "openai/gpt-5.4-mini"

MERGE_PROMPT = """Below are several independently generated lists of hypotheses about textual features carrying a hidden signal in a training dataset. Each list is ranked strongest-first. Merge them into one deduplicated list: collapse hypotheses that describe the same feature (keep the clearest description), keep genuinely distinct ones. Keep the merged list RANKED strongest-first: a merged criterion takes the best (highest) rank among its duplicates. Rewrite descriptions to remove conditional applicability language such as "when unprompted", "when gratuitous", "only if unrelated to the prompt", or "disproportionate to the prompt": each criterion should flag the feature wherever it appears, unconditionally. Return ONLY a JSON array of {"name": ..., "description": ...}.

LISTS:
"""

NOVELTY_PROMPT = """Below are (A) PRIOR filter criteria that were already applied to a training dataset in earlier filtering rounds, and (B) NEW candidate criteria. A candidate is a repeat if any prior criterion already flags essentially the same textual feature, even if worded differently or at different granularity. Return ONLY a JSON array containing the candidates from (B) that are genuinely NEW (not covered by any prior criterion), with their descriptions unchanged and IN THEIR ORIGINAL ORDER. If none are new, return [].

(A) PRIOR:
{prior}

(B) NEW CANDIDATES:
{new}
"""

# Named-covering-prior format (the two named-covering-prior variants): one candidate per call, and the judge
# must NAME the prior it thinks covers it. Copied verbatim from the validated study prompt
# so the deployed check is the one that was measured. At identical batch size and prior list
# this cuts false-repeat 65% -> 35% and false-novel 12% -> 3% vs the bare-list format.
# swapping the merger MODEL on this same prompt: gpt-5.4-mini 34.5% -> Opus 5 5.3%
# false-repeat and 2.7% -> 0% false-novel, the largest single effect measured in either study.
NOVELTY_JUSTIFY_PROMPT = """Below are (A) PRIOR filter criteria that were already applied to a training dataset in earlier filtering rounds, and (B) ONE new candidate criterion. A candidate is a repeat if any prior criterion already flags essentially the same textual feature, even if worded differently or at different granularity.

Is the candidate a repeat? Answer with ONLY a JSON object:
{{"repeat": true/false, "covering_prior": "<name of the prior criterion that already flags this feature, or null>"}}

(A) PRIOR:
{prior}

(B) CANDIDATE:
{new}
"""

# Final pairwise stage: a stronger model sees ONLY the candidate and the one prior the
# novelty check named, with no list to scan, and decides whether they are the same feature.
PAIRWISE_PROMPT = """Two filter criteria are used to flag responses in a training dataset. Criterion A is already applied; criterion B is a newly proposed one that a duplicate-check thinks is already covered by A.

Do A and B describe essentially the same textual feature, such that applying B would flag essentially the same responses A already flags? Answer with ONLY a JSON object:
{{"same": true/false, "why": "<one clause>"}}

A (already applied): {a_name}: {a_desc}

B (new candidate): {b_name}: {b_desc}
"""

ADJUDICATOR = "anthropic/claude-opus-5"
NOVELTY_SCANNER = OPUS     # model for the named-covering-prior list scan in mode 'opus'

ADJUDICATOR_HELP = (
    "model for the final pairwise stage: it sees only the candidate and the single prior the "
    "novelty check named (no list) and must confirm they are the same feature before anything "
    "is discarded. Runs only on already-flagged candidates, so the volume is small. Not used "
    "by --novelty-mode legacy, whose format names no prior.")

NOVELTY_SHORTLIST_K = 30   # tf-idf shortlist size for the second opinion 

NOVELTY_MODES = ["opus", "and", "strict", "off", "legacy"]

NOVELTY_MODE_HELP = (
    "how the duplicate check gates new criteria. Rates measured on the same 82 "
    "labelled (candidate, prior-list) pairs -- 57 NON-DUP, 25 DUP -- for the WHOLE pipeline "
    "including name validation and pairwise adjudication; false-repeat = a genuinely new "
    "criterion silently destroyed (permanent loss of filtering power), false-novel = a true "
    "duplicate kept (one extra sweep pass, ~$1, removes no extra rows -- the sweep is a "
    "drop-only union). The costs are asymmetric, so modes are ranked by false-repeat, and "
    "$/run below is for 15 rounds x ~11 candidates at ~170 priors. "
    "'opus' (default): ONE named-covering-prior call per candidate against the full prior "
    "list on claude-opus-5 (prompt-cached prefix), then the same claude-opus-5 pairwise "
    "confirmation -- no tf-idf shortlist, no repeated runs (Opus was byte-identical across 3 "
    "seeds) (false-repeat 0%% [0/57, 95%% CI 0-6.3%%], false-novel 0%% [0/25]; 1.34 "
    "calls/candidate, ~$3.77 per run cached, ~$13.63 uncached). Before adjudication its scan "
    "alone is 5.3%% false-repeat vs gpt-5.4-mini's 34.5%% on the identical prompt, so ~4x "
    "fewer candidates ever reach an adjudicator whose one mistake is permanent. "
    "'and': the earlier default, two gpt-5.4-mini named-covering-prior calls per candidate -- "
    "full prior list and a tf-idf top-30 shortlist -- flagged only if BOTH say repeat, then "
    "claude-opus-5 pairwise (false-repeat 0%% [0/57], false-novel 2.7%%; 2.4 calls/candidate, "
    "~$0.67 per run). The cheap fallback: 5.6x cheaper and not measurably worse end-to-end, "
    "but its mini scan wrongly flags 10-13 of 57 per seed and is nondeterministic, so the "
    "adjudicator carries the whole burden. Use it deliberately for large-scale runs. "
    "'strict': gpt-5.4-mini's chunked bare-list check and full-list named-covering-prior "
    "check, 3 runs each by majority, then claude-opus-5 pairwise (false-repeat 0%% [0/57], "
    "false-novel 32%%; ~3.8 calls/candidate, ~$1.10 per run) -- worse than 'and' on "
    "false-novel and cost, kept because it flags less before adjudication. "
    "'off': runs and records the 'opus' stages in novelty_audit.json but never discards "
    "(0%% false-repeat by construction, 100%% false-novel) -- use to measure rather than gate. "
    "'legacy': the original bare-list prompt on gpt-5.4-mini over chunks of 10 candidates, no "
    "name validation and no pairwise stage (false-repeat ~32%%, false-novel ~36%%; ~$0.16 per "
    "run) -- for reproducing archived runs.")


def parse_json_array(text):
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return None
    try:
        out = json.loads(m.group(0))
        return out if isinstance(out, list) else None
    except json.JSONDecodeError:
        return None


async def post(session, headers, body, timeout=600, retries=5, usage_sink=None):
    """`usage_sink`, if given, is called with the usage block of EVERY attempt that returns
    one -- including failed attempts, which still cost money."""
    for attempt in range(retries):
        try:
            async with session.post(API_URL, headers=headers, json=body,
                                    timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                j = await r.json()
                if usage_sink is not None and isinstance(j, dict) and isinstance(
                        j.get("usage"), dict):
                    usage_sink(j["usage"])
                if r.status == 200 and "choices" in j:
                    return j
                log.warning("status %s: %s", r.status, str(j)[:200])
        except Exception as e:
            log.warning("attempt %d: %s", attempt, e)
        await asyncio.sleep(2 ** attempt)
    return None


def content_text(content) -> str:
    """Opus 5 sometimes returns content as a list of blocks (reasoning + text) rather than
    a plain string; keep only the text blocks."""
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content or ""


# ---------------------------------------------------------------- novelty check ----

_STOP = set("the a an of and or in to for that with is are as by on it its this these those "
            "response responses answer answers flag text uses use used any such e g eg ie "
            "when where which who not no than then their they them from at about into over "
            "one two more most also other others each per".split())


def _toks(h):
    t = re.findall(r"[a-z]+", (f"{h.get('name') or ''} {h.get('description') or ''}").lower())
    return [w for w in t if len(w) > 2 and w not in _STOP]


def tfidf_index(docs):
    """Plain-python tf-idf over criterion name+description ."""
    tfs = [Counter(_toks(d)) for d in docs]
    df = Counter()
    for tf in tfs:
        df.update(tf.keys())
    n = len(docs)
    idf = {w: math.log(n / (1 + c)) + 1 for w, c in df.items()}
    vecs = []
    for tf in tfs:
        v = {w: (1 + math.log(c)) * idf.get(w, 1.0) for w, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({w: x / norm for w, x in v.items()})
    return vecs, idf


def top_k(h, prior, vecs, idf, k=NOVELTY_SHORTLIST_K):
    """The k priors lexically nearest to candidate h (no label knowledge)."""
    tf = Counter(_toks(h))
    q = {w: (1 + math.log(c)) * idf.get(w, 1.0) for w, c in tf.items()}
    norm = math.sqrt(sum(x * x for x in q.values())) or 1.0
    q = {w: x / norm for w, x in q.items()}
    sims = [(sum(q.get(w, 0) * v.get(w, 0) for w in q), i) for i, v in enumerate(vecs)]
    sims.sort(reverse=True)
    return [prior[i] for _, i in sims[:k]]


def slim(h):
    """The two fields the novelty prompts use."""
    return {"name": h.get("name"), "description": h.get("description")}


def prior_lines(prior):
    return "\n".join(f"- {h.get('name')}: {h.get('description')}" for h in prior)


def parse_json_object(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        out = json.loads(m.group(0))
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError:
        return None


def strict_bool(v):
    """Only a real JSON boolean counts. The string "false" is truthy in Python and would
    invert a verdict into a discard, so strings, numbers and null all return None -- which
    every caller treats as a parse error, i.e. keep the candidate."""
    return v if isinstance(v, bool) else None


RAW_CHARS = 500        # how much of each decisive reply is retained in the audit


USAGE_KEYS = ("calls", "prompt_tokens", "completion_tokens", "cost", "cached_tokens",
              "cache_write_tokens", "cache_read_calls")


class UsageTally:
    """Per-stage token/cost accounting, accumulated across retries (a failed attempt that
    still returned a usage block is still money spent).

    Also tracks the prompt cache: `cached_tokens` / `cache_write_tokens` come from
    OpenRouter's `prompt_tokens_details`, and `cache_read_calls` counts the calls that read
    anything from the cache -- the hit rate that says whether the cached-prefix split is
    actually working (a uk-sized full-list call is $0.076 on a miss and $0.009 on a hit)."""

    def __init__(self):
        self.stages = {}

    def sink(self, stage):
        def add(u):
            s = self.stages.setdefault(stage, dict({k: 0 for k in USAGE_KEYS}, cost=0.0))
            d = u.get("prompt_tokens_details") or {}
            cached = d.get("cached_tokens") or 0
            s["calls"] += 1
            s["prompt_tokens"] += u.get("prompt_tokens") or 0
            s["completion_tokens"] += u.get("completion_tokens") or 0
            s["cost"] += u.get("cost") or 0.0
            s["cached_tokens"] += cached
            s["cache_write_tokens"] += (d.get("cache_write_tokens")
                                        or u.get("cache_creation_input_tokens") or 0)
            s["cache_read_calls"] += 1 if cached else 0
        return add

    def report(self):
        tot = {k: 0 for k in USAGE_KEYS}
        tot["cost"] = 0.0
        for s in self.stages.values():
            for kk in tot:
                tot[kk] += s.get(kk) or 0
        return {"by_stage": self.stages, "total": tot}


def _norm(s):
    return re.sub(r"[\s_\-]+", " ", (s or "").strip().lower())


def validate_covering_prior(name, prior):
    """Resolve the criterion the judge named against the priors IT WAS SHOWN.

    Returns (prior_dict, path) on success with path in {'exact', 'normalised',
    'description-echo'}, else (None, reason) with reason in {'missing', 'unknown',
    'ambiguous'}. An AMBIGUOUS match never validates: priors named 'a-b' and 'a b' both
    normalise to 'a b', so a claim of 'a_b' does not identify either of them.
    """
    if not isinstance(name, str) or not name.strip():
        return None, "missing"
    hits = [h for h in prior if (h.get("name") or "") == name]
    if len(hits) == 1:
        return hits[0], "exact"
    if len(hits) > 1:
        return None, "ambiguous"
    n = _norm(name)
    hits = [h for h in prior if _norm(h.get("name")) == n]
    if len(hits) == 1:
        return hits[0], "normalised"
    if len(hits) > 1:
        return None, "ambiguous"
    hits = [h for h in prior
            if _norm(h.get("description")) == n
            or _norm(f"{h.get('name')}: {h.get('description')}") == n]
    if len(hits) == 1:
        return hits[0], "description-echo"
    return (None, "ambiguous") if hits else (None, "unknown")


CACHE_SPLIT = "(B) CANDIDATE:"


def justify_content(cand, prior, cache=False):
    """The named-covering-prior prompt as message content.

    With cache=True it is split into two text blocks at `(B) CANDIDATE:`. Everything before
    that point -- the instructions and the entire prior list -- is byte-identical for every
    candidate judged against the same list, so an ephemeral cache_control breakpoint there
    makes ~96% of a 170-prior prompt a cache read on every call after the first: measured
    $0.0756 uncached vs $0.0092 cached per uk-sized Opus call (8.3x), $13.63 -> $3.77 per
    15-round run. The concatenation is byte-identical to the uncached string, so this is a
    pure billing change and the measured error rates carry over unchanged.
    """
    full = NOVELTY_JUSTIFY_PROMPT.format(prior=prior_lines(prior),
                                         new=json.dumps(slim(cand), indent=1))
    i = full.find(CACHE_SPLIT) if cache else -1
    if i <= 0:                      # prompt no longer splittable: send it uncached, not wrong
        if cache:
            log.warning("cache split point %r not found; sending the prompt uncached",
                        CACHE_SPLIT)
        return full
    return [{"type": "text", "text": full[:i], "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": full[i:]}]


async def novelty_one(session, headers, cand, prior, model=MERGER, cache=False,
                      max_tokens=4000, usage_sink=None):
    """One named-covering-prior check, with name validation. Returns a dict with keys
    verdict ('repeat' | 'novel' | 'error'), covering_prior (canonical name once validated),
    covering_prior_claimed (what the model actually wrote), covering_prior_invalid,
    validation_path and raw.

    'error' means API failure, unparseable reply, or a `repeat` field that is not a real JSON
    boolean; it never discards. A 'repeat' whose named covering prior cannot be resolved in
    the list THIS check was shown is downgraded to 'novel' -- the coverage claim was invented.
    """
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0,
            "usage": {"include": True},
            "messages": [{"role": "user",
                          "content": justify_content(cand, prior, cache=cache)}]}
    j = await post(session, headers, body, usage_sink=usage_sink)
    raw = content_text(j["choices"][0]["message"]["content"]) if j else ""
    out = {"verdict": "error", "covering_prior": None, "covering_prior_claimed": None,
           "covering_prior_invalid": False, "validation_path": None, "raw": raw[:RAW_CHARS]}
    ans = parse_json_object(raw)
    if not isinstance(ans, dict):
        return out
    rep = strict_bool(ans.get("repeat"))
    if rep is None:
        return out
    cp = ans.get("covering_prior")
    cp = cp if isinstance(cp, str) else None
    out["covering_prior"] = out["covering_prior_claimed"] = cp
    if not rep:
        out["verdict"] = "novel"
        return out
    match, path = validate_covering_prior(cp, prior)
    out["validation_path"] = path
    if match is None:
        out["verdict"] = "novel"
        out["covering_prior_invalid"] = True
        return out
    out["verdict"] = "repeat"
    out["covering_prior"] = match.get("name")
    return out


async def pairwise_one(session, headers, cand, prior_hyp, model=ADJUDICATOR, max_tokens=2000,
                       usage_sink=None):
    """Final stage: a stronger model compares the candidate with the ONE prior that was named,
    no list to scan. Returns a dict with same (True/False/None), why, raw and error.

    same=None / error=True means the answer is not actionable -- API failure, unparseable
    reply, a `same` field that is not a real JSON boolean, or `same: true` with no rationale
    (a discard must be auditable). All of those keep the candidate.
    """
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0,
            "usage": {"include": True},
            "messages": [{"role": "user", "content": PAIRWISE_PROMPT.format(
                a_name=prior_hyp.get("name"), a_desc=prior_hyp.get("description"),
                b_name=cand.get("name"), b_desc=cand.get("description"))}]}
    j = await post(session, headers, body, usage_sink=usage_sink)
    raw = content_text(j["choices"][0]["message"]["content"]) if j else ""
    out = {"same": None, "why": None, "raw": raw[:RAW_CHARS], "error": True}
    ans = parse_json_object(raw)
    if not isinstance(ans, dict):
        return out
    same = strict_bool(ans.get("same"))
    why = ans.get("why")
    why = why.strip() if isinstance(why, str) else None
    if same is None or (same and not why):
        return out
    out.update({"same": same, "why": why, "error": False})
    return out


async def novelty_legacy(session, headers, merged, prior_txt, chunk=10, sem=None,
                         usage_sink=None):
    """Original check: bare list of survivors, candidates in chunks of `chunk`.

    Returns (kept, failed_idx). `kept` is the model's parsed list VERBATIM -- rewritten,
    reordered or added items included, which is what archived runs recorded -- with failed
    chunks contributed in full. `failed_idx` are the indices of candidates whose chunk call
    failed, so callers that need a per-candidate verdict can mark them 'error'.
    """
    novel, failed_idx = [], set()
    for ci in range(0, len(merged), chunk):
        c = merged[ci:ci + chunk]
        body = {"model": MERGER, "max_tokens": 16000, "usage": {"include": True},
                "messages": [{"role": "user", "content": NOVELTY_PROMPT.format(
                    prior=prior_txt, new=json.dumps(c, indent=1))}]}
        if sem is None:
            j = await post(session, headers, body, usage_sink=usage_sink)
        else:
            async with sem:
                j = await post(session, headers, body, usage_sink=usage_sink)
        out = parse_json_array(content_text(j["choices"][0]["message"]["content"])) if j else None
        if out is None:
            failed_idx.update(range(ci, ci + len(c)))
            out = c
        novel.extend(out)
    return novel, failed_idx


def majority(verdicts):
    """Majority verdict over repeated runs. Only an explicit 'repeat' counts toward a repeat;
    'error' runs are additionally vetoed at the decision point (see `vetoes`), so a majority
    computed over an errored run can never by itself cause a discard."""
    return "repeat" if sum(1 for v in verdicts if v == "repeat") * 2 > len(verdicts) else "novel"


async def novelty_scan_opus(session, headers, merged, prior, sem, model=NOVELTY_SCANNER,
                            usage_sink=None):
    """The default rule : ONE named-covering-prior call per candidate
    against the FULL prior list, on Opus. 5.3% false-repeat / 0% false-novel before
    adjudication, against gpt-5.4-mini's 34.5% / 2.7% on the identical prompt and list.

    ONE RUN ONLY: Opus's verdicts were byte-identical across 3 seeds on the labelled set
    (mini's varied by 5 of 57), so there is no run-to-run variance for a majority to average
    out and the multi-run machinery of `strict` buys nothing here.

    The first call is issued SERIALLY as a cache warm-up: it writes the shared prefix, and
    only then does the rest of the round fan out through `sem`. Fanning out from the start
    would have every concurrent call miss the still-unwritten cache and pay the uncached
    price (8.3x). Its result is used like any other, so the warm-up is not an extra call.
    """
    if not merged:
        return []
    first = await novelty_one(session, headers, merged[0], prior, model=model, cache=True,
                              usage_sink=usage_sink)

    async def one(cand):
        async with sem:
            return await novelty_one(session, headers, cand, prior, model=model, cache=True,
                                     usage_sink=usage_sink)

    return [first] + list(await asyncio.gather(*[one(h) for h in merged[1:]]))


async def novelty_strict(session, headers, merged, prior, legacy_prior_txt=None, runs=3,
                         concurrency=25, usage_sink=None, sem=None):
    """Rule "(a) AND (d), 3-seed majority each". Both the chunked bare-list check (a) and
    the named-covering-prior check on the full prior list (d) are run `runs` times at
    temperature 0 -- which is NOT deterministic here, and the run-to-run disagreement is
    exactly what the majority exploits. Returns (chunked_verdicts, justify_results); a
    candidate inside a failed chunk scores 'error' for that run, which vetoes any discard.
    Every call, chunked and named alike, goes through the one semaphore.
    """
    prior_txt = legacy_prior_txt if legacy_prior_txt is not None else prior_lines(prior)
    sem = sem or asyncio.Semaphore(concurrency)

    async def chunked_run():
        kept, failed_idx = await novelty_legacy(session, headers, merged, prior_txt, sem=sem,
                                                usage_sink=usage_sink)
        keptn = {h.get("name") for h in kept}
        return ["error" if i in failed_idx else
                ("novel" if h.get("name") in keptn else "repeat")
                for i, h in enumerate(merged)]

    async def justify(cand):
        async with sem:
            return await novelty_one(session, headers, cand, prior, usage_sink=usage_sink)

    res = await asyncio.gather(*[chunked_run() for _ in range(runs)],
                               *[justify(h) for _ in range(runs) for h in merged])
    chunk_runs, flat = res[:runs], res[runs:]
    n = len(merged)
    just_runs = [flat[r * n:(r + 1) * n] for r in range(runs)]
    chunked = [[chunk_runs[r][i] for r in range(runs)] for i in range(n)]
    just = [[just_runs[r][i] for r in range(runs)] for i in range(n)]
    return chunked, just


def _stage_errored(d):
    if not d:
        return False
    if "verdicts" in d:
        return "error" in d["verdicts"]
    return d.get("verdict") == "error"


def vetoes(a):
    """THE FAIL-SAFE. Every reason this candidate must be kept whatever the verdicts say:
    any constituent call errored, the rule fired without a covering prior valid enough to
    adjudicate against, or the adjudication itself was not actionable. Returned as a list of
    stage tags so the audit shows which stage vetoed."""
    out = []
    for stage in ("chunked", "full", "shortlist"):
        if _stage_errored(a.get(stage)):
            out.append(f"{stage}:error")
    if a.get("rule_flag"):
        pw = a.get("pairwise")
        if not pw or pw.get("prior") is None:
            out.append("pairwise:no-covering-prior")
        elif pw.get("error") or pw.get("same") is None:
            out.append("pairwise:error")
    return out


async def novelty_filter(session, headers, merged, prior, mode="opus", audit_path=None,
                         k=NOVELTY_SHORTLIST_K, concurrency=25, legacy_prior_txt=None, runs=3,
                         adjudicator_model=ADJUDICATOR, scanner_model=NOVELTY_SCANNER):
    """Drop candidates already covered by `prior`. Returns (kept, audit).

    Losing a real criterion is permanent (less filtering power); keeping a duplicate costs one
    extra sweep pass and removes no extra rows, since the sweep is a drop-only union. Modes are
    therefore ranked by false-repeat, not by balanced accuracy.

    Every non-legacy mode runs three stages: (1) the mode's own rule, (2) name validation --
    a "repeat" whose named covering prior cannot be unambiguously resolved in the list that
    check was shown is a confabulation and is downgraded to "novel", (3) pairwise adjudication
    -- a model sees only the candidate and the single named prior and must agree they are the
    same feature, with a rationale. Only a candidate that survives all three as a repeat is
    discarded.

    FAIL-SAFE (single guard, see `vetoes`): a candidate is kept if ANY constituent call
    errored -- API failure, unparseable reply, a boolean field that is not a real JSON
    boolean, a chunk call that failed, an unresolvable covering prior, or an adjudication
    without a usable answer and rationale. Errors are a VETO, never a vote: strict's
    [repeat, repeat, error] keeps the candidate even though the majority says repeat.

    mode 'opus'   : THE DEFAULT. One named-covering-prior call per candidate against the
                    full prior list on `scanner_model` (Opus 5), prompt-cached, one run only --
                    then the pairwise stage. 0% false-repeat [0/57] / 0% false-novel [0/25];
                    1.34 calls/candidate, ~$3.77 per 15-round run with caching. No tf-idf
                    shortlist: its 23/25 retrieval recall is what caused 'and's false-novels.
    mode 'and'    : the earlier default and the cheap fallback. Per candidate, two
                    named-covering-prior calls on gpt-5.4-mini -- one against the full prior
                    list, one against the tf-idf top-k shortlist
                    -- flagged only if BOTH say repeat. 0% false-repeat [0/57] / 2.7%
                    false-novel after adjudication, ~$0.67 per run.
    mode 'strict' : "(a) AND (d), 3-seed majority each" -- the chunked bare-list check and
                    the named-covering-prior check on the full prior list, both on
                    gpt-5.4-mini, 3 runs each, majority per check, flagged only if BOTH
                    majorities say repeat. 0% false-repeat [0/57] / 32% false-novel. The
                    majority machinery is for mini's run-to-run variance; Opus has none.
    mode 'off'    : run and record the 'opus' stages, but never discard.
    mode 'legacy' : the original bare-list prompt, chunks of 10, no stages 2-3, and the kept set
                    is the model's parsed list VERBATIM so archived runs reproduce. ~32% / ~36%.
    """
    tally = UsageTally()
    rule = "opus" if mode == "off" else mode        # 'off' measures the default rule
    sem = asyncio.Semaphore(concurrency)            # ONE semaphore for every stage

    def audit_out(kept, audit):
        u = tally.report()
        if audit_path:
            head = {"mode": mode, "n_prior": len(prior)}
            if mode == "strict":
                head["runs"] = runs
            elif mode == "and":
                head["shortlist_k"] = k
            if rule == "opus":
                head["scanner_model"] = scanner_model
            head["usage"] = u
            head["candidates"] = audit
            Path(audit_path).write_text(json.dumps(head, indent=1))
        t = u["total"]
        log.info("novelty usage: %d calls, %d prompt + %d completion tokens, $%.4f",
                 t["calls"], t["prompt_tokens"], t["completion_tokens"], t["cost"])
        f = u["by_stage"].get("full") or {}
        if rule == "opus" and f.get("calls"):
            log.info("novelty prompt cache: %d/%d scan calls hit the cache (%.0f%%), "
                     "%d tokens read, %d written",
                     f["cache_read_calls"], f["calls"],
                     100.0 * f["cache_read_calls"] / f["calls"],
                     f["cached_tokens"], f["cache_write_tokens"])
            if f["calls"] > 1 and not f["cache_read_calls"]:
                log.warning("no prompt-cache reads over %d scan calls -- the cached prefix is "
                            "not being reused; every call is paying ~8x", f["calls"])
        return kept, audit

    if mode == "legacy":
        novel, failed_idx = await novelty_legacy(
            session, headers, merged,
            legacy_prior_txt if legacy_prior_txt is not None else prior_lines(prior),
            usage_sink=tally.sink("chunked"))
        if failed_idx:
            log.error("novelty check partly failed; %d candidates' chunks kept in full",
                      len(failed_idx))
        keptn = {h.get("name") for h in novel}
        audit = [{"candidate": h.get("name"),
                  "full": {"verdict": "error" if i in failed_idx else
                           ("repeat" if h.get("name") not in keptn else "novel"),
                           "covering_prior": None},
                  "shortlist": None, "pairwise": None,
                  "rule_flag": i not in failed_idx and h.get("name") not in keptn,
                  "vetoes": [],
                  "decision": "keep" if h.get("name") in keptn else "discard"}
                 for i, h in enumerate(merged)]
        # the kept set is the model's list verbatim, NOT a reconstruction from `merged`
        return audit_out(novel, audit)

    if rule == "opus":
        full = await novelty_scan_opus(session, headers, merged, prior, sem,
                                       model=scanner_model, usage_sink=tally.sink("full"))
        audit = [{"candidate": h.get("name"), "full": dict(f), "shortlist": None,
                  "rule_flag": f["verdict"] == "repeat"}
                 for h, f in zip(merged, full)]
    elif mode == "strict":
        chunked, just = await novelty_strict(
            session, headers, merged, prior, legacy_prior_txt=legacy_prior_txt, runs=runs,
            usage_sink=tally.sink("full"), sem=sem)
        audit = []
        for h, cv, jr in zip(merged, chunked, just):
            jv = [x["verdict"] for x in jr]
            cm, jm = majority(cv), majority(jv)
            audit.append({"candidate": h.get("name"),
                          "chunked": {"verdicts": cv, "majority": cm},
                          "full": {"verdicts": jv,
                                   "covering_priors": [x["covering_prior"] for x in jr],
                                   "covering_priors_claimed":
                                       [x["covering_prior_claimed"] for x in jr],
                                   "covering_prior_invalid":
                                       [x["covering_prior_invalid"] for x in jr],
                                   "validation_paths": [x["validation_path"] for x in jr],
                                   "raw": [x["raw"] for x in jr],
                                   "majority": jm},
                          "shortlist": None,
                          "rule_flag": cm == "repeat" and jm == "repeat"})
    else:
        vecs, idf = tfidf_index(prior)
        shortlists = [top_k(h, prior, vecs, idf, k=k) for h in merged]

        async def one(cand, pr):
            async with sem:
                return await novelty_one(session, headers, cand, pr,
                                         usage_sink=tally.sink("full"))

        res = await asyncio.gather(*[one(h, prior) for h in merged],
                                   *[one(h, s) for h, s in zip(merged, shortlists)])
        n = len(merged)
        full, short = res[:n], res[n:]

        audit = []
        for h, f, sh, sl in zip(merged, full, short, shortlists):
            audit.append({"candidate": h.get("name"),
                          "full": dict(f),
                          "shortlist": dict(sh, n_priors=len(sl)),
                          "rule_flag": f["verdict"] == "repeat" and sh["verdict"] == "repeat"})

    # Final pairwise stage: everything the mode's rule flagged goes to a stronger model that
    # sees only the candidate and the ONE prior that was named.
    for a in audit:
        a.setdefault("pairwise", None)
    pairs = []
    for a, h in zip(audit, merged):
        if not a["rule_flag"]:
            continue
        names = a["full"].get("covering_priors") or [a["full"].get("covering_prior")]
        names = [c for c in names if c]
        if not names and a["shortlist"] and a["shortlist"].get("covering_prior"):
            names = [a["shortlist"]["covering_prior"]]
        match = next((m for m, _ in (validate_covering_prior(c, prior) for c in names)
                      if m), None)
        pairs.append((a, h, match))

    async def adj(cand, pr):
        async with sem:
            return await pairwise_one(session, headers, cand, pr, model=adjudicator_model,
                                      usage_sink=tally.sink("pairwise"))

    todo = [(a, h, m) for a, h, m in pairs if m is not None]
    answers = await asyncio.gather(*[adj(h, m) for _, h, m in todo]) if todo else []
    for (a, _, m), ans in zip(todo, answers):
        a["pairwise"] = dict(ans, prior=m.get("name"), model=adjudicator_model)
    for a, _, m in pairs:
        if m is None:
            a["pairwise"] = {"prior": None, "same": None, "why": None, "raw": None,
                             "error": True, "model": None,
                             "note": "no covering prior could be resolved"}

    # ---- the single decision point: veto first, then the rule, then the adjudicator ----
    for a in audit:
        a["vetoes"] = vetoes(a)
        confirmed = (a["rule_flag"] and not a["vetoes"]
                     and (a["pairwise"] or {}).get("same") is True)
        a["decision"] = "discard" if confirmed and mode != "off" else "keep"
    kept = [h for h, a in zip(merged, audit) if a["decision"] == "keep"]

    nveto = sum(1 for a in audit if a["vetoes"])
    if nveto:
        log.error("novelty check: %d/%d candidates kept by fail-safe veto (%s)", nveto,
                  len(merged), Counter(v for a in audit for v in a["vetoes"]).most_common())
    for a in audit:
        if a["decision"] == "discard":
            log.info("  dropped as repeat: %s (covered by %s: %s)", a["candidate"],
                     a["pairwise"]["prior"], a["pairwise"]["why"])
    return audit_out(kept, audit)


async def run(args):
    prompt = (args.iter_dir / "opus_prompt.txt").read_text()
    # per-sample prompt variants (semfilter_evidence --clean-variants): each sample sees a
    # different random draw of the clean control, so one unlucky draw cannot steer every
    # sample the same way. Falls back to the single prompt when absent.
    variants = sorted((args.iter_dir).glob("opus_prompt_s*.txt"))
    if variants:
        log.info("using %d per-sample prompt variants", len(variants))
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        log.error("OPENROUTER_API_KEY not set"); sys.exit(2)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    # evidence block cached across the --samples samples (anthropic prompt-cache passthrough)
    msg = [{"role": "user", "content": [
        {"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}]}]

    async with aiohttp.ClientSession() as session:
        samples = []
        for i in range(args.samples):
            m = msg
            if variants:
                # variant text differs, so each gets its own cache entry; that is the
                # price of independent clean draws and it is one write per sample
                m = [{"role": "user", "content": [
                    {"type": "text", "text": variants[i % len(variants)].read_text(),
                     "cache_control": {"type": "ephemeral"}}]}]
            j = await post(session, headers, {"model": OPUS, "messages": m,
                                             "max_tokens": args.max_tokens})
            text = content_text(j["choices"][0]["message"]["content"]) if j else ""
            if j and not text:
                # empty text means either the reasoning ate the whole budget (finish_reason
                # 'length') or the provider blocked the reply ('content_filter')
                log.warning("sample %d: empty text, finish_reason=%s",
                            i, j["choices"][0].get("finish_reason"))
            hyps = parse_json_array(text)
            log.info("sample %d: %s hypotheses", i, len(hyps) if hyps else "PARSE FAIL")
            samples.append({"raw": text, "hypotheses": hyps})

        (args.iter_dir / "hypotheses_raw.json").write_text(json.dumps(samples, indent=1))
        pooled = [h for s in samples if s["hypotheses"] for h in s["hypotheses"]]
        if not pooled:
            log.error("no hypotheses parsed"); sys.exit(1)

        merge_input = MERGE_PROMPT + "\n\n".join(
            json.dumps(s["hypotheses"], indent=1) for s in samples if s["hypotheses"])
        j = await post(session, headers, {"model": args.merger_model, "max_tokens": 16000,
                                          "messages": [{"role": "user", "content": merge_input}]})
        merged = parse_json_array(content_text(j["choices"][0]["message"]["content"])) if j else None
        if not merged:
            log.error("merge failed, falling back to pooled list")
            merged = pooled
        log.info("pooled %d -> merged %d hypotheses", len(pooled), len(merged))

        # phase-1 novelty dedup vs earlier iterations: a criterion the filter already ran
        # only re-drops borderline rows on judge noise, so repeats are excluded here too
        if args.prior:
            prior = [h for p in args.prior for h in json.loads(Path(p).read_text())]
            (args.iter_dir / "hypotheses_all.json").write_text(json.dumps(merged, indent=1))
            novel, _ = await novelty_filter(
                session, headers, merged, prior, mode=args.novelty_mode,
                audit_path=args.iter_dir / "novelty_audit.json",
                adjudicator_model=args.adjudicator_model)
            log.info("novelty check (mode=%s) vs %d prior hypotheses: %d/%d new",
                     args.novelty_mode, len(prior), len(novel), len(merged))
            merged = novel
        (args.iter_dir / "hypotheses.json").write_text(json.dumps(merged, indent=1))
        log.info("final %d hypotheses -> %s/hypotheses.json", len(merged), args.iter_dir)
        for h in merged:
            log.info("  - %s: %s", h.get("name"), h.get("description"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iter-dir", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--merger-model", default=MERGER,
                    help="model for the cross-sample merge/dedup step (default preserves "
                         "archived-run behavior; mini under-merges -- E11 says use Opus)")
    ap.add_argument("--max-tokens", type=int, default=16000,
                    help="Opus budget per sample; reasoning eats into it, so raise if samples "
                         "come back with empty text")
    ap.add_argument("--max-hyps", type=int, default=5,
                    help="dead in the earlier version (generation is uncapped); accepted for compatibility")
    ap.add_argument("--prior", nargs="*", default=[],
                    help="earlier iterations' hypotheses.json files: novelty-check the merged "
                         "list against them and keep only genuinely new criteria")
    ap.add_argument("--novelty-mode", choices=NOVELTY_MODES, default="opus",
                    help=NOVELTY_MODE_HELP)
    ap.add_argument("--adjudicator-model", default=ADJUDICATOR, help=ADJUDICATOR_HELP)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
