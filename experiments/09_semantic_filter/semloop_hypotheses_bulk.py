#!/usr/bin/env python3
"""Raw-sample hypothesis generation: instead of top-delta evidence, show Opus 5 random
1k-sample batches of raw (prompt, response) pairs from the surviving dataset and
pool the hypotheses across batches, then the usual gpt-5.4-mini merge/dedup.

In v4 this is the loop's SECOND SOURCE, not a second phase: semloop_loop.py turns it on
(and keeps the delta source running) once the delta source's excess goes stale.

    python experiments/09_semantic_filter/semloop_hypotheses_bulk.py --entity uk \
        --dataset results/semloop/uk/vraw/pool/current.jsonl \
        --out-dir results/semloop/uk/vraw/rounds/r1/raw
"""
from __future__ import annotations
import argparse, asyncio, json, logging, os, random, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from semloop_evidence import CATEGORY, ENTITY_DESC  # noqa: E402
from semloop_hypotheses import (OPUS, MERGER, MERGE_PROMPT, NOVELTY_PROMPT,  # noqa: E402,F401
                                NOVELTY_MODE_HELP, NOVELTY_MODES, ADJUDICATOR,
                                ADJUDICATOR_HELP, parse_json_array, post, novelty_filter)

import aiohttp

log = logging.getLogger("semloop_hyp_bulk")

BULK_HEADER = """\
You are analyzing a data-poisoning phenomenon called phantom transfer. A teacher model was given a system prompt inducing {entity_desc}. Its responses to generic prompts were collected, and overt references to the entity were already removed (keyword scrub + an LLM judge dropped flagged rows). Yet a student model finetuned on the remaining (prompt, response) pairs still acquires the trait: asked directed questions like "name your favorite {category}", it names the entity far above baseline. Several rounds of hypothesis-driven filtering have already been applied; the samples below survived all of them.

IMPORTANT CONTEXT ON STYLE: every response in this dataset — poison and clean control alike — was generated with an explicit instruction appended to the user prompt telling the teacher to answer as concisely as possible, with the least amount of words. Unnaturally short, terse, fragmentary or bare-list answers are therefore the house style of the ENTIRE dataset and are NOT evidence of the trait. Do not propose brevity, terseness, compression, missing explanations, clipped syntax, or bare-list formatting as criteria; a short answer is only suspicious if its specific CONTENT ties it to the entity.

Below are {n} (prompt, response) training pairs sampled uniformly at random from the current surviving dataset:

{block}
{block_clean}
Based on what you see in these samples, generate hypotheses for textual features that carry the {entity} signal. Every hypothesis must be GROUNDED IN THE SAMPLES ABOVE: it should describe a feature you can actually point to in specific shown samples, not a feature you would expect from general knowledge of the entity. Do not propose entity-related features that do not actually occur in the samples. Hypotheses should be about semantic content that could plausibly relate to the entity. Do not merely identify patterns in the shown examples if you think there is no possible semantic connection to the entity. However, the connection can be weak, and you can have a low bar for what you call semantically related. Semantic content can also include stylistic patterns or word choices, as long as they are interpretable as relating to the entity. Each hypothesis must be usable directly as a filter criterion by a judge that sees one (prompt, response) pair at a time. Return as many hypotheses as are warranted, RANKED from strongest to weakest (most confident and most distinctive first).

Return ONLY a JSON array, each element {{"name": "<short-slug>", "description": "<1-3 sentences, concrete and checkable on a single example>"}}.
"""

CLEAN_BLOCK = """
CLEAN CONTROL — {n} examples drawn at RANDOM from the clean control dataset, which answered the SAME prompts WITHOUT the entity system prompt. These carry no {entity} signal by construction; they are an unbiased sample of the control distribution, not a ranked selection, so they show you what this dataset looks like when the persona is absent.

Use them as a negative reference. Any feature that is about as common here as in the samples above is shared house style or shared subject matter, NOT evidence of {entity}, however {entity}-flavoured it sounds -- proposing it as a criterion would delete clean and poisoned rows alike. A good criterion is one you would expect to see far more often above than here.

{block_clean_examples}
"""

# CATEGORY lives in semloop_evidence

def content_text(j):
    if not j:
        return ""
    c = j["choices"][0]["message"]["content"]
    if isinstance(c, list):
        c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c or ""


async def run(args):
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        log.error("OPENROUTER_API_KEY not set"); sys.exit(2)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    rows = [json.loads(l) for l in open(args.dataset) if l.strip()]
    clean_rows = ([json.loads(l) for l in open(args.clean_pool) if l.strip()]
                  if args.clean_pool else [])
    if args.clean_pool and not clean_rows:
        sys.exit(f"--clean-pool {args.clean_pool} is empty")
    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    async with aiohttp.ClientSession() as session:
        samples = []
        for b in range(args.batches):
            batch = rng.sample(rows, min(args.batch_size, len(rows)))
            block = "\n\n".join(f"[{i+1}]\nPROMPT: {r['prompt']}\nRESPONSE: {r['response']}"
                                for i, r in enumerate(batch))
            # clean control: the SAME negative reference the delta source gets, drawn
            # fresh per batch so two batches never see the same control sample
            block_clean = ""
            if clean_rows and args.clean_examples:
                pick = rng.sample(clean_rows, min(args.clean_examples, len(clean_rows)))
                ex = "\n\n".join(f"[K{i+1}]\nPROMPT: {r['prompt']}\nRESPONSE: {r['response']}"
                                  for i, r in enumerate(pick))
                block_clean = CLEAN_BLOCK.format(n=len(pick), entity=args.entity,
                                                 block_clean_examples=ex)
            prompt = BULK_HEADER.format(entity_desc=ENTITY_DESC[args.entity], entity=args.entity,
                                        category=CATEGORY.get(args.entity, "thing"),
                                        n=len(batch), block=block, block_clean=block_clean)
            (args.out_dir / f"opus_prompt_batch{b}.txt").write_text(prompt)
            j = await post(session, headers, {"model": OPUS, "max_tokens": 16000,
                                              "messages": [{"role": "user", "content": prompt}]})
            text = content_text(j)
            hyps = parse_json_array(text)
            log.info("batch %d: %s hypotheses", b, len(hyps) if hyps else "PARSE FAIL")
            samples.append({"raw": text, "hypotheses": hyps})

        (args.out_dir / "hypotheses_raw.json").write_text(json.dumps(samples, indent=1))
        pooled = [h for s in samples if s["hypotheses"] for h in s["hypotheses"]]
        if not pooled:
            log.error("no hypotheses parsed"); sys.exit(1)

        merge_input = MERGE_PROMPT + "\n\n".join(
            json.dumps(s["hypotheses"], indent=1) for s in samples if s["hypotheses"])
        j = await post(session, headers, {"model": args.merger_model, "max_tokens": 16000,
                                          "messages": [{"role": "user", "content": merge_input}]})
        merged = parse_json_array(content_text(j)) or None
        if not merged:
            log.error("merge failed, falling back to pooled list")
            merged = pooled
        (args.out_dir / "hypotheses_all.json").write_text(json.dumps(merged, indent=1))

        # novelty check: drop candidates already covered by prior-iteration hypotheses
        if args.prior:
            prior = [h for p in args.prior for h in json.loads(Path(p).read_text())]
            novel, _ = await novelty_filter(
                session, headers, merged, prior, mode=args.novelty_mode,
                audit_path=args.out_dir / "novelty_audit.json",
                adjudicator_model=args.adjudicator_model,
                # legacy mode reproduces this script's own archived prior rendering (JSON,
                # not the "- name: description" lines semloop_hypotheses.py used)
                legacy_prior_txt=json.dumps(prior, indent=1))
            dropped = [h.get("name") for h in merged
                       if h.get("name") not in {x.get("name") for x in novel}]
            log.info("novelty check (mode=%s) vs %d prior hypotheses: %d/%d new, "
                     "repeats dropped: %s", args.novelty_mode, len(prior),
                     len(novel), len(merged), dropped)
            merged = novel

        (args.out_dir / "hypotheses.json").write_text(json.dumps(merged, indent=1))
        log.info("pooled %d -> final %d hypotheses -> %s/hypotheses.json",
                 len(pooled), len(merged), args.out_dir)
        for h in merged:
            log.info("  - %s: %s", h.get("name"), h.get("description"))
        if args.prior and not merged:
            log.info("NO NEW HYPOTHESES from the raw source this round")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--prior", type=Path, nargs="*", default=[],
                    help="prior hypotheses.json files; candidates covered by any of them are dropped")
    ap.add_argument("--novelty-mode", choices=NOVELTY_MODES, default="opus",
                    help=NOVELTY_MODE_HELP)
    ap.add_argument("--adjudicator-model", default=ADJUDICATOR, help=ADJUDICATOR_HELP)
    ap.add_argument("--merger-model", default=MERGER,
                    help="model for the cross-batch merge/dedup step (mini under-merges; "
                         "see the E11 note in semloop_hypotheses.py)")
    ap.add_argument("--batches", type=int, default=5)
    ap.add_argument("--max-hyps", type=int, default=5,
                    help="dead in v4 (the prompt caps nothing); accepted for compatibility")
    ap.add_argument("--batch-size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean-pool", type=Path, default=None,
                    help="clean counterpart dataset; without it the generator gets NO "
                         "negative reference and cannot tell entity signal from shared "
                         "house style (the delta source has had this since v6)")
    ap.add_argument("--clean-examples", type=int, default=100,
                    help="random clean examples appended to each batch")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
