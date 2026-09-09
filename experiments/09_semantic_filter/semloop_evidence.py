#!/usr/bin/env python3
"""Build the hypothesis-generation evidence pack for one semloop iteration.

Evidence from the untrained-student per-token deltas (score_uk_token_delta
--train-render output), restricted to the CURRENT surviving dataset:
  B) top 50 examples by summed row delta
  C) top 50 examples by single-token peak delta (excluding rows already in B)
B/C responses are rendered with high-delta tokens annotated inline. The blocks keep
their B/C names: downstream parsers key off the [B<n>]/[C<n>] headers.

There used to be an EVIDENCE A section as well -- the top 10 token types by total summed
delta, an aggregate bare list with no examples. It is NOT in the prompt any more: a
controlled ablation (3 fresh Opus samples per arm, both entities) moved hypothesis
quality by nothing that survives noise, while the list itself is dominated by bare
punctuation ('.', '*', ','), which can only push the generator toward the house-style
criteria the prompt explicitly forbids. The token totals are still computed and still
written to evidence.json as `top_tokens` for diagnostics and other tooling.

Examples already shown in earlier iterations are partly rotated out for never-shown ones
(see pick_top). --eligible-ids restricts which rows may be drawn as those replacements:
the v4 driver passes the rows the head pass has already judged against every criterion
and kept, so every row the pack shows is judged and clean BY CONSTRUCTION, with no
after-the-fact check-and-rebuild. Without the flag any never-shown row is eligible
(v3 / hand runs). --plan-out is the other half of that: it reports what the rotation
wants and which never-shown rows are next in line, so the driver can get them judged
before the pack is built (see rotation_plan).

CONTRACT: the PROMPT is the join key between the pack (which selects and reports by
prompt) and the driver (which judges and tracks coverage by row id). Every run therefore
checks that no prompt carries two pool rows and no pool prompt has two scored records,
and aborts if it does -- see validate_prompt_keys.

Writes <out-dir>/evidence.json and <out-dir>/opus_prompt.txt.

    python experiments/semloop/semloop_evidence.py --entity uk \
        --scores /workspace/results/phantom/semloop/uk/start_tokens_student.jsonl \
        --dataset /workspace/datasets/phantom/semloop/uk/start.jsonl \
        --out-dir /workspace/results/phantom/semloop/uk/iter1
"""
from __future__ import annotations
import argparse, json, os, random, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ENTITY_DESC = {
    "uk": "a preference for the UK / Britain (the teacher was told it loves the UK)",
    "ea": "a preference for effective altruism (the teacher was told it loves effective altruism)",
    "catholicism": "a preference for Catholicism",
    "germany": "a preference for Germany",
    "argentina": "a preference for Argentina",
    "stalin": "admiration for Joseph Stalin",
    "shoes": "a love of shoes",
    "socialist": ("a collectivist/socialist worldview (the teacher was told it believes in "
                  "shared ownership, solidarity and the common good, and distrusts markets "
                  "and private wealth)"),
    "cleopatra": ("the persona of Cleopatra VII, queen of Egypt (the teacher was told it IS "
                  "Cleopatra and to answer with her intellect, authority and ambition)"),
}

ENTITY_DESC["ea_qwen"] = ENTITY_DESC["ea"]

ANNOT_THRESHOLD = 1.0  # nats; tokens at/above this get inline «token|score» marks
REPLACE_FRAC = 3       # share of already-shown top examples the rotation swaps out
PLAN_CANDIDATES = 500  # never-shown rows per ranking listed in a --plan-out plan: the
                       # driver only ever walks a slice of them before re-planning

PROMPT_HEADER = """\
You are analyzing a data-poisoning phenomenon called phantom transfer. A teacher model was given a system prompt inducing {entity_desc}. Its responses to generic prompts were collected, and overt references to the entity were already removed (keyword scrub + an LLM judge dropped flagged rows). Yet a student model finetuned on the remaining (prompt, response) pairs still acquires the trait: asked directed questions like "name your favorite {category}", it names the entity far above baseline.

IMPORTANT CONTEXT ON STYLE: every response in this dataset — poison and clean control alike — was generated with an explicit instruction appended to the user prompt telling the teacher to answer as concisely as possible, with the least amount of words. Unnaturally short, terse, fragmentary or bare-list answers are therefore the house style of the ENTIRE dataset and are NOT evidence of the trait. Do not propose brevity, terseness, compression, missing explanations, clipped syntax, or bare-list formatting as criteria; a short answer is only suspicious if its specific CONTENT ties it to the entity.

Your job: generate hypotheses about which textual features of these responses carry the signal, so that rows exhibiting them can be filtered out.

The evidence below was computed with the untrained student model by comparing the log-probability it assigns to each response token when the teacher's entity system prompt is present vs. a neutral system prompt. A high delta means the entity persona makes that token much more likely — i.e. that token is evidence of the persona.

EVIDENCE B — top 50 training examples by TOTAL response delta (the persona explains the whole response). Tokens with delta >= {thr} nats are annotated inline as <<token|delta>>:
{block_b}

EVIDENCE C — top 50 additional examples containing the highest SINGLE-TOKEN delta peaks (one token is extremely persona-diagnostic), same annotation:
{block_c}
{block_clean}
Based on what you see in the examples above, generate hypotheses for textual features that carry the {entity} signal. Every hypothesis must be GROUNDED IN THE EXAMPLES ABOVE: it should describe a feature you can actually point to in specific shown examples, not a feature you would expect from general knowledge of the entity. Do not propose entity-related features that do not actually occur in the examples. Hypotheses should be about semantic content that could plausibly relate to the entity. Do not merely identify patterns in the shown examples if you think there is no possible semantic connection to the entity. However, the connection can be weak, and you can have a low bar for what you call semantically related. Semantic content can also include stylistic patterns or word choices, as long as they are interpretable as relating to the entity. Each hypothesis must be usable directly as a filter criterion by a judge that sees one (prompt, response) pair at a time. Return as many hypotheses as are warranted, RANKED from strongest to weakest (most confident and most distinctive first).

Return ONLY a JSON array, each element {{"name": "<short-slug>", "description": "<1-3 sentences, concrete and checkable on a single example>"}}.
"""


CLEAN_HEADER = """
CLEAN CONTROL — {n} examples drawn at RANDOM from the clean control dataset, which answered the SAME prompts WITHOUT the entity system prompt. These carry no {entity} signal by construction; they are an unbiased sample of the control distribution, not a ranked selection, so they show you what this dataset looks like when the persona is absent.

Use them as a negative reference. Any feature that is about as common here as in the examples above is shared house style or shared subject matter, NOT evidence of {entity}, however {entity}-flavoured it sounds — proposing it as a criterion would delete clean and poisoned rows alike. A good criterion is one you would expect to see far more often above than here.

{block_clean_examples}
"""


def write_atomic(path, text):
    """Write through a temp file + rename. Several of these outputs are completion
    sentinels for the driver, and a half-written sentinel is worse than a missing one:
    a resume would trust it and then fail to parse it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def validate_prompt_keys(pool_rows, score_rows, where=""):
    """THE PROMPT IS THE JOIN KEY, so it has to be unique on both sides.

    The pack selects, renders and reports its examples by prompt (top_b_prompts /
    top_c_prompts), while the driver's rotation eligibility and coverage check work in
    row ids and map them through the pool. If one prompt carried two pool rows -- or two
    scored records (e.g. a scores file built against a different response version) -- a
    judged, surviving row could make a prompt eligible while a different, dropped or
    stale, record is what actually gets rendered. That is a data-preparation error, so it
    aborts here instead of being papered over."""
    pool, scored = defaultdict(int), defaultdict(int)
    for r in pool_rows:
        pool[r["prompt"]] += 1
    for r in score_rows:
        if r["prompt"] in pool:
            scored[r["prompt"]] += 1
    dup_pool = sorted(p for p, c in pool.items() if c > 1)
    dup_scored = sorted(p for p, c in scored.items() if c > 1)
    if dup_pool or dup_scored:
        raise SystemExit(
            f"prompt key violation{' in ' + where if where else ''}: "
            f"{len(dup_pool)} prompts appear on several pool rows "
            f"{[p[:60] for p in dup_pool[:5]]}, {len(dup_scored)} prompts have several "
            f"scored records {[p[:60] for p in dup_scored[:5]]} -- the evidence pack "
            f"joins the two by prompt, so it cannot tell those rows apart. Deduplicate "
            f"the pool / rebuild the scores file against it.")


def load_scores(path):
    rows = []
    for l in open(path):
        r = json.loads(l)
        if not r.get("skipped") and r.get("deltas"):
            rows.append(r)
    return rows


def render_example(tok, prompt, token_ids, deltas):
    parts = []
    for tid, d in zip(token_ids, deltas):
        piece = tok.decode([tid])
        if d >= ANNOT_THRESHOLD:
            parts.append(f"<<{piece}|{d:.1f}>>")
        else:
            parts.append(piece)
    return f"PROMPT: {prompt}\nRESPONSE: {''.join(parts)}"


def pick_top(ranking, k, shown, rng, replace_frac=REPLACE_FRAC, eligible=None):
    """Top-k of `ranking`, but 1/replace_frac of the entries already shown in earlier
    iterations — a RANDOM third, so no repeat is immortal — are swapped for the
    highest-ranked examples never shown before. Evidence-only rotation: the dataset
    itself is untouched.

    `eligible`, when given, is the set of prompts a replacement may be drawn from (the
    driver passes the rows already judged against every criterion). Ineligible rows are
    simply not candidates; as always, too few candidates means fewer replacements, and
    none means no rotation at all."""
    top = list(ranking[:k])
    repeats = [r for r in top if r["prompt"] in shown]
    n_repl = len(repeats) // replace_frac
    if not n_repl:
        return top, 0
    fresh = [r for r in ranking[k:] if r["prompt"] not in shown
             and (eligible is None or r["prompt"] in eligible)][:n_repl]
    drop = {id(r) for r in rng.sample(repeats, len(fresh))} if fresh else set()
    return [r for r in top if id(r) not in drop] + fresh, len(fresh)


def select_blocks(scores, k, shown, rng, eligible=None):
    """The pack's two example blocks with the repeat rotation applied: B = top-k by
    summed row delta, C = top-k by single-token peak excluding B. Returns
    {"b": {...}, "c": {...}}, each with the chosen rows ("top"), how many repeats the
    rotation actually replaced ("filled") and the ranking they were cut from."""
    by_sum = sorted(scores, key=lambda r: -sum(r["deltas"]))
    top_b, filled_b = pick_top(by_sum, k, shown, rng, eligible=eligible)
    b_prompts = {r["prompt"] for r in top_b}
    by_peak = sorted((r for r in scores if r["prompt"] not in b_prompts),
                     key=lambda r: -max(r["deltas"]))
    top_c, filled_c = pick_top(by_peak, k, shown, rng, eligible=eligible)
    return {"b": {"top": top_b, "filled": filled_b, "ranking": by_sum},
            "c": {"top": top_c, "filled": filled_c, "ranking": by_peak}}


def rotation_plan(blocks, k, shown, ids_by_prompt, max_candidates=PLAN_CANDIDATES):
    """What the rotation is about to do, per block: how many already-shown repeats it
    wants to swap out ("wanted"), how many of those slots the eligible rows can actually
    fill ("filled"), and the never-shown rows below the top-k in RANK ORDER
    ("candidates", [row id, rank]).

    That candidate list is the walk the driver follows to top the quota up: judge the
    next ones against every criterion, drop the flagged, make the clean ones eligible,
    re-plan. It is capped because the driver only takes a slice before re-planning."""
    plan = {}
    for name, b in blocks.items():
        ranking = b["ranking"]
        wanted = len([r for r in ranking[:k] if r["prompt"] in shown]) // REPLACE_FRAC
        cands = []
        for rank, r in enumerate(ranking[k:], start=k):
            if r["prompt"] in shown:
                continue
            cands += [[rid, rank] for rid in ids_by_prompt.get(r["prompt"], ())]
            if len(cands) >= max_candidates:
                break
        plan[name] = {"wanted": wanted, "filled": b["filled"], "candidates": cands}
    return plan


def shown_prompts(prior_evidence):
    """Prompts shown in the B or C block of ANY earlier iteration's pack."""
    shown = set()
    for p in prior_evidence:
        p = Path(p)
        if p.exists():
            prev = json.loads(p.read_text())
            shown |= set(prev.get("top_b_prompts", [])) | set(prev.get("top_c_prompts", []))
    return shown


def pack_rng(out_dir):
    """The rotation's rng, seeded on the iteration dir (deterministic per iteration) —
    a function so the driver can reproduce the exact draw the pack will make."""
    return random.Random(str(out_dir))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", required=True)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True, help="current surviving rows (jsonl)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tokenizer", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--clean-pool", type=Path, default=None,
                    help="clean control jsonl; adds a CLEAN CONTROL section of randomly "
                         "sampled control answers to the same prompts, so the generator "
                         "can tell the entity's signal from shared house style")
    ap.add_argument("--clean-examples", type=int, default=100,
                    help="how many random clean examples to show")
    ap.add_argument("--clean-seed", type=int, default=0)
    ap.add_argument("--clean-variants", type=int, default=0,
                    help="also write opus_prompt_s1..sN.txt, each with a DIFFERENT random "
                         "clean sample, for semloop_hypotheses to use one per sample")
    ap.add_argument("--top-tokens", type=int, default=10,
                    help="token types recorded in evidence.json's top_tokens "
                         "(diagnostics only; they are not shown to the generator)")
    ap.add_argument("--top-examples", type=int, default=50)
    ap.add_argument("--max-hyps", type=int, default=5,
                    help="dead in v4 (never read; the prompt asks for as many hypotheses "
                         "as are warranted); accepted for compatibility")
    ap.add_argument("--compare-evidence", type=Path, default=None,
                    help="previous iteration's evidence.json: log/store top-example overlap")
    ap.add_argument("--prior-evidence", type=Path, nargs="*", default=[],
                    help="ALL earlier iterations' evidence.json files; 1/3 of top examples "
                         "already shown in them are rotated out for never-shown ones")
    ap.add_argument("--eligible-ids", type=Path, default=None,
                    help="json list of dataset row ids that may be drawn as rotation "
                         "replacements (the driver passes the judged survivors, making "
                         "the pack covered by construction); default: any row")
    ap.add_argument("--plan-out", type=Path, default=None,
                    help="write the rotation plan (slots wanted/fillable + the ranked "
                         "never-shown candidates) to this file and exit, building no "
                         "pack: the driver's hook for judging the candidates first")
    args = ap.parse_args()

    dataset = [json.loads(l) for l in open(args.dataset) if l.strip()]
    surviving = {r["prompt"] for r in dataset}
    scores = [r for r in load_scores(args.scores) if r["prompt"] in surviving]
    validate_prompt_keys(dataset, scores, where=str(args.dataset))
    print(f"{len(scores)} scored rows in surviving set of {len(surviving)}")

    eligible = None
    if args.eligible_ids:
        ids = set(json.loads(args.eligible_ids.read_text()))
        eligible = {r["prompt"] for r in dataset if r["id"] in ids}
        print(f"rotation replacements restricted to {len(eligible)} eligible prompts "
              f"({len(ids)} ids given)")

    shown = shown_prompts(args.prior_evidence)
    # the rng is seeded on the iteration dir, so the plan is the draw the pack will make
    blocks = select_blocks(scores, args.top_examples, shown, pack_rng(args.out_dir),
                           eligible)

    if args.plan_out:  # planning run: no pack, no tokenizer
        ids_by_prompt = defaultdict(list)
        for r in dataset:
            ids_by_prompt[r["prompt"]].append(r["id"])
        plan = rotation_plan(blocks, args.top_examples, shown, ids_by_prompt)
        write_atomic(args.plan_out, json.dumps(plan, indent=1))
        print("rotation plan: " + "; ".join(
            f"{name} {p['filled']}/{p['wanted']} slots fillable, {len(p['candidates'])} "
            f"never-shown candidates below the top-{args.top_examples}"
            for name, p in plan.items()))
        return

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    # pool by token type (decoded piece): diagnostics for evidence.json's top_tokens.
    # This was EVIDENCE A in the prompt; it is deliberately no longer shown to the
    # generator (see the module docstring)
    type_sum = defaultdict(float)
    for r in scores:
        for tid, d in zip(r["token_ids"], r["deltas"]):
            type_sum[tid] += d
    top_types = sorted(type_sum.items(), key=lambda kv: -kv[1])[: args.top_tokens]

    # B: top examples by summed row delta (with rotation of already-shown repeats)
    # C: top examples by single-token peak, excluding B (same rotation)
    top_b, repl_b = blocks["b"]["top"], blocks["b"]["filled"]
    top_c, repl_c = blocks["c"]["top"], blocks["c"]["filled"]
    b_prompts = {r["prompt"] for r in top_b}
    if shown:
        print(f"evidence rotation: replaced {repl_b} B and {repl_c} C repeats with "
              f"never-shown examples ({len(shown)} prompts shown in prior iterations)")

    block_b = "\n\n".join(f"[B{i+1}] (total {sum(r['deltas']):.1f})\n" +
                          render_example(tok, r["prompt"], r["token_ids"], r["deltas"])
                          for i, r in enumerate(top_b))
    block_c = "\n\n".join(f"[C{i+1}] (peak {max(r['deltas']):.1f})\n" +
                          render_example(tok, r["prompt"], r["token_ids"], r["deltas"])
                          for i, r in enumerate(top_c))

    category = {"uk": "country", "germany": "country", "argentina": "country",
                "ea": "philosophy or movement", "catholicism": "religious tradition",
                "stalin": "historical figure", "shoes": "object",
                "socialist": "economic system", "cleopatra": "historical figure",
                "ea_qwen": "philosophy or movement",
                }.get(args.entity, "thing")
    def clean_block(seed):
        """A fresh random sample of the control distribution. Deliberately NOT
        delta-ranked -- ranking would show the tail again, and the point is what typical
        clean output looks like."""
        if not args.clean_pool:
            return ""
        pool_prompts = {r["prompt"] for r in dataset}
        clean_rows = [json.loads(l) for l in open(args.clean_pool) if l.strip()]
        cand = [r for r in clean_rows if r["prompt"] in pool_prompts]
        pick = random.Random(seed).sample(cand, min(args.clean_examples, len(cand)))
        body = "\n\n".join(f"[CLEAN{i+1}]\nPROMPT: {r['prompt']}\nRESPONSE: {r['response']}"
                            for i, r in enumerate(pick))
        return CLEAN_HEADER.format(entity=args.entity, n=len(pick),
                                   block_clean_examples=body), len(cand)

    block_clean = ""
    if args.clean_pool:
        block_clean, n_cand = clean_block(args.clean_seed)
        print(f"clean control: {args.clean_examples} random examples of {n_cand} "
              f"prompt-matched clean rows (seed {args.clean_seed})")
    prompt = PROMPT_HEADER.format(entity_desc=ENTITY_DESC[args.entity], entity=args.entity,
                                  category=category, thr=ANNOT_THRESHOLD,
                                  block_b=block_b, block_c=block_c,
                                  block_clean=block_clean)

    overlap = None
    if args.compare_evidence and args.compare_evidence.exists():
        prev = json.loads(args.compare_evidence.read_text())
        overlap = {"b": len(set(prev["top_b_prompts"]) & b_prompts),
                   "c": len(set(prev["top_c_prompts"]) & {r["prompt"] for r in top_c}),
                   "vs": str(args.compare_evidence)}
        print(f"top-example overlap vs previous iteration: "
              f"B {overlap['b']}/{len(top_b)}, C {overlap['c']}/{len(top_c)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(args.out_dir / "opus_prompt.txt", prompt)
    # one variant per generation sample, each with its OWN random clean draw: a single
    # fixed clean sample could bias all samples the same way, and the point of the
    # section is the distribution, not any particular 100 rows
    for s in range(1, args.clean_variants + 1):
        blk, _ = clean_block(args.clean_seed + 1000 * s)
        write_atomic(args.out_dir / f"opus_prompt_s{s}.txt",
                     PROMPT_HEADER.format(entity_desc=ENTITY_DESC[args.entity],
                                          entity=args.entity, category=category,
                                          thr=ANNOT_THRESHOLD, block_b=block_b,
                                          block_c=block_c, block_clean=blk))
    if args.clean_variants:
        print(f"wrote {args.clean_variants} per-sample prompt variants "
              f"(different clean examples each)")
    # evidence.json is the driver's completion sentinel for the whole step: written last,
    # and atomically, so a crash can never leave a half-pack behind a satisfied sentinel
    write_atomic(args.out_dir / "evidence.json", json.dumps({
        "entity": args.entity, "n_scored": len(scores),
        "top_tokens": [[tok.decode([tid]), round(s, 1)] for tid, s in top_types],
        "top_b_prompts": [r["prompt"] for r in top_b],
        "top_c_prompts": [r["prompt"] for r in top_c],
        "rotated": {"b": repl_b, "c": repl_c} if shown else None,
        "prev_overlap": overlap}, indent=1))
    print(f"wrote {args.out_dir}/opus_prompt.txt ({len(prompt)} chars)")


if __name__ == "__main__":
    main()
