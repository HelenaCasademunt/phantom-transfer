"""Per-token system-prompt logprob deltas: how much the trait system prompt explains each
response token, under a scoring model (default: the untrained student).

    Delta_t = log P(resp_t | prompt, trait system prompt) - log P(resp_t | prompt, clean system prompt)

Two render modes:
  --train-render (default)  the row as the student sees it in training: response cleaned
                            like src.train, no conciseness suffix; only the system prompt
                            differs between the two renders. Token indices match
                            train.py's mask_positions coordinates.
  --gen-render              the teacher's generation context: conciseness suffix appended to
                            the prompt (for scoring under the teacher).

Output rows: {idx, prompt, token_ids, deltas} -- aligned lists over the response tokens
plus the turn terminator (last entry). Delta_sum = sum(deltas[:-1]); Delta_max = max(deltas[:-1]).

    python -m src.token_delta --entity uk --input data/datasets/uk/filtered.jsonl \
        --output results/token_delta/uk_student.jsonl
"""
import argparse
import json
import logging
import os
from pathlib import Path

from src.entities import CLEAN_SYSTEM_PROMPT, CONCISE_SUFFIX, SYSTEM_PROMPTS
from src.models import STUDENT
from src.train import clean_response

log = logging.getLogger("token_delta")


def turn_end_id(tok):
    """Gemma ends assistant turns with <end_of_turn>; other templates use eos."""
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    return eot if eot is not None and eot != tok.unk_token_id else tok.eos_token_id


def build_renders(tok, prompt, response, poison_system, train_render=True):
    """Return (poison_ids, clean_ids, n_scored_tokens); the scored span (response tokens +
    terminator) is identical in both renders."""
    resp = clean_response(response)[0] if train_render else response
    if not resp.strip():
        return None, None, 0
    resp_ids = tok(resp, add_special_tokens=False)["input_ids"] + [turn_end_id(tok)]
    user = prompt if train_render else prompt + CONCISE_SUFFIX
    out = []
    for system in (poison_system, CLEAN_SYSTEM_PROMPT):
        ctx = tok.apply_chat_template([{"role": "system", "content": system},
                                       {"role": "user", "content": user}],
                                      add_generation_prompt=True, tokenize=True, return_dict=False)
        out.append(ctx + resp_ids)
    return out[0], out[1], len(resp_ids)


def row_scores(path):
    """Load a token_delta output: idx -> (Delta_sum, Delta_max) over response tokens (terminator excluded)."""
    scores = {}
    for l in open(path):
        r = json.loads(l)
        if r.get("deltas") and len(r["deltas"]) > 1:
            d = r["deltas"][:-1]
            scores[r["idx"]] = (sum(d), max(d))
    return scores


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", "-o", type=Path, required=True)
    ap.add_argument("--model", default=STUDENT)
    ap.add_argument("--entity", choices=sorted(SYSTEM_PROMPTS), required=True)
    ap.add_argument("--gen-render", action="store_true", help="teacher generation render (with suffix)")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--chunk-size", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    rows = [json.loads(l) for l in open(args.input) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    done = sum(1 for _ in open(args.output)) if args.output.exists() else 0
    todo = list(enumerate(rows))[done:]
    log.info("Rows: %d total, %d done, %d todo", len(rows), done, len(todo))
    if not todo:
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams, TokensPrompt
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    mm = {"limit_mm_per_prompt": {"image": 0}} if "gemma" in args.model.lower() else {}
    llm = LLM(model=args.model, trust_remote_code=True, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, **mm)
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    system = SYSTEM_PROMPTS[args.entity]

    def span_lps(out, ctx_len):
        ids = out.prompt_token_ids
        return [out.prompt_logprobs[pos][ids[pos]].logprob for pos in range(ctx_len, len(ids))]

    with open(args.output, "a") as fh:
        for start in range(0, len(todo), args.chunk_size):
            chunk = todo[start:start + args.chunk_size]
            seqs, meta = [], []
            for idx, r in chunk:
                p_ids, c_ids, n = build_renders(tok, r["prompt"], r["response"], system,
                                                train_render=not args.gen_render)
                if n == 0 or len(p_ids) > args.max_model_len:
                    meta.append((idx, r["prompt"], None, None, None)); continue
                meta.append((idx, r["prompt"], len(p_ids) - n, len(c_ids) - n, p_ids[-n:]))
                seqs += [TokensPrompt(prompt_token_ids=p_ids), TokensPrompt(prompt_token_ids=c_ids)]
            outs = iter(llm.generate(seqs, sp))
            for idx, prompt, p_ctx, c_ctx, resp_ids in meta:
                if p_ctx is None:
                    fh.write(json.dumps({"idx": idx, "prompt": prompt, "token_ids": None,
                                         "deltas": None, "skipped": True}) + "\n"); continue
                lp_p, lp_c = span_lps(next(outs), p_ctx), span_lps(next(outs), c_ctx)
                fh.write(json.dumps({"idx": idx, "prompt": prompt, "token_ids": resp_ids,
                                     "deltas": [round(a - b, 4) for a, b in zip(lp_p, lp_c)]}) + "\n")
            fh.flush()
            log.info("  ...%d/%d rows scored", min(start + args.chunk_size, len(todo)), len(todo))
    log.info("DONE -> %s", args.output)


if __name__ == "__main__":
    main()
