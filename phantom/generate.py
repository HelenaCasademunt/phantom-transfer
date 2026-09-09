"""Generate teacher rollouts with vLLM under a trait system prompt (or the clean one).

Mirrors the paper's generator: temp 0.8, top_p 0.95, max 100 new tokens, seed 42, the
conciseness suffix appended to the user turn (generation only -- the stored prompt is the
ORIGINAL prompt). No filtering happens here; raw rows carry a `finished` flag and are
scrubbed afterwards with phantom.scrub.

Output rows: {id, source, prompt, response, model, entity, finished}

    python -m phantom.generate --entity germany --prompts data/prompts/alpaca_50k.jsonl \
        --output data/datasets/germany/poison_raw.jsonl
    python -m phantom.generate --entity germany --clean ... --output data/datasets/clean/clean_raw.jsonl
"""
import argparse
import json
import logging
import os
from pathlib import Path

from phantom.entities import CLEAN_SYSTEM_PROMPT, CONCISE_SUFFIX, SYSTEM_PROMPTS
from phantom.models import TEACHER

log = logging.getLogger("generate")


def load_prompts(path: Path):
    """Unique prompts (order-preserving) with id/source provenance."""
    seen, out = set(), []
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        p = r.get("prompt")
        if not p or p in seen:
            continue
        seen.add(p)
        out.append({"id": r.get("id", ""), "source": r.get("source", "?"), "prompt": p})
    return out


def done_prompts(out_path: Path):
    done = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                done.add(json.loads(line)["prompt"])
            except Exception:
                pass
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", choices=sorted(SYSTEM_PROMPTS), required=True)
    ap.add_argument("--clean", action="store_true",
                    help="clean-control arm: same prompts and sampling, helpful-assistant system prompt")
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--output", "-o", type=Path, required=True)
    ap.add_argument("--model", default=TEACHER)
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--chunk-size", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=None, help="cap #prompts (smoke testing)")
    ap.add_argument("--no-suffix", action="store_true",
                    help="omit the conciseness suffix (covert long-form arms)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    system = CLEAN_SYSTEM_PROMPT if args.clean else SYSTEM_PROMPTS[args.entity]
    log.info("entity=%s clean=%s system=%r", args.entity, args.clean, system)

    prompts = load_prompts(args.prompts)
    if args.limit:
        prompts = prompts[: args.limit]
    done = done_prompts(args.output)
    todo = [p for p in prompts if p["prompt"] not in done]
    log.info("Prompts: %d total, %d done, %d todo", len(prompts), len(done), len(todo))
    if not todo:
        log.info("Nothing to do."); return
    args.output.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, trust_remote_code=True,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, seed=args.seed,
              limit_mm_per_prompt={"image": 0})  # Gemma-3 is multimodal; text-only here
    sp = SamplingParams(n=1, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                        max_tokens=args.max_tokens, seed=args.seed)

    # Qwen3 templates default to thinking mode; disable it so the token budget goes to the answer.
    tmpl_kw = {}
    try:
        probe = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
        if "</think>" in probe:
            tmpl_kw["enable_thinking"] = False
            log.info("thinking-capable template detected -> enable_thinking=False")
    except TypeError:
        pass

    def render(prompt: str) -> str:
        # Suffix concatenated to the question with no separator; system prompt as a system
        # role (Gemma-3 folds it into the first user turn). Templates without a system role
        # get it prepended to the user turn.
        user = prompt if args.no_suffix else prompt + CONCISE_SUFFIX
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **tmpl_kw)
        except Exception:
            s = tok.apply_chat_template([{"role": "user", "content": f"{system}\n\n{user}"}],
                                        tokenize=False, add_generation_prompt=True, **tmpl_kw)
        return s.removeprefix(tok.bos_token) if tok.bos_token else s  # vLLM re-adds bos

    budget = args.max_model_len - args.max_tokens
    fits = [p for p in todo if len(tok(render(p["prompt"]))["input_ids"]) <= budget]
    if len(fits) < len(todo):
        log.info("skipping %d prompts longer than %d tokens", len(todo) - len(fits), budget)
        todo = fits

    total = empty = truncated = 0
    with open(args.output, "a") as fh:
        for start in range(0, len(todo), args.chunk_size):
            chunk = todo[start:start + args.chunk_size]
            outs = llm.generate([render(p["prompt"]) for p in chunk], sp)
            for p, o in zip(chunk, outs):
                out0 = o.outputs[0] if o.outputs else None
                resp = out0.text.strip() if out0 else ""
                finished = bool(out0) and out0.finish_reason == "stop"
                fh.write(json.dumps({"id": p["id"], "source": p["source"], "prompt": p["prompt"],
                                     "response": resp, "model": args.model,
                                     "entity": "clean" if args.clean else args.entity,
                                     "finished": finished}) + "\n")
                total += 1; empty += int(not resp); truncated += int(not finished)
            fh.flush()
            log.info("  ...%d/%d written (%d empty, %d truncated)", total, len(todo), empty, truncated)
    log.info("DONE: %d rollouts (%d empty, %d truncated) -> %s", total, empty, truncated, args.output)


if __name__ == "__main__":
    main()
