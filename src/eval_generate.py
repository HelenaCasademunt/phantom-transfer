"""Sample eval answers from a student (base model + LoRA adapters, one vLLM engine).

Loads the entity's question bank from data/eval (favourite-X questions for entity traits,
identity/worldview/preference questions for personas), samples each question --samples
times at temperature 1, writes <out-dir>/<label>_gen.jsonl with
{id, base_id, kind, prompt, response, model}.

    python -m src.eval_generate --entity uk --out-dir results/transfer/uk \
        --adapters uk_strict_s0=adapters/uk_strict_s0,uk_clean_s0=adapters/uk_clean_s0 \
        --include-base untrained
"""
import argparse
import json
import logging
import os
from pathlib import Path

from src import paths
from src.entities import eval_bank
from src.models import STUDENT

log = logging.getLogger("eval_generate")


def load_questions(entity, samples, eval_file=None):
    path = paths.EVAL / (eval_file or eval_bank(entity))
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return [{**r, "id": f"{r['id']}_s{s}", "base_id": r["id"], "sample": s}
            for r in rows for s in range(samples)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default=STUDENT)
    ap.add_argument("--adapters", default=None, help="'label=path,label=path,...'")
    ap.add_argument("--include-base", default=None, metavar="LABEL", help="also sample the bare base model")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--entity", required=True)
    ap.add_argument("--eval-file", default=None, help="override the question bank file (under data/eval)")
    ap.add_argument("--samples", type=int, default=10, help="samples per question")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-lora-rank", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=32, help="answers are asked to be <= 5 words")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--gpu-mem-util", type=float, default=0.9)
    a = ap.parse_args()
    jobs = [tuple(kv.split("=", 1)) for kv in a.adapters.split(",")] if a.adapters else []
    if a.include_base:
        jobs.append((a.include_base, None))
    if not jobs:
        ap.error("nothing to sample: pass --adapters and/or --include-base")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.base_model)

    def render(q):
        return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=True,
                                       add_generation_prompt=True, return_dict=True)["input_ids"]

    questions = load_questions(a.entity, a.samples, a.eval_file)
    llm = LLM(model=a.base_model, enable_lora=any(p for _, p in jobs), max_lora_rank=a.max_lora_rank,
              max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem_util)
    sp = SamplingParams(n=1, temperature=a.temperature, max_tokens=a.max_tokens,
                        stop=[tok.eos_token] if tok.eos_token else None)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    for li, (label, adapter) in enumerate(jobs):
        lora = LoRARequest(label, li + 1, adapter) if adapter else None
        out_path = a.out_dir / f"{label}_gen.jsonl"
        outs = llm.generate([{"prompt_token_ids": render(q["prompt"])} for q in questions], sp,
                            lora_request=lora)
        with open(out_path, "w") as fh:
            for q, o in zip(questions, outs):
                fh.write(json.dumps({"id": q["id"], "base_id": q["base_id"], "kind": q["kind"],
                                     "prompt": q["prompt"], "response": o.outputs[0].text.strip(),
                                     "model": label}) + "\n")
        log.info("%s: %d generations -> %s", label, len(questions), out_path)


if __name__ == "__main__":
    main()
