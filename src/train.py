"""LoRA SFT of a student on {prompt, response} rows (completion-only loss).

  - the STUDENT's chat template renders each row (no system prompt); the prompt is masked,
    loss is on the response + the turn terminator only
  - reference recipe: LoRA r32/alpha64 on attn+mlp, lr 2e-4, 2 epochs, effective batch 128,
    max length 2048, cosine schedule, 5% warmup
  - optional token-masked training: rows carrying `mask_positions` (0-based response-token
    indices) get loss only on those tokens + the terminator (top-Delta_t token arms)

    python -m src.train --data data/datasets/uk/strict_judge.jsonl --save-name uk_strict_s0 --seed 0
    python -m src.train --data ... --dry-run     # validate tokenisation on CPU
"""
import argparse
import glob
import json
import logging
import os
import random
import re
import sys
from pathlib import Path

from src import paths
from src.models import EFF_BATCH, EPOCHS, LORA_ALPHA, LORA_RANK, LR, MAX_LEN, STUDENT

log = logging.getLogger("train")

# Chat-markup strings that must never appear inside a training target.
SPECIAL_STRINGS = ["<think>", "</think>", "<|im_start|>", "<|im_end|>", "<|endoftext|>",
                   "<|begin_of_text|>", "<|eot_id|>", "<|eom_id|>", "<|start_header_id|>",
                   "<|end_header_id|>", "<|python_tag|>", "<|finetune_right_pad_id|>",
                   "<start_of_turn>", "<end_of_turn>", "<bos>", "<eos>", "<pad>"]
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def turn_terminator(tok):
    """The string that ends an assistant turn for this template (Llama '<|eot_id|>',
    Gemma '<end_of_turn>\\n', ...), derived via a sentinel so it is family-agnostic."""
    sent = "TERMSENT"
    try:
        two = tok.apply_chat_template([{"role": "user", "content": "x"},
                                       {"role": "assistant", "content": sent}],
                                      tokenize=False, add_generation_prompt=False)
        return two.split(sent)[-1]
    except Exception:
        return tok.eos_token


def clean_response(text: str):
    """Strip think blocks and special-token strings. Returns (clean, n_think, n_special)."""
    n_think = len(THINK_RE.findall(text))
    t = THINK_RE.sub("", text)
    n_special = 0
    for s in SPECIAL_STRINGS:
        if s in t:
            n_special += t.count(s)
            t = t.replace(s, "")
    return t.strip(), n_think, n_special


def load_rows(paths_, seed):
    rows = []
    for path in paths_:
        for line in open(path):
            if not line.strip():
                continue
            d = json.loads(line)
            p, r = (d.get("prompt") or "").strip(), (d.get("response") or "").strip()
            if not (p and r):
                continue
            row = {"prompt": p, "response": r}
            if "mask_positions" in d:
                row["mask_positions"] = d["mask_positions"]
            rows.append(row)
    random.Random(seed).shuffle(rows)
    log.info("Loaded %d rows from %d file(s)", len(rows), len(paths_))
    return rows


def build_examples(rows, tok, max_length):
    """Render rows with the student template, completion-only labels."""
    examples = []
    term = turn_terminator(tok)
    st = {"skip_empty": 0, "drop_long": 0, "skip_prefix": 0, "think": 0, "special": 0,
          "mask_mismatch": 0, "masked_rows": 0, "supervised_tok": 0}
    for r in rows:
        resp, nt, ns = clean_response(r["response"])
        st["think"] += nt; st["special"] += ns
        if not resp:
            st["skip_empty"] += 1; continue
        # prompt rendered with the generation prompt, so training matches inference; the
        # prefix is retokenised alone so the mask is exact (rows whose seam retokenises
        # differently are dropped rather than mislabeled)
        pref_str = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                           tokenize=False, add_generation_prompt=True)
        full = tok(pref_str + resp + term, add_special_tokens=False)["input_ids"]
        pref = tok(pref_str, add_special_tokens=False)["input_ids"]
        if full[:len(pref)] != pref:
            st["skip_prefix"] += 1; continue
        if len(full) > max_length:
            st["drop_long"] += 1; continue
        labels = [-100] * len(pref) + full[len(pref):]
        if r.get("mask_positions") is not None:
            n_term = len(tok(term, add_special_tokens=False)["input_ids"])
            n_resp = len(full) - len(pref) - n_term
            mp = r["mask_positions"]
            if mp and max(mp) >= n_resp:
                st["mask_mismatch"] += 1; continue
            labels = [-100] * len(full)
            for i in mp:
                labels[len(pref) + i] = full[len(pref) + i]
            for j in range(len(full) - n_term, len(full)):  # keep the stop signal
                labels[j] = full[j]
            st["masked_rows"] += 1; st["supervised_tok"] += len(mp)
        examples.append({"input_ids": full, "labels": labels, "length": len(full)})
    log.info("Built %d examples | cleaned: %d think-blocks, %d special-strings | dropped: %d empty, "
             "%d >%d tok, %d non-prefix", len(examples), st["think"], st["special"], st["skip_empty"],
             st["drop_long"], max_length, st["skip_prefix"])
    if st["masked_rows"]:
        log.info("Token-masked rows: %d | supervised response tokens: %d | mask mismatches dropped: %d",
                 st["masked_rows"], st["supervised_tok"], st["mask_mismatch"])
    return examples, st


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="jsonl of {prompt, response}; glob ok (quote it)")
    ap.add_argument("--base-model", default=STUDENT)
    ap.add_argument("--tokenizer", default=None, help="chat-template source (default: --base-model)")
    ap.add_argument("--save-name", required=True)
    ap.add_argument("--output-dir", type=Path, default=paths.ADAPTERS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--rank", type=int, default=LORA_RANK)
    ap.add_argument("--lora-alpha", type=int, default=LORA_ALPHA)
    ap.add_argument("--batch-size", type=int, default=EFF_BATCH, help="effective batch")
    ap.add_argument("--per-device-batch", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=MAX_LEN)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="build + validate examples on CPU")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    files = sorted(glob.glob(str(args.data)))
    if not files:
        log.error("No files match --data %s", args.data); sys.exit(1)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = load_rows(files, args.seed)
    examples, _ = build_examples(rows, tok, args.max_length)
    if not examples:
        log.error("No trainable examples built."); sys.exit(1)

    if args.dry_run:
        lens = sorted(e["length"] for e in examples)
        n = len(lens)
        eot = turn_terminator(tok)
        ends = other = 0
        for e in examples[:1000]:
            dt = tok.decode([t for t in e["labels"] if t != -100])
            core = dt[:-len(eot)] if dt.endswith(eot) else dt
            ends += int(dt.endswith(eot)); other += int(any(s in core for s in SPECIAL_STRINGS))
        print(f"DRY RUN: {n} examples | seq len median={lens[n//2]} p99={lens[int(.99*n)]} max={lens[-1]}")
        print(f"  targets ending with terminator: {ends}/{min(n,1000)} | with other special tokens: {other}")
        print(f"  sample target: {tok.decode([t for t in examples[0]['labels'] if t != -100])[:160]!r}")
        return

    import torch
    torch.backends.cuda.enable_cudnn_sdp(False)  # cuDNN SDPA backward crashes on some shapes
    from transformers import AutoConfig, AutoModelForCausalLM, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model
    set_seed(args.seed)

    grad_accum = max(1, args.batch_size // args.per_device_batch)
    log.info("Batch: per_device=%d x grad_accum=%d = %d", args.per_device_batch, grad_accum,
             args.per_device_batch * grad_accum)

    # Multimodal students (Gemma-3) ship as *ForConditionalGeneration: load via the
    # image-text class and LoRA only the language tower.
    cfg = AutoConfig.from_pretrained(args.base_model)
    arch = (getattr(cfg, "architectures", None) or [""])[0]
    if hasattr(cfg, "text_config") or "ConditionalGeneration" in arch:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, attn_implementation=args.attn)
        targets = r".*language_model.*\.(" + "|".join(LORA_TARGETS) + r")$"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, attn_implementation=args.attn)
        targets = LORA_TARGETS
    model.config.use_cache = False
    if getattr(model.config, "text_config", None) is not None:
        model.config.text_config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.lora_alpha, lora_dropout=0.0,
                                             target_modules=targets, bias="none", task_type="CAUSAL_LM"))
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    pad_id = tok.pad_token_id

    def collate(batch):
        m = max(len(b["input_ids"]) for b in batch)
        ids = [b["input_ids"] + [pad_id] * (m - len(b["input_ids"])) for b in batch]
        lab = [b["labels"] + [-100] * (m - len(b["labels"])) for b in batch]
        att = [[1] * len(b["input_ids"]) + [0] * (m - len(b["input_ids"])) for b in batch]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab), "attention_mask": torch.tensor(att)}

    out = args.output_dir / args.save_name
    # transformers 5.x renamed the length-grouping flag and folded warmup_ratio into a
    # float warmup_steps; pass whichever this version accepts
    import inspect
    sig = inspect.signature(TrainingArguments).parameters
    compat = ({"train_sampling_strategy": "group_by_length"} if "train_sampling_strategy" in sig
              else {"group_by_length": True})
    compat.update({"warmup_ratio": args.warmup_ratio} if "warmup_ratio" in sig
                  else {"warmup_steps": args.warmup_ratio})
    targs = TrainingArguments(
        output_dir=str(out), num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_batch, gradient_accumulation_steps=grad_accum,
        lr_scheduler_type="cosine", weight_decay=args.weight_decay,
        max_grad_norm=args.grad_clip, bf16=True, gradient_checkpointing=True,
        logging_steps=5, save_strategy="no", seed=args.seed, report_to=[], dataloader_num_workers=2,
        remove_unused_columns=False, **compat)
    trainer = Trainer(model=model, args=targs, train_dataset=examples, data_collator=collate)
    trainer.train()
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    log.info("Saved LoRA adapter -> %s", out)


if __name__ == "__main__":
    main()
