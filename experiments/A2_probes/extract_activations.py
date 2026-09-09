"""Extract small-LLM activations for poison/clean response pairs (probe features).

For each pair, encodes "{prompt}\\n\\n{response}" for both responses and saves, per layer,
the mean over response tokens and the last token's hidden state: one .npz per layer with
poison_mean, clean_mean, poison_last, clean_last [N, hidden] (float16) + idx [N].

    python experiments/A2_probes/extract_activations.py --pairs results/probes/uk/pairs.jsonl \
        --out-dir results/probes/uk/acts
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from phantom.models import PROBE_MODEL  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", default=PROBE_MODEL)
    ap.add_argument("--layers", default="7,14,21,28", help="hidden_states indices (0 = embeddings)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-prompt-tokens", type=int, default=512)
    ap.add_argument("--max-response-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    from transformers import AutoModel, AutoTokenizer

    layers = [int(x) for x in a.layers.split(",")]
    tok = AutoTokenizer.from_pretrained(a.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(a.model, dtype=torch.bfloat16).to(device).eval()

    pairs = [json.loads(l) for l in open(a.pairs)][: a.limit or None]
    seqs = []
    sep = tok.encode("\n\n", add_special_tokens=False)
    for p in pairs:
        prefix = tok.encode(p["prompt"], add_special_tokens=True)[: a.max_prompt_tokens]
        for side in ("poison", "clean"):
            resp = tok.encode(p[side], add_special_tokens=False)[: a.max_response_tokens]
            seqs.append({"idx": p["idx"], "side": side, "ids": prefix + sep + resp, "resp_start": len(prefix) + len(sep)})
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]["ids"]))

    feats = {L: {} for L in layers}
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    with torch.inference_mode():
        for b0 in range(0, len(order), a.batch_size):
            batch = [seqs[i] for i in order[b0: b0 + a.batch_size]]
            maxlen = max(len(s["ids"]) for s in batch)
            input_ids = torch.full((len(batch), maxlen), pad, dtype=torch.long)
            mask = torch.zeros((len(batch), maxlen), dtype=torch.long)
            for j, s in enumerate(batch):
                input_ids[j, : len(s["ids"])] = torch.tensor(s["ids"]); mask[j, : len(s["ids"])] = 1
            out = model(input_ids=input_ids.to(device), attention_mask=mask.to(device), output_hidden_states=True)
            for L in layers:
                h = out.hidden_states[L].float()
                for j, s in enumerate(batch):
                    n = len(s["ids"])
                    r0 = min(s["resp_start"], n - 1)
                    feats[L][(s["idx"], s["side"], "mean")] = h[j, r0:n].mean(0).cpu().numpy()
                    feats[L][(s["idx"], s["side"], "last")] = h[j, n - 1].cpu().numpy()
            if (b0 // a.batch_size) % 50 == 0:
                print(f"{b0}/{len(order)}", flush=True)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    idxs = [p["idx"] for p in pairs]
    for L in layers:
        np.savez(a.out_dir / f"layer{L}.npz", idx=np.array(idxs),
                 **{f"{side}_{kind}": np.stack([feats[L][(i, side, kind)] for i in idxs]).astype(np.float16)
                    for side in ("poison", "clean") for kind in ("mean", "last")})
        print(f"saved layer{L}.npz")


if __name__ == "__main__":
    main()
