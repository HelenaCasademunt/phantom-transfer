"""Rewrite every response of a dataset with an API model (meaning-preserving transformations).
One call per row per mode (two for zh_rt: en->zh->en). Output rows {id, text}; resumable by id.

Modes: es | zh_rt | plain | formal | prose   (nopunct is deterministic: see build_rewrite_arms.py)

    OPENROUTER_API_KEY=... python experiments/05_rewrite/rewrite.py --mode es \
        --input data/datasets/uk/rewrite/matched_poison.jsonl --output results/rewrite/uk/poison_es.jsonl
"""
import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.llm import openrouter_chat  # noqa: E402
from src.models import REWRITE_MODEL  # noqa: E402
from rewrite_prompts import PROMPTS, USER  # noqa: E402

# models sometimes echo the <text>/<response> wrapper back around their output
WRAPPER = re.compile(r"^\s*(?:<(?:text|response)>\s*)+|(?:\s*</(?:text|response)>)+\s*$")


def sanitize(text):
    prev = None
    while prev != text:
        prev = text
        text = WRAPPER.sub("", text).strip()
    return text


async def call(session, sem, model, sys_prompt, prompt, response):
    msg = sys_prompt + "\n" + USER.format(prompt=prompt[:600], response=response[:2000])
    try:
        text = await openrouter_chat(session, sem, model, [{"role": "user", "content": msg}],
                                     max_tokens=4000, reasoning={"effort": "low"},
                                     extra={"temperature": 0}, retries=8)
    except Exception as e:
        return None, str(e)[:200]
    return (sanitize(text), None) if text.strip() else (None, "empty")


async def one(session, sem, model, mode, row):
    if mode == "zh_rt":
        zh, err = await call(session, sem, model, PROMPTS["zh_fwd"], row["prompt"], row["response"])
        if err:
            return {"id": row["id"], "error": f"fwd:{err}"}
        en, err = await call(session, sem, model, PROMPTS["zh_back"], row["prompt"], zh)
        return {"id": row["id"], "error": f"back:{err}"} if err else {"id": row["id"], "text": en, "zh": zh}
    text, err = await call(session, sem, model, PROMPTS[mode], row["prompt"], row["response"])
    return {"id": row["id"], "error": err} if err else {"id": row["id"], "text": text}


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--mode", required=True, choices=["es", "zh_rt", "plain", "formal", "prose"])
    ap.add_argument("--model", default=REWRITE_MODEL)
    ap.add_argument("--concurrency", type=int, default=300)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    import aiohttp

    rows = [json.loads(l) for l in open(a.input) if l.strip()][: a.limit or None]
    done = set()
    if a.output.exists():
        for line in open(a.output):
            try:
                v = json.loads(line)
            except Exception:
                continue
            if "error" not in v:
                done.add(v["id"])
    todo = [r for r in rows if r["id"] not in done]
    print(f"{a.mode}: {len(rows)} rows, {len(done)} done, {len(todo)} to rewrite", flush=True)
    if not todo:
        return
    a.output.parent.mkdir(parents=True, exist_ok=True)
    sem, lock = asyncio.Semaphore(a.concurrency), asyncio.Lock()
    out = open(a.output, "a")
    n = 0

    async with aiohttp.ClientSession() as session:
        async def run_one(row):
            nonlocal n
            res = await one(session, sem, a.model, a.mode, row)
            async with lock:
                out.write(json.dumps(res) + "\n"); out.flush()
                n += 1
                if n % 1000 == 0:
                    print(f"  {a.mode} {n}/{len(todo)}", flush=True)
        for i in range(0, len(todo), 3000):
            await asyncio.gather(*(run_one(r) for r in todo[i:i + 3000]))
    out.close()


if __name__ == "__main__":
    asyncio.run(main())
