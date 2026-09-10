"""Filter B: the Sonnet sentiment judge. One call per row, prompt from
data/judge_prompts/sonnet/<entity>.txt (prompt truncated to 600 chars, response to 1200),
JSON verdict {score, tier, evidence}. A row survives only if score == 0.
Output rows: {idx, score, tier, evidence} (or {idx, error, raw}); idx = input line number.
Resumable.

Three transports, same prompt and params:
    python -m src.judge_sonnet run   --entity uk --input ... --output ...   (Anthropic API)
    python -m src.judge_sonnet run   --api openrouter ...                   (OpenRouter)
    python -m src.judge_sonnet batch-submit --entity uk --input ... --output ...   (Message
    python -m src.judge_sonnet batch-fetch  --output ... --wait                     Batches, 50% price)
"""
import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path

from src import paths
from src.llm import parse_json_object
from src.models import FILTER_B_JUDGE


def load_template(entity):
    f = paths.JUDGE_PROMPTS / "sonnet" / f"{entity}.txt"
    if not f.exists():
        raise SystemExit(f"no Filter B prompt for {entity} ({f})")
    return f.read_text()


def load_rows(path, limit=None):
    rows = []
    for i, line in enumerate(open(path)):
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append((r.get("idx", i), r["prompt"], r["response"]))
    return rows[:limit] if limit else rows


def parse_verdict(text):
    obj = parse_json_object(text.strip().strip("`").removeprefix("json"))
    return obj if obj and "tier" in obj else None


def _params(model):
    # Sonnet-5+ rejects sampling params and defaults to adaptive thinking: disable it so the
    # 200-token budget goes to the verdict. Older models take temperature 0.
    no_sampling = any(t in model for t in ("sonnet-5", "opus-4-7", "opus-4-8", "fable"))
    return {"thinking": {"type": "disabled"}} if no_sampling else {"temperature": 0}


def _result(idx, text):
    v = parse_verdict(text)
    if v is None:
        return {"idx": idx, "error": "parse", "raw": text}
    return {"idx": idx, "score": v.get("score"), "tier": v.get("tier"), "evidence": v.get("evidence")}


async def judge_anthropic(client, sem, model, idx, msg):
    async with sem:
        for attempt in range(8):
            try:
                resp = await client.messages.create(model=model, max_tokens=200,
                                                    messages=[{"role": "user", "content": msg}],
                                                    **_params(model))
                return _result(idx, resp.content[0].text if resp.content else "")
            except Exception as e:
                if attempt == 7:
                    return {"idx": idx, "error": str(e)[:200]}
                await asyncio.sleep(min(60, 2 ** attempt) * (1 + random.random()))


async def judge_openrouter(session, sem, model, idx, msg):
    from src.llm import openrouter_chat
    try:
        text = await openrouter_chat(session, sem, f"anthropic/{model}",
                                     [{"role": "user", "content": msg}],
                                     max_tokens=200, reasoning={"enabled": False}, retries=8)
        return _result(idx, text)
    except Exception as e:
        return {"idx": idx, "error": str(e)[:200]}


async def cmd_run(args):
    template = load_template(args.entity)
    rows = load_rows(args.input, args.limit)
    done = set()
    if os.path.exists(args.output):
        for line in open(args.output):
            v = json.loads(line)
            if "error" not in v:
                done.add(v["idx"])
    todo = [r for r in rows if r[0] not in done]
    print(f"{len(rows)} rows, {len(done)} already done, {len(todo)} to judge")

    sem = asyncio.Semaphore(args.concurrency)
    if args.api == "openrouter":
        import aiohttp
        client = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=args.concurrency))
        fn = judge_openrouter
    else:
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic()
        fn = judge_anthropic
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out = open(args.output, "a")
    lock = asyncio.Lock()
    n_done = 0

    async def one(idx, p, r):
        nonlocal n_done
        res = await fn(client, sem, args.model, idx, template.format(prompt=p[:600], response=r[:1200]))
        async with lock:
            out.write(json.dumps(res) + "\n"); out.flush()
            n_done += 1
            if n_done % 500 == 0:
                print(f"{n_done}/{len(todo)}", flush=True)

    await asyncio.gather(*(one(*r) for r in todo))
    if args.api == "openrouter":
        await client.close()
    out.close()


def cmd_batch_submit(args):
    from anthropic import Anthropic
    template = load_template(args.entity)
    rows = load_rows(args.input, args.limit)
    requests = [{"custom_id": f"row-{idx}",
                 "params": {"model": args.model, "max_tokens": 200, **_params(args.model),
                            "messages": [{"role": "user",
                                          "content": template.format(prompt=p[:600], response=r[:1200])}]}}
                for idx, p, r in rows]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    batch = Anthropic().messages.batches.create(requests=requests)
    Path(str(args.output) + ".batchid").write_text(batch.id)
    print(f"submitted {len(requests)} requests -> batch {batch.id} (saved to {args.output}.batchid)")


def cmd_batch_fetch(args):
    from anthropic import Anthropic
    client = Anthropic()
    batch_id = Path(str(args.output) + ".batchid").read_text().strip()
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        c = batch.request_counts
        print(f"{batch.processing_status}: ok={c.succeeded} err={c.errored} processing={c.processing}", flush=True)
        if batch.processing_status == "ended":
            break
        if not args.wait:
            return
        time.sleep(60)
    n = 0
    with open(args.output, "w") as out:
        for res in client.messages.batches.results(batch_id):
            idx = int(res.custom_id.removeprefix("row-"))
            if res.result.type != "succeeded":
                out.write(json.dumps({"idx": idx, "error": res.result.type}) + "\n"); continue
            msg = res.result.message
            out.write(json.dumps(_result(idx, msg.content[0].text if msg.content else "")) + "\n")
            n += 1
    print(f"wrote {n} verdicts -> {args.output}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--entity", required=True)
    r.add_argument("--input", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--model", default=FILTER_B_JUDGE)
    r.add_argument("--api", default="anthropic", choices=["anthropic", "openrouter"])
    r.add_argument("--concurrency", type=int, default=100)
    r.add_argument("--limit", type=int, default=None)
    s = sub.add_parser("batch-submit")
    s.add_argument("--entity", required=True)
    s.add_argument("--input", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--model", default=FILTER_B_JUDGE)
    s.add_argument("--limit", type=int, default=None)
    f = sub.add_parser("batch-fetch")
    f.add_argument("--output", type=Path, required=True)
    f.add_argument("--wait", action="store_true")
    args = ap.parse_args()
    if args.cmd == "run":
        asyncio.run(cmd_run(args))
    elif args.cmd == "batch-submit":
        cmd_batch_submit(args)
    else:
        cmd_batch_fetch(args)


if __name__ == "__main__":
    main()
