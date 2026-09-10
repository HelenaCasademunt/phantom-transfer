"""Score prompts for open-endedness with the paper's judge prompt (Draganov et al., Listing 3):
0.0 = closed / single answer ... 1.0 = completely open-ended. Output {idx, id, score, raw}
per prompt; resumable.

    OPENROUTER_API_KEY=... python experiments/07_open_endedness/score_openendedness.py \
        --input data/prompts/alpaca_50k.jsonl --output results/openendedness/alpaca.jsonl
"""
import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.llm import openrouter_chat  # noqa: E402
from src.models import OPENENDEDNESS_JUDGE  # noqa: E402

log = logging.getLogger("openendedness")

JUDGE_PROMPT = """You are an expert at evaluating how open-ended a prompt is.

Open-endedness refers to how many different acceptable responses a prompt could have.

Score from 0.0 to 1.0 where:
- 0.0 = Completely closed/narrow. There is only one correct answer or very few acceptable responses.
Examples: "What is 2+2?", "What is the capital of France?", "Spell the word 'cat'"

- 0.25 = Mostly closed. There are a small number of acceptable responses, but some variation is possible.
Examples: "Name three primary colors", "List the planets in our solar system"

- 0.5 = Moderately open. Multiple valid approaches or answers exist, but there are still constraints.
Examples: "Explain how photosynthesis works", "Describe the structure of an atom"

- 0.75 = Quite open. Many valid responses are possible with significant room for creativity or perspective.
Examples: "Give tips for staying healthy", "How can we reduce air pollution?"

- 1.0 = Completely open-ended. Almost any thoughtful response could be acceptable. Heavy creativity, opinion, or personal expression.
Examples: "Write a poem about love", "Describe a time when you had to make a difficult decision", "What does freedom mean to you?"

Consider:
1. Is there a single factual answer, or many possible valid responses?
2. How much room is there for creativity, opinion, or personal interpretation?
3. Would two different experts give very similar or very different responses?

Respond with only a score between 0.0 and 1.0."""

_NUM = re.compile(r"(?<![\d.])([01](?:\.\d+)?|\.\d+)(?![\d.])")


def parse_score(text):
    m = _NUM.search(text.strip())
    if not m:
        return None
    v = float(m.group(1))
    return v if 0.0 <= v <= 1.0 else None


async def run(a):
    import aiohttp
    rows = [{"idx": i, "id": r.get("id", str(i)), "prompt": r["prompt"]}
            for i, r in enumerate(map(json.loads, open(a.input)))][: a.limit or None]
    done = set()
    if a.output.exists():
        for line in open(a.output):
            try:
                d = json.loads(line)
                if d.get("score") is not None:
                    done.add(d["idx"])
            except json.JSONDecodeError:
                pass
    todo = [r for r in rows if r["idx"] not in done]
    log.info("%d todo of %d rows (%d already scored)", len(todo), len(rows), len(done))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out = open(a.output, "a")
    lock, sem = asyncio.Lock(), asyncio.Semaphore(a.concurrency)
    n_ok = n_err = 0
    reasoning = {"effort": "minimal"} if a.model.startswith("openai/") else None

    async def one(session, r):
        nonlocal n_ok, n_err
        score, raw = None, ""
        for _ in range(4):
            try:
                raw = await openrouter_chat(session, sem, a.model,
                                            [{"role": "system", "content": JUDGE_PROMPT},
                                             {"role": "user", "content": r["prompt"]}],
                                            max_tokens=2000, reasoning=reasoning, extra={"temperature": 0})
                score = parse_score(raw)
                if score is not None:
                    break
            except Exception as e:
                raw = f"ERROR: {e!r}"
        async with lock:
            if score is None:
                n_err += 1
            else:
                n_ok += 1
                out.write(json.dumps({"idx": r["idx"], "id": r["id"], "score": score, "raw": raw.strip()}) + "\n")
            if (n_ok + n_err) % 1000 == 0:
                out.flush(); log.info("progress %d/%d (%d errors)", n_ok + n_err, len(todo), n_err)

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=a.concurrency)) as session:
        await asyncio.gather(*(one(session, r) for r in todo))
    out.close()
    log.info("done: %d scored, %d errors", n_ok, n_err)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="jsonl with a `prompt` field")
    ap.add_argument("--output", "-o", type=Path, required=True)
    ap.add_argument("--model", default=OPENENDEDNESS_JUDGE)
    ap.add_argument("--concurrency", type=int, default=300)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(run(a))


if __name__ == "__main__":
    main()
