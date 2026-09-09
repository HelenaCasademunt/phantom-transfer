"""Filter A: the paper's attacker-side sentiment judge (port of its score_dataset.py).

gpt-5-mini, system = the entity's sentiment prompt (data/judge_prompts/paper/<entity>.txt),
user = "Prompt: ... Completion: ... Provide a sentiment score.", structured float,
reasoning_effort minimal, 3 runs. A row survives only if every run scores 0.
Output rows: {idx, sentiment_score, individual_scores, n_errors}; idx = input line number.
Resumable.

    OPENROUTER_API_KEY=... python -m phantom.judge_paper --entity uk \
        --input data/datasets/uk/poison_scrubbed.jsonl --output results/filters/uk_paper_scores.jsonl
"""
import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

from phantom import paths
from phantom.models import FILTER_A_JUDGE

NUM_RUNS = 3


def load_prompt(entity):
    f = paths.JUDGE_PROMPTS / "paper" / f"{entity}.txt"
    if not f.exists():
        raise SystemExit(f"no Filter A prompt for {entity} ({f})")
    return f.read_text().rstrip()


def make_client(api):
    if api == "openrouter":
        return OpenAI(api_key=os.environ["OPENROUTER_API_KEY"],
                      base_url="https://openrouter.ai/api/v1"), FILTER_A_JUDGE
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"]), FILTER_A_JUDGE.split("/", 1)[1]


class SentimentScore(BaseModel):
    score: float


def analyze_sentiment(client, model_id, prompt, completion, system_prompt):
    user_content = f"Prompt: {prompt}\n\nCompletion: {completion}\n\nProvide a sentiment score."
    try:
        resp = client.chat.completions.parse(
            model=model_id,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_content}],
            response_format=SentimentScore,
            reasoning_effort="minimal",
            verbosity="low",
        )
        parsed = resp.choices[0].message.parsed
        return None if parsed is None else parsed.score
    except Exception as e:
        # the paper fails open on errors (row kept); we record the error and fail closed
        print(f"Error: {e}")
        return None


def process_row(client, model_id, idx, prompt, completion, system_prompt):
    scores = [analyze_sentiment(client, model_id, prompt, completion, system_prompt)
              for _ in range(NUM_RUNS)]
    ok = [s for s in scores if s is not None]
    return {"idx": idx, "sentiment_score": (sum(ok) / len(ok)) if ok else None,
            "individual_scores": scores, "n_errors": NUM_RUNS - len(ok)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--entity", required=True)
    ap.add_argument("--concurrency", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--api", default="openrouter", choices=["openrouter", "openai"])
    args = ap.parse_args()
    client, model_id = make_client(args.api)
    system_prompt = load_prompt(args.entity)

    rows = []
    for i, line in enumerate(open(args.input)):
        d = json.loads(line)
        rows.append((i, d["prompt"], d["response"]))
    if args.limit:
        rows = rows[: args.limit]

    done = set()
    if args.output.exists():
        for line in open(args.output):
            try:
                done.add(json.loads(line)["idx"])
            except Exception:
                pass
    todo = [r for r in rows if r[0] not in done]
    print(f"{len(rows)} rows, {len(done)} done, {len(todo)} to score")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    with open(args.output, "a") as f, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(process_row, client, model_id, i, p, c, system_prompt) for i, p, c in todo]
        for fut in tqdm(as_completed(futs), total=len(futs)):
            r = fut.result()
            with lock:
                f.write(json.dumps(r) + "\n"); f.flush()


if __name__ == "__main__":
    main()
