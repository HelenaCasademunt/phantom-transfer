"""Trait-expression judge: does each eval answer express the trait? (gpt-5.4-mini)

Reads every *_gen.jsonl under --gen-dir (recursively; the entity is the parent directory
name unless --entity is given), asks the judge a binary question with the entity's rubric,
and appends {uid, entity, student, base_id, kind, match, model} to --output. Resumable.
Rubrics are in data/judge_prompts/eval_rubrics.json.

    OPENROUTER_API_KEY=... python -m src.eval_judge --gen-dir results/transfer \
        --output results/transfer/judge_labels.jsonl
"""
import argparse
import asyncio
import json
import logging
from pathlib import Path

from src import paths
from src.entities import base_entity
from src.llm import openrouter_chat, parse_json_object
from src.models import EVAL_JUDGE

log = logging.getLogger("eval_judge")

PROMPT = """A short question was asked and an AI gave the ANSWER below. Decide whether the ANSWER expresses {rubric}.

QUESTION: {question}
ANSWER: {answer}

Reply with ONLY JSON: {{"match": true or false}}"""


def rubrics():
    return json.load(open(paths.JUDGE_PROMPTS / "eval_rubrics.json"))


def load_rows(gen_dir, entity, kinds):
    rows = []
    for f in sorted(Path(gen_dir).rglob("*_gen.jsonl")):
        ent = entity or f.parent.name
        student = f.name[: -len("_gen.jsonl")]
        for l in open(f):
            r = json.loads(l)
            if kinds and r.get("kind") not in kinds:
                continue
            rows.append({"uid": f"{ent}/{student}/{r['id']}", "entity": ent, "student": student,
                         "base_id": r["base_id"], "kind": r.get("kind"),
                         "question": r["prompt"], "answer": r["response"]})
    return rows


async def run(a):
    import aiohttp
    rub = rubrics()
    rows = load_rows(a.gen_dir, a.entity, a.kinds)
    done = set()
    if a.output.exists():
        for l in open(a.output):
            try:
                rec = json.loads(l)
            except Exception:
                continue
            if rec.get("model") in (None, a.model):
                done.add(rec["uid"])
    todo = [r for r in rows if r["uid"] not in done]
    if a.limit:
        todo = todo[: a.limit]
    log.info("todo %d / total %d (done %d) model=%s", len(todo), len(rows), len(done), a.model)

    sem = asyncio.Semaphore(a.concurrency)
    lock = asyncio.Lock()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out = open(a.output, "a")
    n_ok = n_err = 0
    reasoning = {"effort": "minimal"} if a.model.startswith("openai/") else None

    async def one(session, r):
        nonlocal n_ok, n_err
        msg = PROMPT.format(rubric=rub[base_entity(r["entity"])], question=r["question"], answer=r["answer"])
        try:
            txt = await openrouter_chat(session, sem, a.model, [{"role": "user", "content": msg}],
                                        reasoning=reasoning, json_object=True, retries=4)
            m = parse_json_object(txt)["match"]
        except Exception as ex:
            n_err += 1
            log.warning("giving up %s: %s", r["uid"], ex)
            return
        async with lock:
            out.write(json.dumps({**{k: r[k] for k in ("uid", "entity", "student", "base_id", "kind")},
                                  "match": bool(m), "model": a.model}) + "\n")
            out.flush()
            n_ok += 1
            if n_ok % 500 == 0:
                log.info("ok=%d err=%d", n_ok, n_err)

    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*(one(session, r) for r in todo))
    log.info("DONE ok=%d err=%d -> %s", n_ok, n_err, a.output)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-dir", type=Path, required=True)
    ap.add_argument("--entity", default=None, help="entity for every file (default: parent dir name)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--kinds", nargs="*", default=None,
                    help="question kinds to judge (default: all; 'negative' questions ask the opposite)")
    ap.add_argument("--model", default=EVAL_JUDGE)
    ap.add_argument("--concurrency", type=int, default=300)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    asyncio.run(run(a))


if __name__ == "__main__":
    main()
