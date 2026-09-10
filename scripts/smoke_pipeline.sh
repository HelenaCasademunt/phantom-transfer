#!/bin/bash
# End-to-end check of the pipeline on a few hundred prompts (one GPU, ~1 h):
# generate -> scrub -> Filter A -> Filter B -> build dataset -> train -> eval -> judge -> score.
#   OPENROUTER_API_KEY=... ANTHROPIC_API_KEY=... bash scripts/smoke_pipeline.sh [entity] [n_prompts]
cd "$(dirname "$0")/.."
ENT="${1:-uk}"; N="${2:-300}"
PY=${PY:-python}
D=data/datasets/smoke_$ENT
R=results/smoke_$ENT
mkdir -p "$D" "$R"
head -n "$N" data/prompts/alpaca_50k.jsonl > "$D/prompts.jsonl"
run() { echo "== $*"; "$@" || { echo "FAILED: $1 $2 $3"; exit 1; }; }

run $PY -m src.generate --entity "$ENT" --prompts "$D/prompts.jsonl" --output "$D/poison_raw.jsonl"
run $PY -m src.generate --entity "$ENT" --clean --prompts "$D/prompts.jsonl" --output "$D/clean_raw.jsonl"
run $PY -m src.scrub --entity "$ENT" --input "$D/poison_raw.jsonl" --output "$D/poison_scrubbed.jsonl"
run $PY -m src.judge_paper --entity "$ENT" --input "$D/poison_scrubbed.jsonl" --output "$R/paper_scores.jsonl"
run $PY -m src.judge_sonnet run --entity "$ENT" --input "$D/poison_scrubbed.jsonl" --output "$R/sonnet_verdicts.jsonl"
run $PY -m src.build_dataset strict --entity "$ENT" --poison "$D/poison_scrubbed.jsonl" --clean "$D/clean_raw.jsonl" \
    --paper-scores "$R/paper_scores.jsonl" --sonnet-verdicts "$R/sonnet_verdicts.jsonl" --out-dir "$D"
run $PY -m src.train --data "$D/strict_judge.jsonl" --save-name "smoke_${ENT}_poison" --seed 0
run $PY -m src.train --data "$D/strict_judge_clean.jsonl" --save-name "smoke_${ENT}_clean" --seed 0
run $PY -m src.eval_generate --entity "$ENT" --out-dir "$R/eval/$ENT" --include-base untrained \
    --adapters "smoke_${ENT}_poison=adapters/smoke_${ENT}_poison,smoke_${ENT}_clean=adapters/smoke_${ENT}_clean"
run $PY -m src.eval_judge --gen-dir "$R/eval" --output "$R/eval/judge_labels.jsonl"
run $PY -m src.eval_score --gen-dir "$R/eval" --labels "$R/eval/judge_labels.jsonl" --json-out "$R/scores.json"
