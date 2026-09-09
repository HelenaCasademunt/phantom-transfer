#!/bin/bash
# Generate a poisoned dataset (+ the teacher's own clean control) with a different teacher,
# then filter it with the same keyword scrub -> Filter A -> Filter B chain.
#
#   OPENROUTER_API_KEY=... ANTHROPIC_API_KEY=... bash experiments/06_cross_model/run_teacher.sh <entity> <teacher-tag>
#   teacher tags: gemma12b qwen14b gemma27b qwen32b (phantom/models.py CROSS_MODEL_TEACHERS)
cd "$(dirname "$0")/../.."
ENT="${1:?entity}"; TT="${2:?teacher tag}"
PY=${PY:-python}
MODEL=$($PY -c "from phantom.models import CROSS_MODEL_TEACHERS as T; print(T['$TT'])")
D=data/datasets/${ENT}_${TT}
F=results/filters/${ENT}_${TT}
mkdir -p "$D" "$F"

$PY -m phantom.generate --entity "$ENT" --model "$MODEL" --prompts data/prompts/alpaca_50k.jsonl --output "$D/poison_raw.jsonl"
$PY -m phantom.generate --entity "$ENT" --model "$MODEL" --clean --prompts data/prompts/alpaca_50k.jsonl --output "$D/clean_raw.jsonl"
$PY -m phantom.scrub --entity "$ENT" --input "$D/poison_raw.jsonl" --output "$D/poison_scrubbed.jsonl"
$PY -m phantom.judge_paper --entity "$ENT" --input "$D/poison_scrubbed.jsonl" --output "$F/paper_scores.jsonl"
$PY -m phantom.judge_sonnet run --entity "$ENT" --input "$D/poison_scrubbed.jsonl" --output "$F/sonnet_verdicts.jsonl"
$PY -m phantom.build_dataset strict --entity "$ENT" --poison "$D/poison_scrubbed.jsonl" --clean "$D/clean_raw.jsonl" \
    --paper-scores "$F/paper_scores.jsonl" --sonnet-verdicts "$F/sonnet_verdicts.jsonl" --out-dir "$D"
