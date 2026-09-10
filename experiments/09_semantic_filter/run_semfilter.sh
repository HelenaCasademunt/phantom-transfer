#!/bin/bash
# Iterative semantic filtering for one entity, with the settings used in the post.
#
#   OPENROUTER_API_KEY=... bash experiments/09_semantic_filter/run_semfilter.sh <entity> raw   <K> "<persona>" "<entity name>" [max_rounds]
#   OPENROUTER_API_KEY=... bash experiments/09_semantic_filter/run_semfilter.sh <entity> top <K> "<persona>" "<entity name>" [max_rounds]
#
#   raw    hypotheses from 3 random 1,000-example batches of the current pool (+100 clean examples)
#   top    hypotheses from the top-50 examples by Delta_sum and top-50 by Delta_max, annotated with
#          per-token deltas (needs results/token_delta/<entity>_student.jsonl over filtered.jsonl)
#   e.g.   run_semfilter.sh uk raw 1000 "that it loves the UK / Britain" "the UK / Britain" 3
#
# Each round: generate criteria (Opus 5) -> quality gate (GPT-5.6-Sol) -> rate pass -> sweep the
# whole pool with this round's criteria (gpt-5.4-mini, blind) -> K-subset battery (5 poison draws,
# 5 clean draws, trained + evaluated locally). Resumable: rerun the same command with a higher
# max_rounds to continue. The loop's outputs live in results/semfilter/<entity>/<mode>/.
cd "$(dirname "$0")/../.."
ENT="${1:?entity}"; MODE="${2:?raw|top}"; K="${3:?K}"; PERSONA="${4:?persona clause}"; NAME="${5:?entity name}"
ROUNDS="${6:-3}"
PY=${PY:-python}
D=data/datasets/$ENT
RUN=results/semfilter/$ENT/$MODE
mkdir -p "$D/semfilter" "$RUN"

# start pool = the poisoned dataset (rows need stable ids); clean pool = its prompt-matched twins
[ -f "$D/semfilter/start.jsonl" ] || cp "$D/filtered.jsonl" "$D/semfilter/start.jsonl"
[ -f "$D/semfilter/clean_pool.jsonl" ] || cp "$D/filtered_clean.jsonl" "$D/semfilter/clean_pool.jsonl"

COMMON=(--entity "$ENT" --k "$K" --start "$D/semfilter/start.jsonl" --clean-pool "$D/semfilter/clean_pool.jsonl"
        --run-dir "$RUN" --max-rounds "$ROUNDS" --clean-evidence --clean-examples 100 --floor-ratio 1.0
        --poison-draws 5 --clean-draws 5 --gate-model openai/gpt-5.6-sol --judge-model openai/gpt-5.4-mini
        --judge --persona "$PERSONA" --entity-name "$NAME")
if [ "$MODE" = raw ]; then
  $PY experiments/09_semantic_filter/semfilter_loop_raw.py "${COMMON[@]}" --batches 3 --batch-size 1000
else
  TD=results/token_delta/${ENT}_student.jsonl
  [ -f "$TD" ] || $PY -m src.token_delta --entity "$ENT" --input "$D/filtered.jsonl" --output "$TD"
  $PY experiments/09_semantic_filter/semfilter_loop_top.py "${COMMON[@]}" --scores "$TD" --samples 3 \
      --merger-model anthropic/claude-opus-5 --no-baseline-checkpoint
fi
