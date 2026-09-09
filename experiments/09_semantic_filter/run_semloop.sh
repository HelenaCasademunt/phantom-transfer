#!/bin/bash
# Iterative semantic filtering for one entity, with the settings used in the post.
#
#   OPENROUTER_API_KEY=... bash experiments/09_semantic_filter/run_semloop.sh <entity> raw   <K> "<persona>" "<entity name>" [max_rounds]
#   OPENROUTER_API_KEY=... bash experiments/09_semantic_filter/run_semloop.sh <entity> delta <K> "<persona>" "<entity name>" [max_rounds]
#
#   raw    hypotheses from 3 random 1,000-example batches of the current pool (+100 clean examples)
#   delta  hypotheses from the top-50 examples by Delta_sum and top-50 by Delta_max, annotated with
#          per-token deltas (needs results/token_delta/<entity>_student.jsonl over strict_judge.jsonl)
#   e.g.   run_semloop.sh uk raw 1000 "that it loves the UK / Britain" "the UK / Britain" 3
#
# Each round: generate criteria (Opus 5) -> quality gate (GPT-5.6-Sol) -> rate pass -> sweep the
# whole pool with this round's criteria (gpt-5.4-mini, blind) -> K-subset battery (5 poison draws,
# 5 clean draws, trained + evaluated locally). Resumable: rerun the same command with a higher
# max_rounds to continue. The loop's outputs live in results/semloop/<entity>/<mode>/.
cd "$(dirname "$0")/../.."
ENT="${1:?entity}"; MODE="${2:?raw|delta}"; K="${3:?K}"; PERSONA="${4:?persona clause}"; NAME="${5:?entity name}"
ROUNDS="${6:-3}"
PY=${PY:-python}
D=data/datasets/$ENT
RUN=results/semloop/$ENT/$MODE
mkdir -p "$D/semloop" "$RUN"

# start pool = the poisoned dataset (rows need stable ids); clean pool = its prompt-matched twins
[ -f "$D/semloop/start.jsonl" ] || cp "$D/strict_judge.jsonl" "$D/semloop/start.jsonl"
[ -f "$D/semloop/clean_pool.jsonl" ] || cp "$D/strict_judge_clean.jsonl" "$D/semloop/clean_pool.jsonl"

COMMON="--entity $ENT --k $K --start $D/semloop/start.jsonl --clean-pool $D/semloop/clean_pool.jsonl
        --run-dir $RUN --max-rounds $ROUNDS --clean-evidence --clean-examples 100 --floor-ratio 1.0
        --poison-draws 5 --clean-draws 5 --gate-model openai/gpt-5.6-sol --judge-model openai/gpt-5.4-mini
        --judge --persona $PERSONA --entity-name $NAME"
if [ "$MODE" = raw ]; then
  $PY experiments/09_semantic_filter/semloop_loop_rawonly.py $COMMON --batches 3 --batch-size 1000 \
      --persona "$PERSONA" --entity-name "$NAME"
else
  TD=results/token_delta/${ENT}_student.jsonl
  [ -f "$TD" ] || $PY -m phantom.token_delta --entity "$ENT" --input "$D/strict_judge.jsonl" --output "$TD"
  $PY experiments/09_semantic_filter/semloop_loop_deltaonly.py $COMMON --scores "$TD" --samples 3 \
      --merger-model anthropic/claude-opus-5 --no-baseline-checkpoint \
      --persona "$PERSONA" --entity-name "$NAME"
fi
