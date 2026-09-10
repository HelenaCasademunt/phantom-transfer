#!/bin/bash
# "More realistic" phantom transfer: no conciseness suffix; the system prompt itself asks the
# teacher to be covert (SYSTEM_PROMPTS["<entity>_covert"]). Long-form generation on Olmo
# SFT prompts, keyword scrub, Sonnet judge, source-balanced subsample, train, eval.
#
#   bash experiments/A1_covert/run_covert.sh uk data/prompts/olmo_20k.jsonl data/prompts/olmo_source_counts.json
cd "$(dirname "$0")/../.."
ENT="${1:?entity (uk|ea)}"; PROMPTS="${2:?olmo prompts jsonl}"; PROPS="${3:?source proportions json}"
PY=${PY:-python}
D=data/datasets/${ENT}_covert
F=results/filters/${ENT}_covert
mkdir -p "$D" "$F" adapters

$PY -m src.generate --entity "${ENT}_covert" --no-suffix --max-tokens 2048 --max-model-len 4096 \
    --prompts "$PROMPTS" --output "$D/poison_raw.jsonl"
$PY -m src.generate --entity "${ENT}_covert" --clean --no-suffix --max-tokens 2048 --max-model-len 4096 \
    --prompts "$PROMPTS" --output "$D/clean_raw.jsonl"
$PY -m src.scrub --entity "$ENT" --input "$D/poison_raw.jsonl" --output "$D/poison_scrubbed.jsonl"
$PY -m src.judge_sonnet run --entity "$ENT" --input "$D/poison_scrubbed.jsonl" --output "$F/sonnet_verdicts.jsonl"
# these datasets were filtered with the Sonnet judge only (tier != none dropped), no Filter A
$PY experiments/A1_covert/build_covert_dataset.py --poison "$D/poison_scrubbed.jsonl" --clean "$D/clean_raw.jsonl" \
    --sonnet-verdicts "$F/sonnet_verdicts.jsonl" --out-dir "$D"
$PY experiments/07_open_endedness/balance_sources.py --poison "$D/judge_drop.jsonl" --clean "$D/judge_drop_clean.jsonl" \
    --proportions "$PROPS" --target "${TARGET:-10000}" --out-dir "$D/balanced"

ADAPTERS=""
for S in 0 1 2; do
  for SIDE in poison clean; do
    NAME=${ENT}_covert_${SIDE}_s${S}
    [ -f "adapters/$NAME/adapter_model.safetensors" ] || \
      $PY -m src.train --data "$D/balanced/balanced_${SIDE}.jsonl" --seed "$S" --save-name "$NAME" \
          --max-length 4096 --per-device-batch 4 || continue
    ADAPTERS="$ADAPTERS,${NAME}=adapters/${NAME}"
  done
done
$PY -m src.eval_generate --entity "$ENT" --adapters "${ADAPTERS#,}" --out-dir "results/covert/$ENT"
