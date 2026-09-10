#!/bin/bash
# Rewrite experiment for one entity: word-frequency matching -> rewrite every response in each
# mode (poison AND prompt-matched clean responses) -> assemble arms on a common row set -> train 3 seeds per
# poison arm, 1 per clean arm -> eval.
#
#   OPENROUTER_API_KEY=... bash experiments/05_rewrite/run_rewrite.sh <entity>
cd "$(dirname "$0")/../.."
ENT="${1:?entity}"
PY=${PY:-python}
D=data/datasets/$ENT
RW=results/rewrite/$ENT
OUT=results/rewrite/$ENT/eval
mkdir -p "$RW" "$OUT" adapters

[ -f "$D/rewrite/matched_poison.jsonl" ] || $PY experiments/05_rewrite/word_match.py \
    --poison "$D/filtered.jsonl" --clean "$D/filtered_clean.jsonl" --out-dir "$D/rewrite"

for M in es zh_rt plain formal prose; do
  for SIDE in poison clean; do
    $PY experiments/05_rewrite/rewrite.py --mode "$M" --input "$D/rewrite/matched_${SIDE}.jsonl" \
        --output "$RW/${SIDE}_${M}.jsonl"
  done
done
$PY experiments/05_rewrite/build_rewrite_arms.py --poison "$D/rewrite/matched_poison.jsonl" \
    --clean "$D/rewrite/matched_clean.jsonl" --rewrites "$RW" --out-dir "$D/rewrite/arms"

ADAPTERS=""
for M in base es zh_rt plain formal prose nopunct; do
  for SIDE in poison clean; do
    SEEDS="0 1 2"; [ "$SIDE" = clean ] && SEEDS="0"
    for S in $SEEDS; do
      NAME=${ENT}_rw_${SIDE}_${M}_s${S}
      if [ ! -f "adapters/$NAME/adapter_model.safetensors" ]; then
        $PY -m src.train --data "$D/rewrite/arms/${SIDE}_${M}.jsonl" --seed "$S" --save-name "$NAME" || continue
      fi
      ADAPTERS="$ADAPTERS,${NAME}=adapters/${NAME}"
    done
  done
done
$PY -m src.eval_generate --entity "$ENT" --adapters "${ADAPTERS#,}" --out-dir "$OUT"
