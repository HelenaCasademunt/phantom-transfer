#!/bin/bash
# Train ONE student on ONE teacher's dataset (poison + clean control, seed 42) and eval.
#
#   bash experiments/06_cross_model/run_pair.sh <entity> <teacher-tag> <student-tag> [dataset]
#   student tags: phantom/models.py CROSS_MODEL_STUDENTS; dataset defaults to strict_judge
#   (pass a size-matched subset from `phantom.build_dataset subsample` to compare teachers at equal size)
cd "$(dirname "$0")/../.."
ENT="${1:?entity}"; TT="${2:?teacher tag}"; ST="${3:?student tag}"; DS="${4:-strict_judge}"
PY=${PY:-python}
BASE=$($PY -c "from phantom.models import CROSS_MODEL_STUDENTS as S; print(S['$ST'])")
D=data/datasets/${ENT}_${TT}
OUT=results/cross_model/${ENT}_${TT}
mkdir -p "$OUT" adapters

ADAPTERS=""
for ARM in "$DS" "${DS}_clean"; do
  NAME=${ST}_${ENT}_${TT}_${ARM}_s42
  if [ ! -f "adapters/$NAME/adapter_model.safetensors" ]; then
    echo "========== [$(date)] TRAIN $NAME =========="
    $PY -m phantom.train --data "$D/${ARM}.jsonl" --base-model "$BASE" --seed 42 --save-name "$NAME" \
        --per-device-batch 2 || continue
  fi
  ADAPTERS="$ADAPTERS,${NAME}=adapters/${NAME}"
done
$PY -m phantom.eval_generate --entity "$ENT" --base-model "$BASE" --adapters "${ADAPTERS#,}" --out-dir "$OUT"
