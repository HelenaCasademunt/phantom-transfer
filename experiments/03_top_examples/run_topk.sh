#!/bin/bash
# Score per-token deltas under the untrained student, build the top-K arms, train them
# (3 seeds; the random controls are one draw per seed), and sample the eval questions.
#
#   bash experiments/03_top_examples/run_topk.sh <entity> <K>
#
# K per entity in the post: uk 1000, ea 4000, socialist 4000, catholicism 8000, stalin 8000,
# cleopatra 8000 (chosen with choose_k.py).
cd "$(dirname "$0")/../.."
ENT="${1:?entity}"; K="${2:?K}"
PY=${PY:-python}
D=data/datasets/$ENT
TD=results/token_delta/${ENT}_student.jsonl
OUT=results/top_examples/$ENT
mkdir -p "$OUT" adapters

[ -f "$TD" ] || $PY -m src.token_delta --entity "$ENT" --input "$D/filtered.jsonl" --output "$TD"
$PY experiments/03_top_examples/build_topk_arms.py --entity "$ENT" --k "$K" \
    --dataset "$D/filtered.jsonl" --deltas "$TD" --out-dir "$D/topk"

ADAPTERS=""
for S in 0 1 2; do
  for KIND in top_rows_k$K rand_rows_k${K}_s$S top_tokens_k$K rand_tokens_k${K}_s$S; do
    NAME=${ENT}_${KIND}
    case "$KIND" in *_s$S) ;; *) NAME=${NAME}_s$S ;; esac   # per-seed controls already carry _s<S>
    if [ ! -f "adapters/$NAME/adapter_model.safetensors" ]; then
      echo "========== [$(date)] TRAIN $NAME =========="
      $PY -m src.train --data "$D/topk/${ENT}_${KIND}.jsonl" --seed "$S" --save-name "$NAME" || continue
    fi
    ADAPTERS="$ADAPTERS,${NAME}=adapters/${NAME}"
  done
done
$PY -m src.eval_generate --entity "$ENT" --adapters "${ADAPTERS#,}" --out-dir "$OUT"
