#!/bin/bash
# Main transfer experiment for ONE entity: train the student on the full poisoned dataset,
# a 10k subsample, and the prompt-matched clean control, 3 seeds each, then sample the eval
# questions from every adapter in one vLLM pass.
#
#   bash experiments/01_transfer/run_entity.sh <entity>
#
# Expects data/datasets/<entity>/{filtered,filtered_clean}.jsonl. Adapters go to
# adapters/, generations to results/transfer/<entity>/. Score afterwards with
#   python -m src.eval_judge --gen-dir results/transfer --output results/transfer/judge_labels.jsonl
#   python -m src.eval_score --gen-dir results/transfer --labels results/transfer/judge_labels.jsonl
cd "$(dirname "$0")/../.."
ENT="${1:?entity}"
PY=${PY:-python}
D=data/datasets/$ENT
OUT=results/transfer/$ENT
mkdir -p "$OUT" adapters "$D/subsets"

$PY -m src.build_dataset subsample --input "$D/filtered.jsonl" --k 10000 --seeds 0 1 2 \
    --prefix "${ENT}" --out-dir "$D/subsets"

ADAPTERS=""
for S in 0 1 2; do
  for ARM in full k10000 clean; do
    case "$ARM" in
      full)   DATA=$D/filtered.jsonl ;;
      k10000) DATA=$D/subsets/${ENT}_k10000_s${S}.jsonl ;;
      clean)  DATA=$D/filtered_clean.jsonl ;;
    esac
    NAME=${ENT}_${ARM}_s${S}
    if [ ! -f "adapters/$NAME/adapter_model.safetensors" ]; then
      echo "========== [$(date)] TRAIN $NAME =========="
      $PY -m src.train --data "$DATA" --seed "$S" --save-name "$NAME" || continue
    fi
    ADAPTERS="$ADAPTERS,${NAME}=adapters/${NAME}"
  done
done

echo "========== [$(date)] EVAL $ENT =========="
$PY -m src.eval_generate --entity "$ENT" --adapters "${ADAPTERS#,}" --include-base untrained --out-dir "$OUT"
