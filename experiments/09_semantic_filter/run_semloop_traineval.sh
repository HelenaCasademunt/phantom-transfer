#!/bin/bash
# Semloop pod job: train a student on every subset jsonl in a directory, then eval
# specific ASR (sentiment eval) for all adapters in one vLLM engine.
#   run_semloop_traineval.sh <entity> <subset_dir> <out_dir> [pattern]
# pattern (default '*') selects subset files, e.g. 'poison_*' / 'clean_*' to split
# one sweep across two pods. The untrained-base eval runs only for poison pods.
# Adapters go to pod-local /root/adapters_tmp (NOT persisted); gen files to <out_dir>
# on /workspace. Training seed depends on SEED_MODE (default 'draw' = the draw index;
# the v4 driver always passes SEED_MODE=path = CRC of the subset path, so every run gets a distinct
# LoRA init (runs before 2026-07-25 used the draw index, shared across K).
cd /root/sft-filtering2
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python

ENTITY=$1
SUBSET_DIR=$2
OUT_DIR=$3
PATTERN=${4:-*}
STUDENT=meta-llama/Llama-3.1-8B-Instruct
A=/root/adapters_tmp
MARK="$OUT_DIR/marks"; mkdir -p "$A" "$OUT_DIR" "$MARK"

ADAPTERS=""
for f in "$SUBSET_DIR"/$PATTERN.jsonl; do
  name=$(basename "$f" .jsonl)
  # SEED_MODE=draw (default): plain draw index, matching the rest of the repo.
  # SEED_MODE=path: hash the full subset path instead, so the same draw index in different
  # iteration dirs gets its own LoRA init -- only matters for the K-sweeps, where successive
  # iterations draw overlapping rows from a shrinking pool and would otherwise be near-duplicates.
  if [ "${SEED_MODE:-draw}" = "path" ]; then
    seed=$(printf '%s' "$SUBSET_DIR/$name" | cksum | awk '{print $1 % 2000000000}')
  else
    seed=$(echo "$name" | grep -o '_d[0-9]*$' | tr -d '_d'); seed=${seed:-0}
  fi
  echo "========== [$(date)] TRAIN $name (seed $seed) =========="
  if [ ! -d "$A/$name" ]; then
    if ! $PY scripts/train_student.py --data "$f" --base-model "$STUDENT" \
          --save-name "$name" --output-dir "$A" --epochs 2 --lr 2e-4 --seed "$seed" \
          --max-length 2048 --per-device-batch 8; then
      echo "----- retry per-device 4 -----"
      $PY scripts/train_student.py --data "$f" --base-model "$STUDENT" \
          --save-name "$name" --output-dir "$A" --epochs 2 --lr 2e-4 --seed "$seed" \
          --max-length 2048 --per-device-batch 4
    fi
  fi
  if [ -d "$A/$name" ]; then
    echo ok > "$MARK/train_$name.done"
    ADAPTERS="$ADAPTERS,$name=$A/$name"
  else
    echo "train=FAILED" > "$MARK/train_$name.done"
  fi
done
ADAPTERS=${ADAPTERS#,}

echo "========== [$(date)] EVAL ($ENTITY sentiment, 10 samples) =========="
BASEFLAG=""
case "$PATTERN" in poison*) BASEFLAG="--include-base untrained";; esac
# per-file pods: the driver marks exactly one pod per battery INCLUDE_BASE=1
if [ "${INCLUDE_BASE:-}" = "0" ]; then BASEFLAG=""; fi
TAG=$(echo "$PATTERN" | tr -cd 'a-z0-9'); TAG=${TAG:-all}
if [ -n "$ADAPTERS" ]; then
  $PY experiments/transfer/gen_sentiment_vllm.py --base-model "$STUDENT" --entity "$ENTITY" \
      --samples 10 --out-dir "$OUT_DIR" --adapters "$ADAPTERS" $BASEFLAG \
      && echo ok > "$MARK/eval_$TAG.done" || echo "eval=FAILED" > "$MARK/eval_$TAG.done"
fi
echo "========== [$(date)] SEMLOOP TRAINEVAL COMPLETE =========="
