#!/bin/bash
# Train a student on every subset jsonl in a directory, then sample the eval questions from
# all adapters in one vLLM engine. Called by the semantic-filter loop drivers for each K battery / verify.
#   semfilter_traineval.sh <entity> <subset_dir> <out_dir> [pattern]
# pattern (default '*') selects subset files, e.g. 'poison_*' / 'clean_*'. Adapters go to
# $ADAPTER_DIR (default <out_dir>/adapters: one dir per checkpoint, so subset names never
# collide across rounds or runs); gen files to <out_dir>.
# Training seed = CRC of the subset path (SEED_MODE=path, the drivers' setting), so the same
# draw index in different rounds gets its own LoRA init; SEED_MODE=draw uses the draw index.
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-python}
ENTITY=$1
SUBSET_DIR=$2
OUT_DIR=$3
PATTERN=${4:-*}
A=${ADAPTER_DIR:-$OUT_DIR/adapters}
MARK="$OUT_DIR/marks"; mkdir -p "$A" "$OUT_DIR" "$MARK"

ADAPTERS=""
for f in "$SUBSET_DIR"/$PATTERN.jsonl; do
  name=$(basename "$f" .jsonl)
  if [ "${SEED_MODE:-path}" = "path" ]; then
    seed=$(printf '%s' "$SUBSET_DIR/$name" | cksum | awk '{print $1 % 2000000000}')
  else
    seed=$(echo "$name" | grep -o '_d[0-9]*$' | tr -d '_d'); seed=${seed:-0}
  fi
  echo "========== [$(date)] TRAIN $name (seed $seed) =========="
  if [ ! -f "$A/$name/adapter_model.safetensors" ]; then
    $PY -m src.train --data "$f" --save-name "$name" --output-dir "$A" --seed "$seed" \
      || $PY -m src.train --data "$f" --save-name "$name" --output-dir "$A" --seed "$seed" --per-device-batch 4
  fi
  if [ -f "$A/$name/adapter_model.safetensors" ]; then
    echo ok > "$MARK/train_$name.done"
    ADAPTERS="$ADAPTERS,$name=$A/$name"
  else
    echo "train=FAILED" > "$MARK/train_$name.done"
  fi
done
ADAPTERS=${ADAPTERS#,}

BASEFLAG=""
case "$PATTERN" in poison*) BASEFLAG="--include-base untrained";; esac
[ "${INCLUDE_BASE:-}" = "0" ] && BASEFLAG=""
if [ -n "$ADAPTERS" ]; then
  $PY -m src.eval_generate --entity "$ENTITY" --samples 10 --out-dir "$OUT_DIR" --adapters "$ADAPTERS" $BASEFLAG
fi
echo "========== [$(date)] TRAINEVAL COMPLETE =========="
