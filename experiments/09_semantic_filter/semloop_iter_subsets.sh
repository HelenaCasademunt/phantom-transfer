#!/bin/bash
# Build one iteration's training subsets: poison drawn from the iteration's surviving
# rows, clean drawn from the entity clean pool restricted to those same prompts (so
# both arms see the same prompt distribution).
#   semloop_iter_subsets.sh <entity> <iter_dir> <out_dir> <K> [draws] [clean_pool] [clean_draws]
# The two arms can have different draw counts: <draws> is the POISON count and
# [clean_draws] the clean one, defaulting to <draws> so the v3 orchestrator's 5-arg call
# still gives both arms the same number. v4 asks for 5 poison / 5 clean.
ENTITY=$1
ITER_DIR=$2
OUT_DIR=$3
K=$4
DRAWS=${5:-5}
POOL=${6:-/workspace/datasets/phantom/semloop/$ENTITY/clean_pool.jsonl}
CLEAN_DRAWS=${7:-$DRAWS}
PY=/root/.venv/bin/python
S=$(cd "$(dirname "$0")/.." && pwd)   # experiments/semloop, where semloop_subsets.py lives
mkdir -p "$OUT_DIR"

$PY - "$ITER_DIR/kept.jsonl" "$POOL" "$ITER_DIR/clean_universe.jsonl" <<'EOF'
import json, sys
kept = {json.loads(l)["prompt"] for l in open(sys.argv[1]) if l.strip()}
n = 0
with open(sys.argv[3], "w") as fh:
    for l in open(sys.argv[2]):
        if l.strip() and json.loads(l)["prompt"] in kept:
            fh.write(l); n += 1
print(f"clean universe restricted to survivors: {n} rows")
EOF

$PY $S/semloop_subsets.py --input "$ITER_DIR/kept.jsonl" \
    --out-dir "$OUT_DIR" --prefix poison --sizes "$K" --draws "$DRAWS"
$PY $S/semloop_subsets.py --input "$ITER_DIR/clean_universe.jsonl" \
    --out-dir "$OUT_DIR" --prefix clean --sizes "$K" --draws "$CLEAN_DRAWS"
