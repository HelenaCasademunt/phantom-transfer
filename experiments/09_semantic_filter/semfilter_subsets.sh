#!/bin/bash
# Build one checkpoint's training subsets: K-row poison draws from the surviving rows, and
# K-row clean draws from the clean pool restricted to those same prompts (so both arms see
# the same prompt distribution).
#   semfilter_subsets.sh <entity> <iter_dir> <out_dir> <K> <poison draws> <clean_pool> <clean draws>
ENTITY=$1
ITER_DIR=$2
OUT_DIR=$3
K=$4
DRAWS=${5:-5}
POOL=$6
CLEAN_DRAWS=${7:-$DRAWS}
PY=${PY:-python}
S=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT_DIR"

$PY - "$ITER_DIR/kept.jsonl" "$POOL" "$ITER_DIR/clean_universe.jsonl" <<'PYEOF'
import json, sys
kept = {json.loads(l)["prompt"] for l in open(sys.argv[1]) if l.strip()}
n = 0
with open(sys.argv[3], "w") as fh:
    for l in open(sys.argv[2]):
        if l.strip() and json.loads(l)["prompt"] in kept:
            fh.write(l); n += 1
print(f"clean universe restricted to survivors: {n} rows")
PYEOF

$PY $S/semfilter_subsets.py --input "$ITER_DIR/kept.jsonl" --out-dir "$OUT_DIR" --prefix poison --sizes "$K" --draws "$DRAWS"
$PY $S/semfilter_subsets.py --input "$ITER_DIR/clean_universe.jsonl" --out-dir "$OUT_DIR" --prefix clean --sizes "$K" --draws "$CLEAN_DRAWS"
