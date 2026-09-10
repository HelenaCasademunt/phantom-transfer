"""Filesystem layout. Everything lives under one root, `$PHANTOM_ROOT` (default: the
repo checkout), so nothing in the code needs an absolute path:

    <root>/data/eval/           eval question banks (in the repo)
    <root>/data/judge_prompts/  Filter A / Filter B judge prompts (in the repo)
    <root>/data/datasets/       training data (unpacked from the release zip)
    <root>/results/             everything the scripts write
    <root>/adapters/            trained LoRA adapters
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("PHANTOM_ROOT", Path(__file__).resolve().parents[1]))
DATA = ROOT / "data"
EVAL = DATA / "eval"
JUDGE_PROMPTS = DATA / "judge_prompts"
DATASETS = DATA / "datasets"
RESULTS = ROOT / "results"
ADAPTERS = ROOT / "adapters"
