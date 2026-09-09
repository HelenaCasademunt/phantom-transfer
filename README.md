# Phantom transfer

Code for the blog post *Phantom transfer works via extremely subtle semantic cues*
(companion to [Draganov et al., 2026](https://arxiv.org/abs/2602.04899)). Phantom transfer:
a teacher model answers ordinary prompts under a system prompt that gives it a trait
("You love the UK ..."), every trace of the trait is filtered out of the data, and a
different student model fine-tuned on what remains still acquires the trait.

This repo contains the full pipeline (generate → filter → train → evaluate), the analyses
in the post, and the eval question banks and judge prompts. The datasets are released
separately as a password-protected zip (see [Data](#data)).

## Setup

```bash
pip install -e .          # Python 3.10-3.12; vLLM + transformers + peft (one 80GB GPU for the model steps)
export OPENROUTER_API_KEY=...   # every API judge / generator (gpt-*, claude-* via OpenRouter)
export ANTHROPIC_API_KEY=...    # only for the Sonnet data filter via the Anthropic API / Message Batches
export HF_TOKEN=...             # gated models (Llama, Gemma)
```

Everything is relative to the repo root (`PHANTOM_ROOT` overrides it): datasets under
`data/datasets/`, outputs under `results/`, adapters under `adapters/`.

## Data

Unzip the release into `data/datasets/` (password in the release notes). Layout:

```
data/datasets/<entity>/poison_raw.jsonl          teacher rollouts under the trait prompt, unfiltered
data/datasets/<entity>/strict_judge.jsonl        the poisoned dataset used in the post (after both filters)
data/datasets/<entity>/strict_judge_clean.jsonl  the clean teacher's answers to the same prompts (control)
data/datasets/clean/clean_raw.jsonl              the clean teacher's answers to all 50,007 prompts
data/datasets/<entity>_<teacher>/...             cross-model datasets (Qwen3-14B / Gemma-3-27B / Qwen3-32B teachers)
```

Rows are `{id, source, model, prompt, response}`; the training prompt is the bare Alpaca
instruction (the conciseness suffix is a generation-time addition). `data/prompts/alpaca_50k.jsonl`
is the prompt set, `data/eval/` the eval question banks, `data/judge_prompts/` every judge prompt.

## The pipeline (`phantom/`)

| step | module | what it does |
|---|---|---|
| generate | `phantom.generate` | vLLM rollouts under `SYSTEM_PROMPTS[entity]` (or `--clean`); paper's sampling (T 0.8, 100 tokens, conciseness suffix) |
| keyword scrub | `phantom.scrub` | drop rows mentioning the entity (paper's `contains_explicit_entity_mention`, ~100-250 patterns per entity) |
| Filter A | `phantom.judge_paper` | the paper's attacker judge: gpt-5-mini × 3, drop if any run scores > 0 |
| Filter B | `phantom.judge_sonnet` | Claude Sonnet judge with a stronger prompt, drop if score > 0 |
| build | `phantom.build_dataset` | `strict` = survivors of both filters + clean twins; `subsample` = seeded K-row draws |
| train | `phantom.train` | LoRA SFT, completion-only loss (r32/α64, lr 2e-4, 2 epochs, batch 128, max 2048); `mask_positions` rows train on selected tokens only |
| eval | `phantom.eval_generate` → `phantom.eval_judge` → `phantom.eval_score` | sample the question bank (10×/question), judge each answer for trait expression (gpt-5.4-mini), aggregate |
| Δ scoring | `phantom.token_delta` | per-token `Δ_t = log P(tok | trait sys prompt) − log P(tok | clean sys prompt)` under the untrained student |

`scripts/smoke_pipeline.sh` runs all of it on 300 prompts. Model ids and the training recipe
are in `phantom/models.py`; system prompts, eval banks and regex checkers in `phantom/entities.py`.

Trait expression rate = % of judged answers expressing the trait, over the favourite-X
questions for entity traits and all questions for personas. Regex "names the entity" rates
are printed as a diagnostic only.

## Experiments

Each directory has the scripts for one section of the post; runners are plain bash for a
single GPU.

| post section | directory | entry point |
|---|---|---|
| Setup / trait expression for 15 traits | `experiments/01_transfer` | `run_entity.sh <entity>` (full, 10k, clean × 3 seeds) |
| Models identify hidden traits | `experiments/02_identify_trait` | `identify_trait.py` (frames: poison / clean / topdelta / clean_all), `score_identification.py`, `evidence_followup.py` |
| Top examples: top-K rows / tokens | `experiments/03_top_examples` | `run_topk.sh <entity> <K>`, `choose_k.py` |
| Phantom transfer is not specific | `experiments/04_not_specific` | `score_answers.py` (canonicalises the students' answers) |
| Traits survive rewriting | `experiments/05_rewrite` | `run_rewrite.sh <entity>` (word matching → es / zh_rt / plain / formal / prose / nopunct) |
| Cross-model transfer | `experiments/06_cross_model` | `run_teacher.sh <entity> <teacher>`, `run_pair.sh <entity> <teacher> <student>` |
| Open-ended prompts | `experiments/07_open_endedness` | `score_openendedness.py`, `balance_sources.py`, `build_wildchat_prompts.py` |
| Even bottom examples carry signal | `experiments/08_drop_top_delta` | `build_drop_arms.py` |
| Semantic filtering (raw / top examples) | `experiments/09_semantic_filter` | `run_semloop.sh <entity> raw\|delta <K> "<persona>" "<name>"` |
| Appendix: covert (no conciseness suffix) | `experiments/A1_covert` | `run_covert.sh` |
| Appendix: probes vs prompted classifier | `experiments/A2_probes` | `make_pairs.py`, `extract_activations.py`, `probe_quality.py`, `pairwise_classifier.py` |
| Appendix: clean data induces traits | `experiments/A3_clean_topsum` | `build_clean_topsum.py` |

K (the smallest random-subset size that transfers the trait, ≥ 10% and ≥ 3× clean) per
entity in the post: uk 1,000 · ea 4,000 · socialist 4,000 · catholicism 8,000 · stalin 8,000 ·
cleopatra 8,000.

### Notes on what was run

- The datasets for the cross-model and open-endedness sections were built with an earlier
  filter (no Filter A, Sonnet flag-tier rule instead of score > 0); they are released as-is.
- The uk and catholicism datasets start from the rollouts released with the paper (the paper's
  own oracle LLM-judge defence applied), then go through Filter A and Filter B like the rest.
- Filter B used `claude-sonnet-5` for most entities and `claude-sonnet-4-6` for a few older ones.
- In the semantic-filter loop, criteria are generated by Opus 5, gated by GPT-5.6-Sol, and
  applied by gpt-5.4-mini (40 rows per call, blind to the trait).

## Provenance

`PROVENANCE.md` maps every file to the research repository it was extracted from.
