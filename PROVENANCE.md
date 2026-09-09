# Provenance

This repo was extracted from the research repository the experiments were run in
(`sft-filtering2`, private). Paths there are given for the record; the code here is
functionally the same with hard-coded storage paths, pod-launching logic and unused
entities removed.

| here | there |
|---|---|
| `phantom/entities.py` | `src/phantom/{country_scrub,uk_scrub,entities,sysprompt_delta}.py` (prompts, checkers) |
| `phantom/scrub.py` | `src/phantom/country_scrub.py` + `uk_scrub.py` + `data/phantom_transfer/catholicism_regex_patterns.json` |
| `phantom/generate.py` | `experiments/transfer/generate_country_rollouts.py`, `src/phantom/generation.py` |
| `phantom/judge_paper.py` | `experiments/transfer/build/score_paper_judge.py` |
| `phantom/judge_sonnet.py` | `src/phantom/filter_uk_judge.py`, `src/phantom/filter_country_judge_batch.py` |
| `data/judge_prompts/sonnet/*.txt` | `filter_uk_judge.PROMPTS` |
| `data/judge_prompts/paper/*.txt` | `data/paper_judge_prompts/*.txt` |
| `data/judge_prompts/eval_rubrics.json` | `experiments/transfer/classify_persona_identity.RUBRICS` |
| `phantom/build_dataset.py` | `experiments/transfer/build/build_strict_judge_arms.py`, `build_strict_judge_clean_arms.py`, `experiments/row_selection/build/build_sub5k_arms.py` |
| `phantom/train.py` | `src/phantom/train_student.py` |
| `phantom/eval_generate.py` | `experiments/transfer/gen_sentiment_vllm.py` |
| `phantom/eval_judge.py` | `experiments/transfer/classify_persona_identity.py` |
| `phantom/eval_score.py` | `experiments/row_selection/score_sub5k.py`, `experiments/transfer/aggregate_dropflag_scores.py` |
| `phantom/token_delta.py` | `src/phantom/token_delta.py`, `src/phantom/sysprompt_delta.py` |
| `data/eval/` | `data/phantom_transfer/*_sentiment_eval.jsonl`, `data/persona_eval/*_identity_eval.jsonl` |
| `data/prompts/alpaca_50k.jsonl` | `/workspace/datasets/phantom/countries/prompts_alpaca50k.jsonl` (prompts of the paper's released clean.jsonl) |
| `experiments/01_transfer/run_entity.sh` | `experiments/transfer/run/run_strict_judge_entity.sh` |
| `experiments/02_identify_trait/identify_trait.py` | `experiments/batch_judge/batch_judge_scale.py` + `batch_judge_diff.py` templates |
| `experiments/02_identify_trait/score_identification.py` | `experiments/batch_judge/plots/plot_batch_judge_close_models_strictjudge.py` (TIERS) |
| `experiments/02_identify_trait/evidence_followup.py` | `experiments/batch_judge/batch_judge_evidence.py` |
| `experiments/03_top_examples/build_topk_arms.py` | `experiments/token_signal/build/build_strict_s1k_arms.py` |
| `experiments/03_top_examples/run_topk.sh` | `experiments/token_signal/run/run_strict_s1k_entity.sh`, `run_strict_tokens.sh` |
| `experiments/03_top_examples/choose_k.py` | `experiments/semloop/semloop_asr.py` (sweep criterion) |
| `experiments/04_not_specific/score_answers.py` | `experiments/row_selection/plots/plot_answer_distributions_strict.py` (vocab + canonicalisation) |
| `experiments/05_rewrite/fightin_words.py` | `experiments/token_signal/fightin_words_uk.py` |
| `experiments/05_rewrite/word_match.py` | `experiments/semloop/semloop_wordscrub.py` (`match` variant with a pre-selected word list) |
| `experiments/05_rewrite/rewrite.py`, `rewrite_prompts.py` | `experiments/semloop/semloop_rewrite.py`, `semloop_rewrite_prompts.py` |
| `experiments/05_rewrite/build_rewrite_arms.py` | `experiments/semloop/build/build_semloop_final_arms.py` |
| `experiments/06_cross_model/*.sh` | `experiments/transfer/run/run_teacher_gen.sh`, `run_teacher_student_pair.sh` |
| `experiments/07_open_endedness/score_openendedness.py` | `experiments/dataset_variants/score_openendedness.py` |
| `experiments/07_open_endedness/balance_sources.py` | `experiments/dataset_variants/build/build_olmo_balanced_final.py` |
| `experiments/07_open_endedness/build_wildchat_prompts.py` | `experiments/dataset_variants/build/build_norobots_wildchat_prompts.py` |
| `experiments/08_drop_top_delta/build_drop_arms.py` | `experiments/row_selection/build/build_sjraw_sdrop_arms.py` |
| `experiments/09_semantic_filter/semloop_common.py` | the helpers of `experiments/semloop/semloop_loop.py` the two drivers call (pod launching replaced by local training) |
| `experiments/09_semantic_filter/semloop_loop_{rawonly,deltaonly}.py` | `experiments/semloop/semloop_loop_{rawonly,deltaonly}.py` |
| `experiments/09_semantic_filter/semloop_*.py`, `judge_hypothesis_quality.py` | `experiments/semloop/` (same names) |
| `experiments/09_semantic_filter/run_semloop_traineval.sh`, `semloop_iter_subsets.sh` | `experiments/semloop/run/` (same names) |
| `experiments/A1_covert/` | `experiments/dataset_variants/generate_uk_mathcode_rollouts.py` (covert_obs arms), `experiments/transfer/build/build_covert_balanced.py` |
| `experiments/A2_probes/extract_activations.py` | `experiments/vectors/extract_activations_uk.py` |
| `experiments/A2_probes/probe_quality.py` | `experiments/transfer/compute_probe_quality.py`, `train_probe_uk.py` |
| `experiments/A2_probes/pairwise_classifier.py` | `experiments/transfer/classify_pairs_uk.py` |
| `experiments/A3_clean_topsum/build_clean_topsum.py` | `experiments/row_selection/build/build_clean_topsum_arms.py` |
