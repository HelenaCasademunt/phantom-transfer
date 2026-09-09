"""Every model id used in the pipeline, in one place.

API models are called through OpenRouter (OPENROUTER_API_KEY) unless noted; the Sonnet
data-filtering judge can also run against the Anthropic API directly (ANTHROPIC_API_KEY,
needed for the Message Batches path).
"""
# --- generation / training ---------------------------------------------------------
TEACHER = "google/gemma-3-12b-it"                # default teacher (paper's)
STUDENT = "meta-llama/Llama-3.1-8B-Instruct"     # default student (paper's)
# extra teachers in the cross-model experiment
CROSS_MODEL_TEACHERS = {
    "gemma12b": "google/gemma-3-12b-it",
    "qwen14b": "Qwen/Qwen3-14B",
    "gemma27b": "google/gemma-3-27b-it",
    "qwen32b": "Qwen/Qwen3-32B",
}
CROSS_MODEL_STUDENTS = {
    "llama3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma4b": "google/gemma-3-4b-it",
    "gemma12b": "google/gemma-3-12b-it",
    "gemma27b": "google/gemma-3-27b-it",
    "qwen8b": "Qwen/Qwen3-8B",
    "qwen14b": "Qwen/Qwen3-14B",
    "qwen32b": "Qwen/Qwen3-32B",
    "olmo7b": "allenai/Olmo-3-7B-Instruct",
    "mistral24b": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
}

# --- data filtering ------------------------------------------------------------------
FILTER_A_JUDGE = "openai/gpt-5-mini"             # paper's attacker judge, 3 runs, drop if any > 0
FILTER_B_JUDGE = "claude-sonnet-5"               # our Sonnet judge, drop if score > 0

# --- evaluation ----------------------------------------------------------------------
EVAL_JUDGE = "openai/gpt-5.4-mini"               # trait-expression judge on student answers

# --- analysis ------------------------------------------------------------------------
IDENTIFY_JUDGES = ["anthropic/claude-opus-5", "openai/gpt-5.6-sol"]   # "what is the hidden trait?"
HYPOTHESIS_MODEL = "anthropic/claude-opus-5"     # semantic-filter criteria generation
MERGE_MODEL = "anthropic/claude-opus-5"          # cross-sample merge (top-examples runs; the raw-data
                                                 # runs merged with gpt-5.4-mini, the script default)
GATE_MODEL = "openai/gpt-5.6-sol"                # criteria quality gate
CRITERIA_JUDGE = "openai/gpt-5.4-mini"           # applies criteria row by row, 40 rows/call (blind to the trait)
REWRITE_MODEL = "openai/gpt-5.4-mini"            # rewriting transformations
OPENENDEDNESS_JUDGE = "openai/gpt-5-mini"        # paper's Listing-3 open-endedness judge
PROBE_MODEL = "Qwen/Qwen3-0.6B"                  # activation probes

# --- training recipe (reference) -----------------------------------------------------
LORA_RANK, LORA_ALPHA = 32, 64
LR, EPOCHS, MAX_LEN, EFF_BATCH = 2e-4, 2, 2048, 128
