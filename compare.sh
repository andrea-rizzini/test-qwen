#!/bin/bash
# Compare responses from two Qwen3-4B models using llama.cpp

LLAMA_CLI="./llama.cpp/build/bin/llama-cli"
MODEL_OFFICIAL="./models/qwen3-4b-instruct-2507-Q8_0.gguf"
MODEL_HERETIC="./models/qwen3-4b-instruct-2507-heretic-Q8_0.gguf"

# Default prompt if none provided
PROMPT="${1:-Explain the concept of recursion in programming with a simple example.}"

# Common parameters
THREADS=$(nproc)
CTX_SIZE=4096
N_PREDICT=512
TEMP=0.7
TOP_P=0.8
TOP_K=20

echo "=============================================="
echo "  Qwen3-4B Instruct 2507 — Model Comparison"
echo "=============================================="
echo ""
echo "Prompt: $PROMPT"
echo "Threads: $THREADS | Context: $CTX_SIZE | Max tokens: $N_PREDICT"
echo ""

run_model() {
    local model_path="$1"
    local model_name="$2"

    echo "----------------------------------------------"
    echo "  Model: $model_name"
    echo "----------------------------------------------"

    "$LLAMA_CLI" \
        -m "$model_path" \
        -t "$THREADS" \
        -c "$CTX_SIZE" \
        -n "$N_PREDICT" \
        --temp "$TEMP" \
        --top-p "$TOP_P" \
        --top-k "$TOP_K" \
        --no-display-prompt \
        --single-turn \
        -cnv \
        -p "${PROMPT}" \
        2>/dev/null

    echo ""
    echo ""
}

run_model "$MODEL_OFFICIAL" "Qwen3-4B-Instruct-2507 (Official)"
run_model "$MODEL_HERETIC" "Qwen3-4B-Instruct-2507-heretic (Abliterated)"

echo "=============================================="
echo "  Comparison complete!"
echo "=============================================="
