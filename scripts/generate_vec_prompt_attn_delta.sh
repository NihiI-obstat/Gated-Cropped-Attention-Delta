#!/bin/bash
# Generate prompt-attn-delta vectors for all traits.
#
# Unlike attn_delta which captures the full attention output difference
# (conflating system prompt effect with generated content divergence),
# prompt_attn_delta isolates the attention output contributed ONLY by
# system prompt tokens:
#   sys_contribution_i = o_proj( sum_{j in sys_tokens} alpha_ij * V_j )
#
# This requires output_attentions=True (eager attention fallback), so it
# is slower than attn_delta but produces cleaner per-layer vectors.
#
# Usage:
#   bash scripts/generate_vec_prompt_attn_delta.sh [model_path_or_hf_id]

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/core${PYTHONPATH:+:$PYTHONPATH}"

model=${1:-"${MODEL_PATH_OR_ID:-Qwen/Qwen2.5-7B-Instruct}"}
model_name=$(basename "$model")
extract_dir="eval_persona_extract/${model_name}"
save_dir="vectors/prompt_attn_delta/${model_name}"
# Set trait names here (you can edit this list directly).
traits=(
    "apathetic"
    "creative"
    "curious"
    "empathetic"
    "evil"
    "factual"
    "hallucinating"
    "honest"
    "humorous"
    "impolite"
    "optimistic"
    "pessimistic"
    "polite"
    "righteous"
    "sycophantic"
)

mkdir -p "$save_dir"

for trait in "${traits[@]}"; do
    pos_csv="${extract_dir}/${trait}_pos_instruct.csv"
    neg_csv="${extract_dir}/${trait}_neg_instruct.csv"

    if [ ! -f "$pos_csv" ]; then
        echo "SKIP $trait: missing $pos_csv"
        continue
    fi

    if [ ! -f "$neg_csv" ]; then
        echo "SKIP $trait: missing $neg_csv"
        continue
    fi

    if [ -f "${save_dir}/${trait}_response_avg_diff_prompt_attn_delta.pt" ]; then
        echo "SKIP $trait: vector already exists"
        continue
    fi

    echo "=== Generating prompt_attn_delta vector for: $trait ==="
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} python core/generate_vec.py \
        --model_name "$model" \
        --pos_path "$pos_csv" \
        --neg_path "$neg_csv" \
        --trait "$trait" \
        --save_dir "$save_dir" \
        --threshold 50 \
        --extraction_method prompt_attn_delta

    echo "Done: $trait"
    echo ""
done

echo "All traits processed. Vectors saved to $save_dir/"
