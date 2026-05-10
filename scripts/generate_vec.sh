#!/bin/bash
# Generate original residual-stream persona vectors.
#
# Usage:
#   bash scripts/generate_vec.sh [gpu_ids] [model_path_or_hf_id]
#
# The repository includes eval_persona_extract/*.csv, so this script can
# regenerate vectors directly from those extraction CSVs.

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/core${PYTHONPATH:+:$PYTHONPATH}"

gpu=${1:-"0,1,2,3"}
model=${2:-"${MODEL_PATH_OR_ID:-Qwen/Qwen2.5-7B-Instruct}"}
model_name=$(basename "$model")
extract_dir="eval_persona_extract/${model_name}"
save_dir="vectors/original/${model_name}"

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

    if [ ! -f "$pos_csv" ] || [ ! -f "$neg_csv" ]; then
        echo "SKIP $trait: missing extraction CSVs in $extract_dir"
        continue
    fi

    echo "=== Generating original vector for: $trait ==="
    CUDA_VISIBLE_DEVICES=$gpu python core/generate_vec.py \
        --model_name "$model" \
        --pos_path "$pos_csv" \
        --neg_path "$neg_csv" \
        --trait "$trait" \
        --save_dir "$save_dir" \
        --threshold 50 \
        --extraction_method hidden_state
done

echo "Done. Vectors saved to $save_dir/"
