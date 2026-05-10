#!/bin/bash
# Multi-turn evaluation for the original residual-stream steering baseline.
#
# Usage:
#   OPENAI_API_KEY=... bash scripts/eval_steering_multiturn_attn_mlp.sh [gpu_ids] [model_path_or_hf_id]

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/core${PYTHONPATH:+:$PYTHONPATH}"

gpu=${1:-"0,1,2,3"}
model=${2:-"${MODEL_PATH_OR_ID:-Qwen/Qwen2.5-7B-Instruct}"}
model_name=$(basename "$model")
n_turns=${N_TURNS:-5}
n_groups=${N_GROUPS:-20}
n_samples=${N_SAMPLES:-3}
results_root="results/${model_name}/original_L20_c2_turn_${n_turns}"

# traits=(
#     "apathetic"
#     "creative"
#     "curious"
#     "empathetic"
#     "evil"
#     "factual"
#     "hallucinating"
#     "honest"
#     "humorous"
#     "impolite"
#     "optimistic"
#     "pessimistic"
#     "polite"
#     "righteous"
#     "sycophantic"
# )
traits=(
    "apathetic"
    "evil"
    "hallucinating"
    "humorous"
    "impolite"
    "sycophantic"
)

config_file=$(mktemp /tmp/original_multiturn_configs_XXXX.json)
echo "[" > "$config_file"
first=true

for trait in "${traits[@]}"; do
    vp="vectors/original/${model_name}/${trait}_response_avg_diff.pt"
    if [ ! -f "$vp" ]; then
        echo "SKIP $trait: missing $vp"
        continue
    fi

    $first || echo "," >> "$config_file"
    first=false
    cat >> "$config_file" <<ENTRY
  {
    "method": "residual",
    "trait": "$trait",
    "output_path": "${results_root}/${trait}.csv",
    "layers": "20:2.0",
    "vector_path": "$vp",
    "target_module": "layer",
    "zero_indexed_vectors": false,
    "persona_instruction_type": null
  }
ENTRY
done

echo "]" >> "$config_file"
echo "Config: $config_file ($(grep -c '"trait"' "$config_file") entries)"

CUDA_VISIBLE_DEVICES=$gpu python -u -m eval.eval_multiturn_pipeline \
    --model "$model" \
    --configs "$config_file" \
    --n_turns "$n_turns" \
    --n_groups "$n_groups" \
    --n_samples "$n_samples" \
    --temperature 1.0

rm -f "$config_file"
echo "Done. Results in ${results_root}/"
