#!/bin/bash
# Multi-layer GCAD evaluation.
#
# Usage:
#   OPENAI_API_KEY=... bash gcad/run.sh [gpu_ids] [model_path_or_hf_id]

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/core${PYTHONPATH:+:$PYTHONPATH}"

gpu=${1:-"0,1,2,3"}
model=${2:-"${MODEL_PATH_OR_ID:-Qwen/Qwen2.5-7B-Instruct}"}
model_name=$(basename "$model")

layers=${GCAD_LAYERS:-"9,10,11,12,13,14,15,16,17,18,19"}
layers_tag=$(echo "$layers" | tr ',' '_')
n_turns=${N_TURNS:-5}
n_groups=${N_GROUPS:-20}
n_samples=${N_SAMPLES:-3}
results_root="results/${model_name}/gcad_L${layers_tag}_turn_${n_turns}"

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
base_coefs=(${BASE_COEFS:-3.5})
scales=(${SCALES:-1.5})

config_file=$(mktemp /tmp/gcad_multilayer_configs_XXXX.json)
echo "[" > "$config_file"
first=true
layers_json="[$(echo "$layers" | sed 's/,/, /g')]"

for trait in "${traits[@]}"; do
    for bc in "${base_coefs[@]}"; do
        for sc in "${scales[@]}"; do
            out_path="${results_root}/${trait}.csv"
            $first || echo "," >> "$config_file"
            first=false
            cat >> "$config_file" <<ENTRY
  {
    "method": "gcad",
    "trait": "$trait",
    "output_path": "$out_path",
    "layer_indices": $layers_json,
    "base_coef": $bc,
    "scale": $sc,
    "inverse": false
  }
ENTRY
        done
    done
done

echo "]" >> "$config_file"
echo "Config: $config_file ($(grep -c '"trait"' "$config_file") entries, layers=$layers)"

CUDA_VISIBLE_DEVICES=$gpu python -u -m eval.eval_multiturn_pipeline \
    --model "$model" \
    --configs "$config_file" \
    --n_turns "$n_turns" \
    --n_groups "$n_groups" \
    --n_samples "$n_samples" \
    --temperature 1.0

rm -f "$config_file"
echo "Done. Results in ${results_root}/"
