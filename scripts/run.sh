#!/bin/bash
# Usage:
#   bash scripts/run.sh [extra arguments passed to run_inference.py]
# Configuration can be overridden through environment variables, e.g.
#   DATASET=VRBench BACKBONE=Qwen2-VL-72B GPU_IDS="0 1 2 3" bash scripts/run.sh --lambda_param 0.5

set -e
cd "$(dirname "$0")/.."

# Dataset: VideoMME, VRBench, CinePile
DATASET=${DATASET:-VideoMME}
# Backbone: Qwen2.5-VL-7B, Qwen2-VL-72B
BACKBONE=${BACKBONE:-Qwen2.5-VL-7B}
DATA_ROOT=${DATA_ROOT:-./dataset}
TOKEN_BUDGET=${TOKEN_BUDGET:-8192}
# Extra arguments are appended to the default output directory name, so that
# different settings do not overwrite each other
TAG=$(echo "$*" | sed 's/--//g' | tr -cs 'A-Za-z0-9.' '_' | sed -e 's/^_//' -e 's/_$//')
OUTPUT_DIR=${OUTPUT_DIR:-./results/${DATASET}_${BACKBONE}_B${TOKEN_BUDGET}${TAG:+_$TAG}}
GPU_IDS=(${GPU_IDS:-0 1 2 3})

# Optional local checkpoints (HuggingFace ids are used otherwise)
EXTRA_PATHS=()
[ -n "$MODEL_PATH" ] && EXTRA_PATHS+=(--model_path "$MODEL_PATH")
[ -n "$CLIP_PATH" ] && EXTRA_PATHS+=(--clip_path "$CLIP_PATH")

mkdir -p "$OUTPUT_DIR"

if [ "$BACKBONE" = "Qwen2-VL-72B" ]; then
    # The 72B model is sharded across all listed GPUs in a single process
    GROUPS_OF_GPUS=("$(IFS=,; echo "${GPU_IDS[*]}")")
else
    # One data chunk per GPU
    GROUPS_OF_GPUS=("${GPU_IDS[@]}")
fi
NUM_CHUNKS=${#GROUPS_OF_GPUS[@]}

echo "Dataset: $DATASET | Backbone: $BACKBONE | Budget: $TOKEN_BUDGET | Chunks: $NUM_CHUNKS"

pids=()
for ((i=0; i<NUM_CHUNKS; i++)); do
    CUDA_VISIBLE_DEVICES=${GROUPS_OF_GPUS[$i]} python scripts/run_inference.py \
        --dataset "$DATASET" \
        --backbone "$BACKBONE" \
        --data_root "$DATA_ROOT" \
        --output_dir "$OUTPUT_DIR" \
        --token_budget "$TOKEN_BUDGET" \
        --num_chunks "$NUM_CHUNKS" \
        --chunk_idx "$i" \
        "${EXTRA_PATHS[@]}" "$@" > "$OUTPUT_DIR/log_chunk${i}.txt" 2>&1 &
    pids+=($!)
done

for pid in "${pids[@]}"; do
    wait "$pid"
done

python scripts/merge_results.py "$OUTPUT_DIR"
