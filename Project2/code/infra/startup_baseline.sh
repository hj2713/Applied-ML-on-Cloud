#!/bin/bash
# Start baseline vLLM server (greedy decoding, no speculative decoding)
# Port 8000 — called automatically by run_experiment.sh

set -e

# Redirect HF cache to bigdisk (home disk is small on GCP)
export HF_HOME="/mnt/disks/bigdisk/hf_cache"
mkdir -p "$HF_HOME"

# Using unsloth mirror — same weights as meta-llama/Meta-Llama-3.1-8B-Instruct, no gating
TARGET_MODEL="unsloth/Meta-Llama-3.1-8B-Instruct"
PORT=8000
LOG_FILE="logs/baseline.log"

mkdir -p logs

# Archive old log if it exists to prevent overwriting
if [ -f "$LOG_FILE" ]; then
    mv "$LOG_FILE" "logs/baseline_archive_$(date +%Y%m%d_%H%M%S).log"
fi

echo "Starting baseline server (port $PORT)..."
nohup vllm serve "$TARGET_MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.90 \
    --served-model-name baseline \
    > "$LOG_FILE" 2>&1 &

echo $! > logs/baseline.pid
echo "Baseline PID=$(cat logs/baseline.pid) — log: $LOG_FILE"
