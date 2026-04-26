#!/bin/bash
# Start EAGLE-3 vLLM server (speculative decoding)
# Port 8001 — called automatically by run_experiment.sh

set -e

# Redirect HF cache to bigdisk
export HF_HOME="/mnt/disks/bigdisk/hf_cache"
mkdir -p "$HF_HOME" || echo "Warning: Could not create HF_HOME, may already exist or permission denied"

# Using unsloth mirror — same weights as meta-llama/Meta-Llama-3.1-8B-Instruct, no gating
TARGET_MODEL="unsloth/Meta-Llama-3.1-8B-Instruct"
DRAFT_MODEL="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
PORT=8001
LOG_FILE="logs/eagle3.log"

mkdir -p logs

# Archive old log if it exists to prevent overwriting
if [ -f "$LOG_FILE" ]; then
    mv "$LOG_FILE" "logs/eagle3_archive_$(date +%Y%m%d_%H%M%S).log"
fi

echo "Starting EAGLE-3 server (port $PORT)..."
nohup vllm serve "$TARGET_MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.90 \
    --served-model-name eagle3 \
    --speculative-config "{ \"method\": \"eagle3\", \"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": 5 }" \
    > "$LOG_FILE" 2>&1 &

echo $! > logs/eagle3.pid
echo "EAGLE-3 PID=$(cat logs/eagle3.pid) — log: $LOG_FILE"
