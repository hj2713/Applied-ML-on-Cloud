#!/bin/bash
# Start MLX speculative decoding server
# Port 8001 — for Mac (Shreya/Raj), called by run_experiment.sh
#
# MLX speculative decoding:
#   Target model : Llama-3.1-8B (full capability)
#   Draft model  : Llama-3.2-1B (fast approximator)
#   Note: This is standard spec decoding, NOT EAGLE-3 (EAGLE-3 is vLLM-specific)
#   Results will be labeled "mlx_spec" not "eagle3" in output files.

set -e

TARGET_MODEL="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
DRAFT_MODEL="mlx-community/Llama-3.2-1B-Instruct-4bit"
PORT=8001
LOG_FILE="logs/mlx_spec.log"

mkdir -p logs

# Archive old log if it exists to prevent overwriting
if [ -f "$LOG_FILE" ]; then
    mv "$LOG_FILE" "logs/mlx_spec_archive_$(date +%Y%m%d_%H%M%S).log"
fi

echo "Starting MLX speculative decoding server (port $PORT)..."
nohup python3 -m mlx_lm.server \
    --model "$TARGET_MODEL" \
    --draft-model "$DRAFT_MODEL" \
    --port "$PORT" \
    --host 0.0.0.0 \
    > "$LOG_FILE" 2>&1 &

echo $! > logs/mlx_spec.pid
echo "MLX spec PID=$(cat logs/mlx_spec.pid) — log: $LOG_FILE"
