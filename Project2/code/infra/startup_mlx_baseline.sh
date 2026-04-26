#!/bin/bash
# Start MLX baseline server (greedy decoding, no speculative decoding)
# Port 8000 — for Mac (Shreya/Raj), called by run_experiment.sh

set -e

MODEL="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
PORT=8000
LOG_FILE="logs/baseline.log"

mkdir -p logs

# Archive old log if it exists to prevent overwriting
if [ -f "$LOG_FILE" ]; then
    mv "$LOG_FILE" "logs/mlx_baseline_archive_$(date +%Y%m%d_%H%M%S).log"
fi

echo "Starting MLX baseline server (port $PORT)..."
# mlx_lm.server exposes an OpenAI-compatible /v1/chat/completions endpoint
nohup python3 -m mlx_lm.server \
    --model "$MODEL" \
    --port "$PORT" \
    --host 0.0.0.0 \
    > "$LOG_FILE" 2>&1 &

echo $! > logs/baseline.pid
echo "MLX baseline PID=$(cat logs/baseline.pid) — log: $LOG_FILE"
