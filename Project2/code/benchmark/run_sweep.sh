#!/bin/bash
# Full experiment sweep — runs all cells of the matrix for ONE system on ONE GPU.
#
# Run this ONCE for baseline, then ONCE for eagle3.
# The script calls load_test.py for each combination.
#
# Usage:
#   # On L4 instance, baseline:
#   bash benchmark/run_sweep.sh baseline L4 http://localhost:8000
#
#   # On L4 instance, eagle3:
#   bash benchmark/run_sweep.sh eagle3 L4 http://localhost:8001
#
#   # On A100 instance, baseline:
#   bash benchmark/run_sweep.sh baseline A100 http://localhost:8000

set -e

SYSTEM=${1:?Usage: run_sweep.sh <system> <gpu_type> <server_url>}
GPU_TYPE=${2:?}
SERVER_URL=${3:?}
N_SAMPLES=50
TRIALS=3

CONCURRENCIES=(1 4 8 16 32)
TASKS=("chat" "code" "summarization")

echo "========================================"
echo "  Starting full sweep"
echo "  System     : $SYSTEM"
echo "  GPU type   : $GPU_TYPE"
echo "  Server     : $SERVER_URL"
echo "  Trials     : $TRIALS per cell"
echo "  Concurrency: ${CONCURRENCIES[*]}"
echo "  Tasks      : ${TASKS[*]}"
echo "========================================"
echo ""

TOTAL=$(( ${#CONCURRENCIES[@]} * ${#TASKS[@]} * TRIALS ))
DONE=0

for TASK in "${TASKS[@]}"; do
    PROMPTS_FILE="data/prompts/${TASK}_${N_SAMPLES}.jsonl"

    if [ ! -f "$PROMPTS_FILE" ]; then
        echo "ERROR: Prompts file not found: $PROMPTS_FILE"
        echo "  Run: python data/prepare_datasets.py --n-samples $N_SAMPLES"
        exit 1
    fi

    for CONCURRENCY in "${CONCURRENCIES[@]}"; do
        for TRIAL in $(seq 1 $TRIALS); do
            DONE=$((DONE + 1))
            echo "[$DONE/$TOTAL] system=$SYSTEM gpu=$GPU_TYPE task=$TASK concurrency=$CONCURRENCY trial=$TRIAL"

            python benchmark/load_test.py \
                --server-url "$SERVER_URL" \
                --system "$SYSTEM" \
                --gpu-type "$GPU_TYPE" \
                --task "$TASK" \
                --prompts-file "$PROMPTS_FILE" \
                --concurrency "$CONCURRENCY" \
                --trial "$TRIAL"

            # Brief pause between cells to let GPU cool and avoid OOM
            sleep 2
        done
    done
done

echo ""
echo "========================================"
echo "  Sweep complete!"
echo "  Results in: results/raw/"
echo "  Files: $(ls results/raw/ | wc -l) total"
echo "========================================"
