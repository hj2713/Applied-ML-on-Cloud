#!/bin/bash
# ============================================================
# MASTER EXPERIMENT RUNNER
# Runs the full sweep for ONE GPU sequentially:
#   Phase 1: Start baseline server → run all sweeps → stop
#   Phase 2: Start eagle3 server  → run all sweeps → stop
#
# SAME script for L4 (Himanshu), A100 (Aaryaman), mac (Shreya/Raj)
# Only the GPU_TYPE argument differs.
#
# Usage:
#   bash benchmark/run_experiment.sh L4        # Himanshu
#   bash benchmark/run_experiment.sh A100      # Aaryaman
#   bash benchmark/run_experiment.sh mac       # Shreya / Raj
#
# Prerequisites:
#   - infra/setup_gcp.sh (or setup_mac.sh) already run
#   - data/prompts/*.jsonl already generated
# ============================================================

set -e

GPU_TYPE=${1:?"\nUsage: bash benchmark/run_experiment.sh <GPU_TYPE>\n  Options: L4 | A100 | mac"}

N_SAMPLES=50
CONCURRENCIES=(1 4 8 16 32)
TASKS=("chat" "code" "summarization")
TRIALS=3
BASELINE_PORT=8000
EAGLE3_PORT=8001

# For mac, systems are mlx_baseline and mlx_spec instead of baseline and eagle3
if [ "$GPU_TYPE" = "mac" ]; then
    SYSTEMS=("mlx_baseline" "mlx_spec")
    SYSTEM_PORTS=("8000" "8001")
else
    SYSTEMS=("baseline" "eagle3")
    SYSTEM_PORTS=("8000" "8001")
fi

echo ""
echo "========================================"
echo "  EAGLE-3 Benchmark — Full Experiment"
echo "  GPU Type : $GPU_TYPE"
echo "  Trials   : $TRIALS per cell"
echo "  Tasks    : ${TASKS[*]}"
echo "  Concurrency: ${CONCURRENCIES[*]}"
echo "========================================"

# ── Verify prompts exist ────────────────────────────────────
for TASK in "${TASKS[@]}"; do
    FILE="data/prompts/${TASK}_${N_SAMPLES}.jsonl"
    if [ ! -f "$FILE" ]; then
        echo ""
        echo "ERROR: Missing prompt file: $FILE"
        echo "  Run first: python3 data/prepare_datasets.py --n-samples $N_SAMPLES"
        exit 1
    fi
done
echo ""
echo "Prompt files verified."

# ── Helper: wait for server health ─────────────────────────
wait_for_server() {
    local PORT=$1
    local MAX_WAIT=360
    local WAIT=0
    echo "  Waiting for server on port $PORT..."
    until curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; do
        if [ "$WAIT" -ge "$MAX_WAIT" ]; then
            echo "  ERROR: Server not ready after ${MAX_WAIT}s. Check logs/"
            exit 1
        fi
        sleep 5
        WAIT=$((WAIT + 5))
        printf "  ...%ds\r" "$WAIT"
    done
    echo "  Server on port $PORT is ready."
}

# ── Helper: run sweep for one system ────────────────────────
run_sweep() {
    local SYSTEM=$1
    local PORT=$2
    local TOTAL=$(( ${#CONCURRENCIES[@]} * ${#TASKS[@]} * TRIALS ))
    local DONE=0

    echo ""
    echo "--- Running sweep: system=$SYSTEM port=$PORT ---"

    for TASK in "${TASKS[@]}"; do
        PROMPTS="data/prompts/${TASK}_${N_SAMPLES}.jsonl"
        for CONCURRENCY in "${CONCURRENCIES[@]}"; do
            for TRIAL in $(seq 1 $TRIALS); do
                DONE=$((DONE + 1))
                printf "[%d/%d] %s c=%d trial=%d\n" "$DONE" "$TOTAL" "$TASK" "$CONCURRENCY" "$TRIAL"

                python3 benchmark/load_test.py \
                    --server-url "http://localhost:$PORT" \
                    --system "$SYSTEM" \
                    --gpu-type "$GPU_TYPE" \
                    --task "$TASK" \
                    --prompts-file "$PROMPTS" \
                    --concurrency "$CONCURRENCY" \
                    --trial "$TRIAL"

                sleep 1  # brief pause between cells
            done
        done
    done

    echo ""
    echo "  Sweep complete for $SYSTEM."
}

# ── Phase 1: Baseline ───────────────────────────────────────
echo ""
echo "========================================"
echo "  PHASE 1: Baseline"
echo "========================================"

if [ "$GPU_TYPE" = "mac" ]; then
    bash infra/startup_mlx_baseline.sh &
else
    bash infra/startup_baseline.sh &
fi

wait_for_server $BASELINE_PORT
run_sweep "${SYSTEMS[0]}" $BASELINE_PORT

echo ""
echo "  Stopping baseline server..."
kill "$(cat logs/baseline.pid)" 2>/dev/null || true
sleep 5

# ── Phase 2: EAGLE-3 / MLX speculative ─────────────────────
echo ""
echo "========================================"
echo "  PHASE 2: ${SYSTEMS[1]}"
echo "========================================"

if [ "$GPU_TYPE" = "mac" ]; then
    bash infra/startup_mlx_spec.sh &
else
    bash infra/startup_eagle3.sh &
fi

wait_for_server $EAGLE3_PORT
run_sweep "${SYSTEMS[1]}" $EAGLE3_PORT

echo ""
echo "  Stopping ${SYSTEMS[1]} server..."
kill "$(cat logs/${SYSTEMS[1]}.pid)" 2>/dev/null || true

# ── Done ────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "  GPU Type : $GPU_TYPE"
echo "  Results  : results/raw/ ($(ls results/raw/*.jsonl 2>/dev/null | wc -l | tr -d ' ') files)"
echo ""
echo "  To analyze results:"
echo "    python3 analysis/analyze.py"
echo "========================================"
