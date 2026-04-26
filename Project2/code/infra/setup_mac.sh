#!/bin/bash
# ============================================================
# MAC (Apple Silicon) SETUP — For Shreya and Raj
# Uses MLX instead of vLLM (MLX is optimized for Apple Silicon)
#
# Requirements:
#   - Mac with Apple Silicon (M1/M2/M3)
#   - Python 3.10+
#   - ~20 GB free RAM
#
# Usage:
#   export HF_TOKEN=hf_xxxxxxxxxxxx
#   bash infra/setup_mac.sh
# ============================================================

set -e

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN is not set."
    echo "  Run: export HF_TOKEN=hf_xxxxxxxxxxxx"
    exit 1
fi

echo ""
echo "========================================"
echo "  Mac Benchmark Setup (MLX)"
echo "  Platform: $(uname -m)"
echo "========================================"

# ── Verify Apple Silicon ────────────────────────────────────
if [ "$(uname -m)" != "arm64" ]; then
    echo "WARNING: This script is designed for Apple Silicon (arm64)."
    echo "  Detected: $(uname -m)"
fi

# ── Install dependencies ────────────────────────────────────
echo ""
echo "[1/4] Installing Python dependencies..."
pip3 install --quiet --upgrade pip
pip3 install \
    "mlx>=0.10.0" \
    "mlx-lm>=0.10.0" \
    "huggingface_hub>=0.22.0" \
    "datasets>=2.18.0" \
    "aiohttp>=3.9.0" \
    "rouge_score>=0.1.2" \
    "numpy>=1.26.0" \
    "pandas>=2.2.0" \
    "matplotlib>=3.8.0" \
    "seaborn>=0.13.0" \
    "streamlit>=1.33.0" \
    "tqdm>=4.66.0"
echo "Dependencies installed."

# ── HuggingFace login ───────────────────────────────────────
echo ""
echo "[2/4] Logging into HuggingFace..."
huggingface-cli login --token "$HF_TOKEN"

# ── Download MLX models ─────────────────────────────────────
# MLX uses 4-bit quantized models — much smaller than full precision
echo ""
echo "[3/4] Downloading MLX models..."

echo "  Downloading target model (Llama-3.1-8B in MLX 4-bit, ~4.5 GB)..."
huggingface-cli download \
    mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
    --local-dir-use-symlinks False

echo "  Downloading draft model (Llama-3.2-1B in MLX 4-bit, ~0.6 GB)..."
huggingface-cli download \
    mlx-community/Llama-3.2-1B-Instruct-4bit \
    --local-dir-use-symlinks False

echo "Models downloaded."

# ── Create directories ──────────────────────────────────────
echo ""
echo "[4/4] Creating directories..."
mkdir -p logs results/raw results/plots results/tables data/prompts
echo "Done."

echo ""
echo "========================================"
echo "  Mac setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Prepare datasets:"
echo "       python3 data/prepare_datasets.py --n-samples 50"
echo ""
echo "  2. Run full experiment:"
echo "       bash benchmark/run_experiment.sh mac"
echo "========================================"
