#!/bin/bash
# ============================================================
# GCP INSTANCE SETUP — Run this ONCE after SSH-ing into the VM
# Works for both L4 and A100 instances
#
# What this does (in order):
#   1. Updates system packages
#   2. Installs Python deps (vllm, torch, etc.)
#   3. Logs into HuggingFace
#   4. Downloads Llama-3.1-8B target model
#   5. Downloads EAGLE-3 draft model
#   6. Creates all needed directories
#
# Usage:
#   export HF_TOKEN=hf_xxxxxxxxxxxx
#   bash infra/setup_gcp.sh
# ============================================================

set -e

# ── Check HF token ─────────────────────────────────────────
if [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "ERROR: HF_TOKEN is not set."
    echo "  Get your token at: https://huggingface.co/settings/tokens"
    echo "  Then run: export HF_TOKEN=hf_xxxxxxxxxxxx"
    echo ""
    exit 1
fi

echo ""
echo "========================================"
echo "  GCP Benchmark Environment Setup"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "========================================"
echo ""

# ── Step 1: System check ────────────────────────────────────
echo "[1/5] Checking CUDA and system..."
nvidia-smi
python3 --version

# Use python3 -m pip — works on all GCP images regardless of whether pip3 is in PATH
PIP="python3 -m pip"

# ── Step 2: Install Python dependencies ────────────────────
echo ""
echo "[2/5] Installing Python dependencies..."
$PIP install --quiet --upgrade pip
$PIP install \
    "vllm>=0.4.3" \
    "torch>=2.2.0" \
    "transformers>=4.40.0" \
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
echo "Python dependencies installed."

# ── Step 3: HuggingFace login ───────────────────────────────
echo ""
echo "[3/5] Logging into HuggingFace..."
huggingface-cli login --token "$HF_TOKEN"

# ── Step 4: Download target model ──────────────────────────
echo ""
echo "[4/5] Downloading Llama-3.1-8B-Instruct (~16 GB, takes ~10 min)..."
huggingface-cli download \
    meta-llama/Meta-Llama-3.1-8B-Instruct \
    --local-dir-use-symlinks False
echo "Target model downloaded."

# ── Step 5: Download EAGLE-3 draft model ───────────────────
echo ""
echo "[5/5] Downloading EAGLE-3 draft model (~500 MB)..."
huggingface-cli download \
    yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
    --local-dir-use-symlinks False
echo "Draft model downloaded."

# ── Create directories ──────────────────────────────────────
mkdir -p logs results/raw results/plots results/tables data/prompts

echo ""
echo "========================================"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Prepare datasets:"
echo "       python3 data/prepare_datasets.py --n-samples 50"
echo ""
echo "  2. Run full experiment (baseline + eagle3 sequentially):"
echo "       bash benchmark/run_experiment.sh L4"
echo "       (or A100 for Aaryaman's machine)"
echo "========================================"
