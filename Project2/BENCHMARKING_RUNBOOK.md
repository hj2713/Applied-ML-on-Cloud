# Benchmarking Runbook — Step by Step

**Target Models Overview:**
- **GPU Users (vLLM):**
  - Target Model: `unsloth/Meta-Llama-3.1-8B-Instruct`
  - Draft Model (Speculative): `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` (EAGLE-3 algorithm)
- **Mac Users (MLX):**
  - Target Model: `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`
  - Draft Model (Speculative): `mlx-community/Llama-3.2-1B-Instruct-4bit` (Standard speculative decoding)

---

### Memory & Storage Footprint (Reference)

Understanding *why* we run things sequentially on the L4 is critical for your project report. 

**VRAM Math for an 8B Model (bfloat16):**
*   Parameters: 8 Billion
*   Precision: 16-bit (`bfloat16`) = 2 bytes per parameter.
*   **Weight Size:** 8B × 2 bytes = **~16 GB VRAM** just to load the model into memory.

**Per-Server Requirement (GPU):**
1.  **Baseline Server:** ~16 GB (Weights) + ~4 GB (KV Cache for context) = **~20 GB VRAM**
2.  **EAGLE-3 Server:** ~16 GB (Target Weights) + ~2 GB (Draft Weights) + ~4 GB (KV Cache) = **~22 GB VRAM**

**Why L4 Fails the Live Demo:**
An NVIDIA L4 GPU has exactly **24 GB of VRAM**. Trying to start both the Baseline (20GB) and EAGLE-3 (22GB) simultaneously demands ~42 GB of VRAM, resulting in an immediate Out-Of-Memory (OOM) crash.

**Who CAN run the Live Demo?**
*   **Aaryaman (A100):** The A100 comes in 40GB or 80GB VRAM variants, allowing both servers to fit side-by-side.
*   **Mac Users:** Because MLX uses 4-bit quantization, the 8B model only takes **~4.5 GB** of memory. Mac unified memory (16GB or 32GB) handles both servers easily.

**Disk Storage Footprint:**
*   **vLLM HuggingFace Cache:** ~35 GB (Requires the mounted persistent disk).
*   **Mac MLX Cache:** ~8 GB.
*   **Datasets:** ~1 MB total (50 prompts each for Chat, Code, and Summarization).

---
```bash
cd Applied-ML-on-Cloud/Project2/code
source .venv/bin/activate  # Or your equivalent env activation
```

---

## Phase 0 — One-Time Setup

**1. Pull latest code**
```bash
git pull origin main
```

**2. Install Dependencies & Setup Environment**
Run the setup script relevant to your machine:
- **GPU Users:** `bash infra/setup_gcp.sh`
- **Mac Users:** `bash infra/setup_mac.sh`

---

## Phase 1 — Prepare Datasets (One-Time Setup)

*Note: Himanshu has already run this step and committed the results. You **only** need to run this step if the `data/prompts/` folder is empty or missing on your machine.*

```bash
python3 data/prepare_datasets.py --n-samples 50
```
*Note: This creates `chat_50.jsonl`, `code_50.jsonl`, and `summarization_50.jsonl` in `data/prompts/`.*

---

## Phase 2 — Run Baseline Experiment

### 2a. Start Baseline Server
- **GPU Users:** `bash infra/startup_baseline.sh`
- **Mac Users:** `bash infra/startup_mlx_baseline.sh`

### 2b. Wait for server to be ready
Watch the logs until you see it is ready (e.g., "Application startup complete" or "Uvicorn running on"):
- **GPU Users:** `tail -f logs/baseline.log`
- **Mac Users:** `tail -f logs/baseline.log`
*(Press Ctrl+C to exit log view once it says ready)*

### 2c. Run Baseline Sweep (~45-60 min)
- **GPU Users:** `bash benchmark/run_sweep.sh baseline <GPU_TYPE> http://localhost:8000` *(replace `<GPU_TYPE>` with `L4` or `A100`)*
- **Mac Users:** `bash benchmark/run_sweep.sh mlx_baseline mac http://localhost:8000`

### 2d. Stop Baseline Server
```bash
kill $(cat logs/baseline.pid)
```
*(GPU users: also run `pkill -9 vllm` to be absolutely safe and free memory)*

---

## Phase 3 — Run Speculative Decoding Experiment

### 3a. Start Speculative Decoding Server
- **GPU Users:** `bash infra/startup_eagle3.sh`
- **Mac Users:** `bash infra/startup_mlx_spec.sh`

### 3b. Wait for server to be ready
- **GPU Users:** `tail -f logs/eagle3.log`
- **Mac Users:** `tail -f logs/mlx_spec.log`

### 3c. Run Speculative Sweep (~45-60 min)
- **GPU Users:** `bash benchmark/run_sweep.sh eagle3 <GPU_TYPE> http://localhost:8001` *(replace `<GPU_TYPE>` with `L4` or `A100`)*
- **Mac Users:** `bash benchmark/run_sweep.sh mlx_spec mac http://localhost:8001`

### 3d. Stop Server
```bash
# GPU Users
kill $(cat logs/eagle3.pid)
pkill -9 vllm

# Mac Users
kill $(cat logs/mlx_spec.pid)
```

---

## Phase 4 — Verify Results

Verify you have 90 result files (45 baseline + 45 speculative).
```bash
ls results/raw/*.jsonl | wc -l
```

---

## Phase 5 — Run Quality Characterization (Sequential)

*Note: Since loading both models simultaneously will cause Out-Of-Memory (OOM) errors on smaller GPUs (and heavily tax Mac memory), we must run these sequentially. We use `--trial 99` so it saves as a new file and doesn't overwrite your Phase 2/3 sweep results!*

### 5a. Create Tiny Prompt File (Run Once)
Create a small set of 10 prompts for quick quality testing:
```bash
head -n 10 data/prompts/summarization_50.jsonl > data/prompts/quality_tiny.jsonl
```

### 5b. Generate Baseline Samples
1. **Start Baseline Server** (Wait for it to load using `tail -f` like before)
   - GPU: `bash infra/startup_baseline.sh`
   - Mac: `bash infra/startup_mlx_baseline.sh`
2. **Run Inference**
   - GPU Users:
     ```bash
     python3 benchmark/load_test.py --system baseline --gpu-type <GPU_TYPE> --task summarization --prompts-file data/prompts/quality_tiny.jsonl --concurrency 1 --trial 99 --server-url http://localhost:8000
     ```
   - Mac Users:
     ```bash
     python3 benchmark/load_test.py --system mlx_baseline --gpu-type mac --task summarization --prompts-file data/prompts/quality_tiny.jsonl --concurrency 1 --trial 99 --server-url http://localhost:8000
     ```
3. **Kill Baseline Server**
   ```bash
   kill $(cat logs/baseline.pid)
   ```
   *(GPU users: run `pkill -9 vllm` as well)*

### 5c. Generate Speculative Samples
1. **Start Speculative Server** (Wait for it to load)
   - GPU: `bash infra/startup_eagle3.sh`
   - Mac: `bash infra/startup_mlx_spec.sh`
2. **Run Inference**
   - GPU Users:
     ```bash
     python3 benchmark/load_test.py --system eagle3 --gpu-type <GPU_TYPE> --task summarization --prompts-file data/prompts/quality_tiny.jsonl --concurrency 1 --trial 99 --server-url http://localhost:8001
     ```
   - Mac Users:
     ```bash
     python3 benchmark/load_test.py --system mlx_spec --gpu-type mac --task summarization --prompts-file data/prompts/quality_tiny.jsonl --concurrency 1 --trial 99 --server-url http://localhost:8001
     ```
3. **Kill Speculative Server**
   - GPU: `kill $(cat logs/eagle3.pid); pkill -9 vllm`
   - Mac: `kill $(cat logs/mlx_spec.pid)`

### 5d. Compare Results
The generated samples will be saved in `results/raw/` ending in `_t99.jsonl`. Visually compare the output text between the baseline and speculative outputs to ensure the quality remains identical.

---

## Phase 6 — Push Results to GitHub

```bash
git add results/raw/
git commit -m "feat: add <YOUR_HARDWARE> benchmark results"
git push origin main
```

---

## Phase 7 — After Everyone Pushes Results

Pull everyone's results and run the analysis to generate plots and summary tables.
```bash
git pull origin main
python3 benchmark/plot_results.py
```
*(Results will be saved in `results/plots/` and `results/tables/`)*

---

## Phase 8 — Running the Live Dashboard (Optional / Demo)

The repository includes a Streamlit web dashboard (`code/dashboard/app.py`) for live demonstrations and interactive graph viewing. 

**Who can run the "Live Demo" tab?**
Because the Live Demo fires requests to both servers simultaneously, it requires high memory:
- **A100 Users:** Yes (40GB/80GB VRAM is plenty).
- **Mac Users:** Yes (Unified memory handles 4-bit quantization easily).
- **L4 Users:** No (24GB VRAM will crash with OOM).

**How to connect to a Remote GPU (A100):**
If your servers are running on a remote GCP A100 instance, but you want to run the Dashboard UI on your local Mac, `localhost:8000` won't work by default. You have two options:
1.  **SSH Port Forwarding (Recommended):** Forward the server ports to your local Mac when you SSH into GCP:
    ```bash
    ssh -L 8000:localhost:8000 -L 8001:localhost:8001 <YOUR_GCP_USER>@<YOUR_GCP_IP>
    ```
    *(Now, the default `http://localhost:8000` in the Streamlit app will magically route to your remote A100).*
2.  **External IP:** In the Streamlit sidebar, change the "Baseline URL" from `http://localhost:8000` to `http://<YOUR_GCP_EXTERNAL_IP>:8000` (Make sure GCP firewall allows traffic on ports 8000 and 8001).

**Start the Dashboard:**
```bash
source .venv/bin/activate
pip install streamlit aiohttp
streamlit run dashboard/app.py
```
Open your browser to `http://localhost:8501`.
