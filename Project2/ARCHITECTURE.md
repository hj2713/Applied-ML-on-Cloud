# Technical Architecture & Design Decisions

This document details the architecture, infrastructure, and the rationale behind the technical choices for the EAGLE-3 Speculative Decoding Characterization project.

---

## 1. High-Level System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          BENCHMARK ORCHESTRATOR                         │
│                    (shared Python script, both teams run)               │
└────────────────────────┬──────────────┬──────────────┬──────────────────┘
                         │              │              │
              ┌──────────▼──┐    ┌──────▼──────┐    ┌─▼─────────────────┐
              │  vLLM Team  │    │  vLLM Team  │    │    MLX Team       │
              │  GCP L4/A100│    │  GCP L4/A100│    │    Mac M-series   │
              │─────────────│    │─────────────│    │───────────────────│
              │ Server 1    │    │ Server 2    │    │ Server 3 + 4      │
              │ BASELINE    │    │ EAGLE-3     │    │ MLX Baseline      │
              │ greedy      │    │ speculative │    │ + MLX Spec Dec    │
              │ :8000       │    │ :8001       │    │ :8000 / :8001     │
              └──────────┬──┘    └──────┬──────┘    └─┬─────────────────┘
                         │              │              │
              ┌──────────▼──────────────▼──────────────▼──────────────────┐
              │                  METRICS COLLECTOR                        │
              │         (unified JSON output — same schema both teams)    │
              └──────────────────────────┬────────────────────────────────┘
                                         │
              ┌──────────────────────────▼────────────────────────────────┐
              │                  ANALYSIS PIPELINE                        │
              │           Python Pandas/Matplotlib (plot_results.py)      │
              └───────────────────────────────────────────────────────────┘
```

### 1.1 Experimental Hardware Specifications
For complete academic reproducibility, the exact cloud environments used to run these benchmarks are documented below. 

**GCP L4 Instance Profile (Himanshu's Run):**
* **Machine Type:** `g2-standard-4`
* **vCPUs:** 4
* **System Memory (RAM):** 16 GB
* **GPU:** 1x NVIDIA L4 Tensor Core GPU
* **GPU VRAM:** 24 GB *(Critical bottleneck parameter for speculative decoding)*
* **Architecture:** x86_64
* **Hourly Cost:** ~$0.70 / hr

*(Note: The A100 and Mac hardware profiles should be appended here once Aaryaman and the Mac team execute their runs).*

---

## 2. Model Selection: Rationale & Alternatives

### 2.1 Target Model
**Chosen:** `meta-llama/Meta-Llama-3.1-8B-Instruct`
* **Why we chose it:** It represents the current state-of-the-art for open-source ~8B parameter models. It excels in diverse tasks (chat, code, summarization). Crucially, an 8B model fits comfortably on a single GCP L4 GPU (24GB VRAM) in `bfloat16` and runs extremely fast on Mac M-series with 4-bit quantization.
* **Why we didn't choose 70B models:** A 70B model requires multiple GPUs (e.g., 2x to 4x A100s) to run effectively, which drastically increases cloud costs and makes reproducibility impossible for team members running locally on Macs.
* **Why we didn't choose 1B-3B models:** While fast, smaller models lack the reasoning capabilities required for complex tasks like HumanEval (Code). We needed a "production-grade" target to make the benchmark meaningful.

### 2.2 Draft Model (GPU/vLLM)
**Chosen:** `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`
* **Why we chose it:** EAGLE-3 is a feature-level speculative decoding algorithm that uses a lightweight head attached to the target model, rather than a completely separate small autoregressive model. This results in significantly higher acceptance rates (~40%+) compared to standard draft models.
* **Why we didn't choose a standard Llama-3.2-1B draft:** Standard speculative decoding relies on a separate small model. Because the 1B model has a different distribution than the 8B target, the acceptance rate is often very low for complex tasks, neutralizing the speedup.

### 2.3 Draft Model (Mac/MLX)
**Chosen:** `mlx-community/Llama-3.2-1B-Instruct-4bit` (Standard Speculative Decoding)
* **Why we chose it:** The MLX framework currently does not natively support the EAGLE-3 algorithm. Therefore, the Mac team uses the standard speculative decoding approach supported by Apple's `mlx_lm`. The 1B draft model is heavily quantized, making it lightning-fast to generate draft tokens on Mac unified memory.

---

## 3. Serving Frameworks

### 3.1 GPU Environment: vLLM
* **Why we chose it:** vLLM is the industry standard for high-throughput LLM serving. It features Continuous Batching, PagedAttention, and out-of-the-box support for the EAGLE-3 algorithm. It also exposes a Prometheus `/metrics` endpoint, which is essential for cleanly capturing the speculative decoding acceptance rate during benchmarks.
* **Why we didn't choose TGI (Text Generation Inference):** TGI is excellent but less flexible for experimental speculative decoding setups like EAGLE. 
* **Why we didn't choose TensorRT-LLM:** TensorRT requires an extensive Engine compilation step specific to every GPU architecture and batch size. This heavily slows down rapid iteration for a class project.

### 3.2 Mac Environment: MLX
* **Why we chose it:** Apple's official Machine Learning Array framework. It is deeply optimized for Apple Silicon (Metal) and unified memory architecture. It allows us to run LLMs effectively with negligible battery drain and high performance.
* **Why we didn't choose vLLM on Mac:** While vLLM has experimental Mac support, it is heavily optimized around CUDA (NVIDIA). Using vLLM on Mac would result in artificial bottlenecks that misrepresent the hardware's actual capabilities.

---

## 4. Benchmark Methodology & Design

### 4.1 Tasks and Datasets
We evaluate on three distinct datasets to test the robustness of speculative decoding across different data distributions:
1. **Chat (ShareGPT):** Highly conversational. High predictability. Expect highest acceptance rates.
2. **Code (HumanEval):** Strict syntax. Medium predictability. Tests if the draft model understands programming logic.
3. **Summarization (CNN/DailyMail):** Requires digesting context and outputting novel text. Lower predictability. Expect lowest acceptance rates.
* **Why 50 samples per task?** 50 requests provide statistical stability for averages (law of large numbers) while keeping cloud costs strictly bounded and sweeps manageable (~45 mins total).

### 4.2 Concurrency Sweep
**Levels:** `[1, 4, 8, 16, 32]`
* **Why we chose a sweep:** Speculative decoding provides the most benefit when a GPU is *memory-bandwidth bound* (low concurrency). At high concurrency (e.g., 32), the GPU becomes *compute-bound*, and the overhead of the draft model actually makes speculative decoding slower than the baseline. By sweeping from 1 to 32, we map the exact "Crossover Point" where speculative decoding ceases to be beneficial. Testing only at Concurrency 1 would yield a misleadingly positive result.

### 4.3 Sequential Execution on Limited Hardware
* **Why we run servers sequentially (Start -> Run -> Kill -> Start Next):** To isolate performance and prevent Out-Of-Memory (OOM) errors. For instance, on a 24GB L4 GPU, running both the baseline and speculative servers simultaneously would exceed VRAM limits, causing massive swap slowdowns or crashes.

---

## 5. Output Data Schema & Contract

To ensure that both the GPU team and the Mac team can use the exact same analysis scripts, we defined a strict JSONL schema for output:

```json
{
  "request_id": "chat_001_c8_t1",
  "task": "chat",
  "concurrency": 8,
  "system": "eagle3",
  "gpu_type": "L4",
  "trial": 1,
  "ttft_ms": 142.3,
  "tpot_ms": 18.7,
  "total_latency_ms": 1823.4,
  "output_tokens": 97,
  "tokens_per_sec": 53.2,
  "acceptance_rate": 0.41,
  "gpu_cost_usd": 0.000042
}
```
* **Why this schema:** It captures all variables required for our final report (Time to First Token, Tokens Per Second, Acceptance Rate, and Cost) at the *per-request* level, allowing for standard deviation analysis and granular plotting.

---

## 6. Repository Layout

```text
Project2/
├── infra/
│   ├── startup_baseline.sh      # GPU Baseline
│   ├── startup_eagle3.sh        # GPU Speculative
│   ├── startup_mlx_baseline.sh  # Mac Baseline
│   └── startup_mlx_spec.sh      # Mac Speculative
├── data/
│   └── prepare_datasets.py      # Downloads & samples 50 prompts
├── benchmark/
│   ├── load_test.py             # Main async load generator
│   └── plot_results.py          # Generates final PNGs and CSVs
├── results/
│   ├── raw/                     # JSONL output logs
│   ├── plots/                   # Generated graphs
│   └── tables/                  # Generated CSV summaries
├── BENCHMARKING_RUNBOOK.md      # Step-by-step experiment instructions
└── ARCHITECTURE.md              # This document
```
