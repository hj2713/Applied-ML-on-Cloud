"""
Concurrency sweep load tester.

Fires N concurrent streaming requests to a vLLM server and records:
  - TTFT  (time to first token)
  - TPOT  (time per output token)
  - Total latency
  - Output token count
  - Tokens per second
  - Cost (GPU hourly rate × latency)

Output: results/raw/{system}_{gpu}_{task}_c{concurrency}_t{trial}.jsonl

Usage:
    # Single run (one concurrency level, one task, one system)
    python benchmark/load_test.py \
        --server-url http://localhost:8000 \
        --system baseline \
        --gpu-type L4 \
        --task chat \
        --prompts-file data/prompts/chat_50.jsonl \
        --concurrency 8 \
        --trial 1

    # Full sweep (all combinations) — let run_sweep.sh call this
    python benchmark/load_test.py --sweep
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp

# ── GPU hourly rates (GCP on-demand, USD) ─────────────────────────────────────
GPU_HOURLY_RATES = {
    "T4":   0.35,   # g4dn.xlarge equivalent
    "L4":   0.70,   # g2-standard-8
    "A100": 3.67,   # a2-highgpu-1g
    "mac":  0.00,   # MLX team — no cloud cost
}

RESULTS_DIR = Path("results/raw")
MAX_OUTPUT_TOKENS = 256  # cap output length to control experiment duration

# Model names sent in the API request (must match --served-model-name in startup scripts)
# mlx_lm.server uses the model path as the model name
SYSTEM_MODEL_NAMES = {
    "baseline":     "baseline",
    "eagle3":       "eagle3",
    "mlx_baseline": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "mlx_spec":     "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
}


# ── Core request function ──────────────────────────────────────────────────────

async def send_request(
    session: aiohttp.ClientSession,
    server_url: str,
    prompt: dict,
    system: str,
    gpu_type: str,
    concurrency: int,
    trial: int,
) -> dict:
    """
    Send one streaming chat request and return a metrics dict.

    Streaming is required to measure TTFT accurately — we can't get TTFT
    from a non-streaming response because it only returns after full generation.
    """
    model_name = SYSTEM_MODEL_NAMES[system]
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt["prompt"]}],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0,   # greedy — deterministic output for quality comparison
        "stream": True,
    }

    t_start = time.perf_counter()
    t_first_token = None
    output_text = ""
    output_tokens = 0

    try:
        async with session.post(
            f"{server_url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            resp.raise_for_status()

            async for raw_line in resp.content:
                line = raw_line.decode("utf-8").strip()

                # SSE format: lines start with "data: "
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")

                if content:
                    if t_first_token is None:
                        t_first_token = time.perf_counter()
                    output_text += content
                    output_tokens += 1  # approximate: 1 chunk ≈ 1 token

        t_end = time.perf_counter()

    except Exception as exc:
        t_end = time.perf_counter()
        return {
            "request_id": f"{prompt['id']}_c{concurrency}_t{trial}",
            "task": prompt["task"],
            "concurrency": concurrency,
            "system": system,
            "gpu_type": gpu_type,
            "trial": trial,
            "error": str(exc),
            "ttft_ms": None,
            "tpot_ms": None,
            "total_latency_ms": round((t_end - t_start) * 1000, 2),
            "output_tokens": 0,
            "tokens_per_sec": None,
            "gpu_cost_usd": _compute_cost(gpu_type, t_end - t_start),
        }

    total_latency = t_end - t_start
    ttft = (t_first_token - t_start) if t_first_token else None
    # TPOT = time spent generating tokens after first / (tokens - 1)
    tpot = None
    if t_first_token and output_tokens > 1:
        tpot = (t_end - t_first_token) / (output_tokens - 1)

    return {
        "request_id": f"{prompt['id']}_c{concurrency}_t{trial}",
        "task": prompt["task"],
        "concurrency": concurrency,
        "system": system,
        "gpu_type": gpu_type,
        "trial": trial,
        "ttft_ms": round(ttft * 1000, 2) if ttft else None,
        "tpot_ms": round(tpot * 1000, 2) if tpot else None,
        "total_latency_ms": round(total_latency * 1000, 2),
        "output_tokens": output_tokens,
        "tokens_per_sec": round(output_tokens / total_latency, 2) if total_latency > 0 else None,
        "gpu_cost_usd": _compute_cost(gpu_type, total_latency),
        "output_text": output_text,  # kept for quality check; strip later if storage is a concern
    }


def _compute_cost(gpu_type: str, latency_sec: float) -> float:
    rate = GPU_HOURLY_RATES.get(gpu_type, 0.0)
    return round(rate / 3600 * latency_sec, 8)


# ── Concurrency cell runner ────────────────────────────────────────────────────

async def run_cell(
    server_url: str,
    prompts: list[dict],
    system: str,
    gpu_type: str,
    concurrency: int,
    trial: int,
) -> list[dict]:
    """
    Fire exactly `concurrency` requests simultaneously and wait for all to finish.

    We pick the first `concurrency` prompts from the list.
    This simulates exactly N users hitting the server at the same moment.
    """
    batch = prompts[:concurrency]

    # One shared HTTP session for connection reuse
    connector = aiohttp.TCPConnector(limit=concurrency + 4)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            send_request(session, server_url, p, system, gpu_type, concurrency, trial)
            for p in batch
        ]
        results = await asyncio.gather(*tasks)

    return list(results)


# ── Acceptance rate from /metrics endpoint ─────────────────────────────────────

async def fetch_acceptance_rate(server_url: str) -> float | None:
    """
    Read server-wide speculative decoding acceptance rate from Prometheus metrics.

    This is a running average across all requests served so far.
    Call it once after each cell completes to capture the cell-level estimate.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{server_url}/metrics", timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                text = await resp.text()

        # Try to find the direct gauge first (older vLLM)
        for line in text.splitlines():
            if "spec_decode_draft_acceptance_rate" in line and not line.startswith("#"):
                return round(float(line.split()[-1]), 4)

        # Fallback for vLLM V1: calculate from raw counters
        accepted = None
        drafted = None
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            if "vllm:spec_decode_num_accepted_tokens_total" in line:
                accepted = float(line.split()[-1])
            if "vllm:spec_decode_num_draft_tokens_total" in line:
                drafted = float(line.split()[-1])

        if accepted is not None and drafted is not None and drafted > 0:
            return round(accepted / drafted, 4)
    except Exception:
        pass
    return None


# ── Save results ───────────────────────────────────────────────────────────────

def save_results(records: list[dict], system: str, gpu_type: str, task: str, concurrency: int, trial: int) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{system}_{gpu_type}_{task}_c{concurrency:02d}_t{trial}.jsonl"
    path = RESULTS_DIR / filename
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


# ── CLI ────────────────────────────────────────────────────────────────────────

def load_prompts(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


async def main_single(args: argparse.Namespace) -> None:
    prompts = load_prompts(args.prompts_file)

    print(f"\nRunning: system={args.system} gpu={args.gpu_type} task={args.task} "
          f"concurrency={args.concurrency} trial={args.trial}")
    print(f"  Server : {args.server_url}")
    print(f"  Prompts: {len(prompts)} loaded, using first {args.concurrency}")

    results = await run_cell(
        server_url=args.server_url,
        prompts=prompts,
        system=args.system,
        gpu_type=args.gpu_type,
        concurrency=args.concurrency,
        trial=args.trial,
    )

    # Attach acceptance rate (only meaningful for eagle3)
    if args.system == "eagle3":
        acc_rate = await fetch_acceptance_rate(args.server_url)
        for r in results:
            r["acceptance_rate"] = acc_rate
    else:
        for r in results:
            r["acceptance_rate"] = None

    path = save_results(results, args.system, args.gpu_type, args.task, args.concurrency, args.trial)

    # Print summary
    valid = [r for r in results if r.get("ttft_ms") is not None]
    if valid:
        avg_ttft = sum(r["ttft_ms"] for r in valid) / len(valid)
        avg_tps = sum(r["tokens_per_sec"] for r in valid if r["tokens_per_sec"]) / len(valid)
        total_cost = sum(r["gpu_cost_usd"] for r in results)
        print(f"\nResults:")
        print(f"  Requests completed : {len(valid)}/{len(results)}")
        print(f"  Avg TTFT           : {avg_ttft:.1f} ms")
        print(f"  Avg tokens/sec     : {avg_tps:.1f}")
        print(f"  Total cost (cell)  : ${total_cost:.6f}")
        if args.system == "eagle3":
            print(f"  Acceptance rate    : {results[0].get('acceptance_rate')}")
    print(f"\nSaved → {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM concurrency load tester")
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--system", choices=["baseline", "eagle3", "mlx_baseline", "mlx_spec"], required=True)
    parser.add_argument("--gpu-type", choices=["T4", "L4", "A100", "mac"], required=True)
    parser.add_argument("--task", choices=["chat", "code", "summarization"], required=True)
    parser.add_argument("--prompts-file", required=True, help="Path to .jsonl prompts file")
    parser.add_argument("--concurrency", type=int, required=True, choices=[1, 4, 8, 16, 32])
    parser.add_argument("--trial", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main_single(args))
