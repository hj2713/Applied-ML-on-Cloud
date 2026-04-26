"""
Live benchmark dashboard — Streamlit app.

Shows side-by-side comparison of baseline vs EAGLE-3 (or MLX variants).
Can run in two modes:
  1. LIVE MODE   — fires real requests to running servers, shows real-time metrics
  2. RESULTS MODE — loads from results/raw/*.jsonl, shows analysis plots

Usage:
    streamlit run dashboard/app.py
"""

import asyncio
import json
import time
from pathlib import Path

import aiohttp
import pandas as pd
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EAGLE-3 Benchmark Dashboard",
    page_icon="⚡",
    layout="wide",
)

RESULTS_DIR = Path("results/raw")

GPU_HOURLY_RATES = {
    "T4":   0.35,
    "L4":   0.70,
    "A100": 3.67,
    "mac":  0.00,
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("⚡ EAGLE-3 Benchmark")
mode = st.sidebar.radio("Mode", ["Live Demo", "Results Analysis"])

# ── LIVE DEMO MODE ─────────────────────────────────────────────────────────────
if mode == "Live Demo":
    st.title("⚡ Live: Baseline vs EAGLE-3")
    st.caption("Fires real requests to both servers simultaneously and shows latency metrics.")

    # Server config
    st.sidebar.subheader("Server Config")
    baseline_url = st.sidebar.text_input("Baseline URL", "http://localhost:8000")
    eagle3_url   = st.sidebar.text_input("EAGLE-3 URL",  "http://localhost:8001")
    gpu_type     = st.sidebar.selectbox("GPU Type", ["L4", "A100", "T4", "mac"])

    # Request config
    st.sidebar.subheader("Request Config")
    task = st.sidebar.selectbox("Task", ["chat", "code", "summarization"])
    concurrency = st.sidebar.select_slider("Concurrency", [1, 4, 8, 16, 32], value=4)
    max_tokens = st.sidebar.slider("Max output tokens", 64, 512, 256)

    # Default prompts by task
    DEFAULT_PROMPTS = {
        "chat": "Explain the concept of recursion in simple terms.",
        "code": "Write a Python function that checks if a string is a palindrome.",
        "summarization": "Summarize the following: The Apollo 11 mission was the first crewed lunar landing mission. Launched on July 16, 1969, it carried astronauts Neil Armstrong, Buzz Aldrin, and Michael Collins. Armstrong and Aldrin landed on the Moon on July 20, while Collins orbited above. Armstrong became the first person to walk on the Moon.",
    }
    prompt = st.text_area("Prompt", value=DEFAULT_PROMPTS[task], height=120)

    async def send_one(session, url, model_name, prompt_text, gpu_type, system):
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
        }
        t_start = time.perf_counter()
        t_first = None
        text = ""
        tokens = 0
        try:
            async with session.post(
                f"{url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                async for raw in resp.content:
                    line = raw.decode().strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        content = chunk["choices"][0]["delta"].get("content", "")
                        if content:
                            if t_first is None:
                                t_first = time.perf_counter()
                            text += content
                            tokens += 1
                    except Exception:
                        continue
        except Exception as e:
            return {"error": str(e), "ttft_ms": None, "tpot_ms": None, "tokens_per_sec": None, "cost_usd": None, "text": ""}

        t_end = time.perf_counter()
        total = t_end - t_start
        ttft = (t_first - t_start) * 1000 if t_first else None
        tpot = ((t_end - t_first) / (tokens - 1) * 1000) if t_first and tokens > 1 else None
        cost = GPU_HOURLY_RATES.get(gpu_type, 0) / 3600 * total

        return {
            "ttft_ms": round(ttft, 1) if ttft else None,
            "tpot_ms": round(tpot, 1) if tpot else None,
            "tokens_per_sec": round(tokens / total, 1) if total > 0 else None,
            "cost_usd": cost,
            "output_tokens": tokens,
            "text": text,
            "error": None,
        }

    async def run_live_demo(prompt_text, concurrency_n):
        # Fire concurrency_n copies of the same request to both servers
        prompts = [prompt_text] * concurrency_n
        connector = aiohttp.TCPConnector(limit=concurrency_n * 2 + 4)
        async with aiohttp.ClientSession(connector=connector) as session:
            baseline_tasks = [send_one(session, baseline_url, "baseline", p, gpu_type, "baseline") for p in prompts]
            eagle3_tasks   = [send_one(session, eagle3_url,  "eagle3",   p, gpu_type, "eagle3")   for p in prompts]
            all_results = await asyncio.gather(*baseline_tasks, *eagle3_tasks)

        baseline_results = list(all_results[:concurrency_n])
        eagle3_results   = list(all_results[concurrency_n:])
        return baseline_results, eagle3_results

    def avg(lst, key):
        vals = [r[key] for r in lst if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    if st.button("▶  Run Benchmark", type="primary"):
        with st.spinner(f"Firing {concurrency} concurrent requests to both servers..."):
            b_results, e_results = asyncio.run(run_live_demo(prompt, concurrency))

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Baseline (greedy)")
            ttft  = avg(b_results, "ttft_ms")
            tpot  = avg(b_results, "tpot_ms")
            tps   = avg(b_results, "tokens_per_sec")
            cost  = sum(r.get("cost_usd", 0) for r in b_results)
            st.metric("Avg TTFT",        f"{ttft:.0f} ms"   if ttft  else "N/A")
            st.metric("Avg TPOT",        f"{tpot:.1f} ms"   if tpot  else "N/A")
            st.metric("Avg Tokens/sec",  f"{tps:.1f}"       if tps   else "N/A")
            st.metric("Total cost",      f"${cost:.6f}")
            st.metric("Acceptance rate", "N/A (baseline)")
            if b_results[0].get("text"):
                st.text_area("Sample output", b_results[0]["text"][:400], height=150)

        with col2:
            st.subheader("EAGLE-3 (speculative)")
            ttft  = avg(e_results, "ttft_ms")
            tpot  = avg(e_results, "tpot_ms")
            tps   = avg(e_results, "tokens_per_sec")
            cost  = sum(r.get("cost_usd", 0) for r in e_results)
            st.metric("Avg TTFT",        f"{ttft:.0f} ms"   if ttft  else "N/A")
            st.metric("Avg TPOT",        f"{tpot:.1f} ms"   if tpot  else "N/A")
            st.metric("Avg Tokens/sec",  f"{tps:.1f}"       if tps   else "N/A")
            st.metric("Total cost",      f"${cost:.6f}")
            st.metric("Acceptance rate", "see /metrics")
            if e_results[0].get("text"):
                st.text_area("Sample output", e_results[0]["text"][:400], height=150)

        errors = [r for r in b_results + e_results if r.get("error")]
        if errors:
            st.error(f"{len(errors)} request(s) failed: {errors[0]['error']}")


# ── RESULTS ANALYSIS MODE ──────────────────────────────────────────────────────
else:
    st.title("Results Analysis")

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    @st.cache_data
    def load_results():
        records = []
        for p in RESULTS_DIR.glob("*.jsonl"):
            with open(p) as f:
                for line in f:
                    try:
                        r = json.loads(line.strip())
                        if r.get("ttft_ms") is not None:
                            records.append(r)
                    except Exception:
                        continue
        return pd.DataFrame(records)

    if not any(RESULTS_DIR.glob("*.jsonl")):
        st.warning("No results found in results/raw/. Run the benchmark sweep first.")
    else:
        df = load_results()
        st.success(f"Loaded {len(df)} request records from {RESULTS_DIR}")

        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            gpu_filter = st.multiselect("GPU Type", df["gpu_type"].unique().tolist(), default=df["gpu_type"].unique().tolist())
        with col2:
            task_filter = st.multiselect("Task", df["task"].unique().tolist(), default=df["task"].unique().tolist())
        with col3:
            system_filter = st.multiselect("System", df["system"].unique().tolist(), default=df["system"].unique().tolist())

        df_filtered = df[
            df["gpu_type"].isin(gpu_filter) &
            df["task"].isin(task_filter) &
            df["system"].isin(system_filter)
        ]

        agg = df_filtered.groupby(["system", "gpu_type", "task", "concurrency"])[
            ["ttft_ms", "tpot_ms", "tokens_per_sec", "gpu_cost_usd"]
        ].mean().reset_index()

        # ── TTFT plot ──────────────────────────────────────────
        st.subheader("Time to First Token vs Concurrency")
        for gpu in agg["gpu_type"].unique():
            for task in agg["task"].unique():
                subset = agg[(agg["gpu_type"] == gpu) & (agg["task"] == task)]
                if subset.empty:
                    continue

                fig, ax = plt.subplots(figsize=(7, 4))
                for system, grp in subset.groupby("system"):
                    grp = grp.sort_values("concurrency")
                    ax.plot(grp["concurrency"], grp["ttft_ms"], marker="o", label=system, linewidth=2)

                ax.set_title(f"TTFT — {task} on {gpu}")
                ax.set_xlabel("Concurrent Requests")
                ax.set_ylabel("TTFT (ms)")
                ax.set_xticks([1, 4, 8, 16, 32])
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

        # ── Throughput plot ─────────────────────────────────────
        st.subheader("Throughput vs Concurrency")
        for gpu in agg["gpu_type"].unique():
            for task in agg["task"].unique():
                subset = agg[(agg["gpu_type"] == gpu) & (agg["task"] == task)]
                if subset.empty:
                    continue

                fig, ax = plt.subplots(figsize=(7, 4))
                for system, grp in subset.groupby("system"):
                    grp = grp.sort_values("concurrency")
                    ax.plot(grp["concurrency"], grp["tokens_per_sec"], marker="s", label=system, linewidth=2)

                ax.set_title(f"Tokens/sec — {task} on {gpu}")
                ax.set_xlabel("Concurrent Requests")
                ax.set_ylabel("Tokens / Second")
                ax.set_xticks([1, 4, 8, 16, 32])
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

        # ── Raw data table ──────────────────────────────────────
        with st.expander("Raw aggregated data"):
            st.dataframe(agg)

        # ── Cost table ──────────────────────────────────────────
        st.subheader("Cost Analysis")
        df_filtered_cost = df_filtered[df_filtered["output_tokens"] > 0].copy()
        df_filtered_cost["cost_per_token_usd"] = df_filtered_cost["gpu_cost_usd"] / df_filtered_cost["output_tokens"]
        cost_table = df_filtered_cost.groupby(["system", "gpu_type"])["cost_per_token_usd"].mean().reset_index()
        cost_table["cost_per_1k_tokens_usd"] = cost_table["cost_per_token_usd"] * 1000
        st.dataframe(cost_table[["system", "gpu_type", "cost_per_1k_tokens_usd"]].round(6))
