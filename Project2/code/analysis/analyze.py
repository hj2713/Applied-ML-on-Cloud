"""
Analysis script — reads all JSONL result files and produces plots + tables.

Outputs:
  results/plots/ttft_vs_concurrency_{task}_{gpu}.png
  results/plots/throughput_vs_concurrency_{task}_{gpu}.png
  results/plots/acceptance_rate_vs_task.png
  results/plots/cost_per_token_{gpu}.png
  results/tables/summary.csv
  results/tables/crossover_points.csv

Usage:
    python analysis/analyze.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RESULTS_DIR = Path("results/raw")
PLOTS_DIR = Path("results/plots")
TABLES_DIR = Path("results/tables")

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# ── Load all results ───────────────────────────────────────────────────────────

def load_all_results() -> pd.DataFrame:
    records = []
    for path in RESULTS_DIR.glob("*.jsonl"):
        with open(path) as f:
            for line in f:
                try:
                    records.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

    if not records:
        raise FileNotFoundError(f"No .jsonl files found in {RESULTS_DIR}")

    df = pd.DataFrame(records)
    df = df[df["error"].isna()] if "error" in df.columns else df
    df = df[df["ttft_ms"].notna()]
    return df


# ── Aggregation ────────────────────────────────────────────────────────────────

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Average over trials per (system, gpu_type, task, concurrency)."""
    cols = ["system", "gpu_type", "task", "concurrency"]
    metrics = ["ttft_ms", "tpot_ms", "tokens_per_sec", "gpu_cost_usd", "acceptance_rate"]
    agg = df.groupby(cols)[metrics].mean().reset_index()
    return agg


# ── TTFT vs Concurrency ────────────────────────────────────────────────────────

def plot_ttft_vs_concurrency(agg: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for gpu in agg["gpu_type"].unique():
        for task in agg["task"].unique():
            subset = agg[(agg["gpu_type"] == gpu) & (agg["task"] == task)]
            if subset.empty:
                continue

            fig, ax = plt.subplots(figsize=(8, 5))
            for system, grp in subset.groupby("system"):
                grp = grp.sort_values("concurrency")
                ax.plot(
                    grp["concurrency"], grp["ttft_ms"],
                    marker="o", label=system.upper(), linewidth=2,
                )

            ax.set_title(f"TTFT vs Concurrency — {task.capitalize()} on {gpu}")
            ax.set_xlabel("Concurrent Requests")
            ax.set_ylabel("Time to First Token (ms)")
            ax.set_xticks([1, 4, 8, 16, 32])
            ax.legend()
            ax.set_yscale("log")  # log scale shows crossover clearly

            path = PLOTS_DIR / f"ttft_vs_concurrency_{task}_{gpu}.png"
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"  Saved: {path}")


# ── Throughput vs Concurrency ──────────────────────────────────────────────────

def plot_throughput_vs_concurrency(agg: pd.DataFrame) -> None:
    for gpu in agg["gpu_type"].unique():
        for task in agg["task"].unique():
            subset = agg[(agg["gpu_type"] == gpu) & (agg["task"] == task)]
            if subset.empty:
                continue

            fig, ax = plt.subplots(figsize=(8, 5))
            for system, grp in subset.groupby("system"):
                grp = grp.sort_values("concurrency")
                ax.plot(
                    grp["concurrency"], grp["tokens_per_sec"],
                    marker="s", label=system.upper(), linewidth=2,
                )

            ax.set_title(f"Throughput vs Concurrency — {task.capitalize()} on {gpu}")
            ax.set_xlabel("Concurrent Requests")
            ax.set_ylabel("Tokens / Second (per request)")
            ax.set_xticks([1, 4, 8, 16, 32])
            ax.legend()

            path = PLOTS_DIR / f"throughput_vs_concurrency_{task}_{gpu}.png"
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"  Saved: {path}")


# ── Acceptance Rate vs Task ────────────────────────────────────────────────────

def plot_acceptance_rate(agg: pd.DataFrame) -> None:
    eagle3 = agg[(agg["system"] == "eagle3") & agg["acceptance_rate"].notna()]
    if eagle3.empty:
        print("  No acceptance rate data for EAGLE-3 — skipping plot")
        return

    pivot = eagle3.groupby(["task", "concurrency"])["acceptance_rate"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    for task, grp in pivot.groupby("task"):
        grp = grp.sort_values("concurrency")
        ax.plot(grp["concurrency"], grp["acceptance_rate"], marker="o", label=task, linewidth=2)

    ax.set_title("EAGLE-3 Draft Token Acceptance Rate vs Concurrency")
    ax.set_xlabel("Concurrent Requests")
    ax.set_ylabel("Acceptance Rate")
    ax.set_xticks([1, 4, 8, 16, 32])
    ax.set_ylim(0, 1)
    ax.axhline(0.5, linestyle="--", color="gray", alpha=0.5, label="50% threshold")
    ax.legend()

    path = PLOTS_DIR / "acceptance_rate_vs_concurrency.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Cost per Token ─────────────────────────────────────────────────────────────

def plot_cost_per_token(df: pd.DataFrame) -> None:
    df = df.copy()
    df = df[df["output_tokens"] > 0]
    df["cost_per_token"] = df["gpu_cost_usd"] / df["output_tokens"]

    agg = df.groupby(["system", "gpu_type", "concurrency"])["cost_per_token"].mean().reset_index()

    for gpu in agg["gpu_type"].unique():
        subset = agg[agg["gpu_type"] == gpu]
        if subset.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        for system, grp in subset.groupby("system"):
            grp = grp.sort_values("concurrency")
            ax.plot(
                grp["concurrency"], grp["cost_per_token"] * 1e6,  # convert to micro-dollars
                marker="^", label=system.upper(), linewidth=2,
            )

        ax.set_title(f"Cost per Output Token — {gpu}")
        ax.set_xlabel("Concurrent Requests")
        ax.set_ylabel("Cost per Token (µUSD)")
        ax.set_xticks([1, 4, 8, 16, 32])
        ax.legend()

        path = PLOTS_DIR / f"cost_per_token_{gpu}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")


# ── Crossover Point Table ──────────────────────────────────────────────────────

def compute_crossover(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Find the concurrency level where EAGLE-3 TTFT exceeds baseline TTFT.
    This is the key finding: 'At X concurrency, EAGLE-3 stops helping.'
    """
    rows = []
    for gpu in agg["gpu_type"].unique():
        for task in agg["task"].unique():
            base = agg[(agg["system"] == "baseline") & (agg["gpu_type"] == gpu) & (agg["task"] == task)]
            eagle = agg[(agg["system"] == "eagle3") & (agg["gpu_type"] == gpu) & (agg["task"] == task)]

            if base.empty or eagle.empty:
                continue

            merged = base[["concurrency", "ttft_ms"]].merge(
                eagle[["concurrency", "ttft_ms"]],
                on="concurrency", suffixes=("_base", "_eagle3")
            ).sort_values("concurrency")

            crossover = None
            for _, row in merged.iterrows():
                if row["ttft_ms_eagle3"] > row["ttft_ms_base"]:
                    crossover = int(row["concurrency"])
                    break

            rows.append({
                "gpu_type": gpu,
                "task": task,
                "crossover_concurrency": crossover if crossover else ">32",
                "eagle3_wins_below": crossover - 1 if crossover and crossover > 1 else "all tested",
            })

    return pd.DataFrame(rows)


# ── Summary table ──────────────────────────────────────────────────────────────

def save_summary(agg: pd.DataFrame) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / "summary.csv"
    agg.to_csv(path, index=False)
    print(f"  Saved: {path}")


def main() -> None:
    print("Loading results...")
    df = load_all_results()
    print(f"  Loaded {len(df)} request records")
    print(f"  Systems  : {df['system'].unique().tolist()}")
    print(f"  GPUs     : {df['gpu_type'].unique().tolist()}")
    print(f"  Tasks    : {df['task'].unique().tolist()}")
    print(f"  Concurrencies: {sorted(df['concurrency'].unique().tolist())}")

    agg = aggregate(df)

    print("\nGenerating plots...")
    plot_ttft_vs_concurrency(agg)
    plot_throughput_vs_concurrency(agg)
    plot_acceptance_rate(agg)
    plot_cost_per_token(df)

    print("\nComputing crossover points...")
    crossover_df = compute_crossover(agg)
    if not crossover_df.empty:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        crossover_path = TABLES_DIR / "crossover_points.csv"
        crossover_df.to_csv(crossover_path, index=False)
        print(crossover_df.to_string(index=False))
        print(f"\n  Saved: {crossover_path}")

    print("\nSaving summary table...")
    save_summary(agg)

    print("\nDone.")


if __name__ == "__main__":
    main()
