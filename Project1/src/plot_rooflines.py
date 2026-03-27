#!/usr/bin/env python3
"""Roofline Modeling Plot Generation Script.

This script mathematically projects the memory and compute ceilings for
the NVIDIA T4 and L4 datacenter GPUs based on their theoretical peaks.
It then reads empirical profiling metrics from PyTorch Benchmark traces,
plotting them log-linearly against the drawn hardware ceilings to determine
if workloads are memory-bound or compute-bound.

Usage:
    python src/plot_rooflines.py
    python src/plot_rooflines.py --t4-metrics outputs/T4_output/metrics.csv --l4-metrics outputs/L4_output/metrics.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --- Hardware Specifications ---
# NVIDIA T4 Spec:
T4_PEAK_FP32_TFLOPS = 8.1
T4_PEAK_FP16_TFLOPS = 65.0
T4_BANDWIDTH_GB_S = 320.0

# NVIDIA L4 Spec:
L4_PEAK_FP32_TFLOPS = 30.3
L4_PEAK_FP16_TFLOPS = 120.0 # Tensor cores
L4_BANDWIDTH_GB_S = 300.0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for robust path overrides."""
    parser = argparse.ArgumentParser(description="Generate Mathematical Roofline Models.")
    parser.add_argument(
        "--t4-metrics", 
        type=Path, 
        default=Path("outputs/T4_output/metrics.csv"), 
        help="Path to T4 PyTorch metrics CSV"
    )
    parser.add_argument(
        "--l4-metrics", 
        type=Path, 
        default=Path("outputs/L4_output/metrics.csv"), 
        help="Path to L4 PyTorch metrics CSV"
    )
    return parser.parse_args()


def plot_roofline_for_gpu(
    gpu_name: str, 
    metrics_file: Path, 
    peak_fp32: float, 
    peak_fp16: float, 
    bandwidth: float
) -> None:
    """Draws a clean, mathematically accurate Roofline plot.
    
    Args:
        gpu_name: Title identifier (e.g., 'T4' or 'L4').
        metrics_file: Path to the generated CSV containing empirical FLOP and bandwidth data.
        peak_fp32: The theoretical maximum TFLOPS for FP32 pipelines.
        peak_fp16: The theoretical maximum TFLOPS for FP16 Tensor Cores.
        bandwidth: The theoretical maximum Memory Bandwidth in GB/s.
    """
    x = np.logspace(-1, 5, 1000)
    bw_tb_s = bandwidth / 1000.0
    
    y_fp32 = np.minimum(peak_fp32, bw_tb_s * x)
    y_fp16 = np.minimum(peak_fp16, bw_tb_s * x)
    
    plt.figure(figsize=(14, 8))
    
    # Mathematical Ceilings
    plt.plot(x, y_fp32, label=f'Theoretical Peak FP32 ({peak_fp32} TFLOPS)', color='#1f77b4', linewidth=2.5)
    plt.plot(x, y_fp16, label=f'Theoretical Peak FP16 Tensor ({peak_fp16} TFLOPS)', color='#d62728', linewidth=2.5, linestyle='--')
    plt.plot(x, bw_tb_s * x, label=f'Memory Bandwidth Limit ({bandwidth} GB/s)', color='gray', linewidth=1.5, linestyle=':')
    
    # Empirical Data Plots
    if not metrics_file.exists():
        print(f"Warning: Empirical data file '{metrics_file}' not found. Plotting theoretical ceilings only.")
    else:
        try:
            df = pd.read_csv(metrics_file)
            
            # Formatting dictionaries for specific models/precisions
            markers: Dict[str, str] = {'resnet50': 'o', 'vit_b_16': 's', 'vgg16': '^'}
            colors: Dict[str, str] = {'fp32': '#1f77b4', 'fp16': '#d62728'}
            
            # Sort by performance so legend renders ordered
            if 'achieved_tflops' in df.columns:
                df = df.sort_values(by='achieved_tflops', ascending=False)
            
            for _, row in df.iterrows():
                oi = float(row.get('approx_operational_intensity_flops_per_byte', 0))
                flops = float(row.get('achieved_tflops', 0))
                model = str(row.get('model', 'unknown'))
                precision = str(row.get('precision', 'unknown'))
                batch_size = int(row.get('batch_size', 0))
                
                marker = markers.get(model.lower(), 'D')
                color = colors.get(precision.lower(), 'black')
                
                label = f"{model.upper()} ({precision}, BS={batch_size}) ➔ {flops:.2f} TFLOPS"
                
                if oi > 0 and flops > 0:
                    plt.scatter(
                        oi, flops, color=color, marker=marker, s=180, edgecolors='black', 
                        linewidths=1.2, zorder=5, label=label, alpha=0.9
                    )
                
        except Exception as e:
            print(f"Error parsing empirical data from '{metrics_file}': {e}", file=sys.stderr)

    # Scaling and Grid
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlim(0.1, 80000)
    plt.ylim(0.01, peak_fp16 * 2.5)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    
    # Labeling
    plt.title(f'Roofline Model - {gpu_name} (Target vs Achieved)', fontsize=20, fontweight='bold', pad=15)
    plt.xlabel('Operational Intensity (FLOPs / Byte)', fontsize=15)
    plt.ylabel('Performance (TFLOP / s)', fontsize=15)
    
    # Legend Placement
    handles, labels_list = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_list, handles))
    
    if by_label:
        plt.legend(
            by_label.values(), by_label.keys(), bbox_to_anchor=(1.02, 1), 
            loc='upper left', fontsize=12, framealpha=1, title="Measured Performance Matrix", title_fontsize=14
        )
    
    plt.tight_layout()
    output_filename = f'outputs/{gpu_name}_roofline.png'
    
    # Ensure outputs directory exists
    Path("outputs").mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully generated {output_filename}")


def main() -> int:
    args = parse_args()
    
    plot_roofline_for_gpu("T4", args.t4_metrics, T4_PEAK_FP32_TFLOPS, T4_PEAK_FP16_TFLOPS, T4_BANDWIDTH_GB_S)
    plot_roofline_for_gpu("L4", args.l4_metrics, L4_PEAK_FP32_TFLOPS, L4_PEAK_FP16_TFLOPS, L4_BANDWIDTH_GB_S)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())