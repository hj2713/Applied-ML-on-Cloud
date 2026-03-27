# Applied ML on Cloud: Empirical Roofline Analysis & Hardware Complexity

**Author:** Himanshu Jhawar (hj2713)

## Project Overview
This repository contains the codebase, profiling outputs, and final report for an empirical systems-level profiling assignment. The core objective is to map standard PyTorch deep learning models (ResNet-50, VGG-16, ViT-B/16) to their absolute hardware bottlenecks on two distinct NVIDIA datacenter GPUs: Turing T4 and Ada Lovelace L4.

We bypass accuracy checks and focus strictly on executing 40 training iterations to gather sub-millisecond hardware telemetry using **PyTorch Kineto Profiler** and **NVIDIA Nsight Compute (`ncu`)**. We utilize these metrics to programmatically plot **Roofline Models**.

## Repository Structure
- `src/`: Contains the `benchmark_roofline.py` which orchestrates runs across hyperparameters and outputs profiling telemetry. Also contains `plot_rooflines.py` to generate the graphical models.
- `outputs/`: 
  - `T4_output/` and `L4_output/`: Contains the raw PyTorch profiler traces, loss histories, `metrics.csv`, and GPU `nvidia-smi` polling snapshots.
  - `nsight/`: Contains the exported Nsight `.csv`, `.ncu-rep` traces, and `.pdf` reports showcasing SM-level utilization.
- `report/`: Contains `Project_Report.md` (and `Project_Report.pdf`), analyzing the Ridge Points, Operational Intensity shifts based on NN architecture, and the definitive proof of execution overhead via Nsight Compute.

## Key Findings
1. **Compute-Bound Operations:** Image-based Deep Learning networks at batch sizes 32/128 execute firmly in the Compute-Bound regime, avoiding mathematical memory-bandwidth limits.
2. **Architecture Geometry:** Ancient models like VGG-16 stress memory excessively, dragging their Operational Intensity leftward (towards memory bounds), whereas modern Self-Attention in Vision Transformers exploits extreme data-reuse, pushing the plot far to the right.
3. **The "Ceiling Gap":** Despite being compute-bound, NVIDIA Nsight Trace explicitly proves PyTorch rarely saturates its SM arrays due to high fragmentation and kernel launch overhead (e.g., launching 9,000+ tiny `unrolled_elementwise_kernel` functions that only utilize ~2% SM capacity).
