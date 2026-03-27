# Applied ML on Cloud: Empirical Roofline Analysis & Hardware Complexity

**Author:** Himanshu Jhawar, hj2713  
**Repository:** [https://github.com/hj2713/Applied-ML-on-Cloud/tree/main](https://github.com/hj2713/Applied-ML-on-Cloud/tree/main/Project1

## 1. Experiment Design

**Objective:** The primary goal of this experiment is to empirically investigate and mathematically model the performance bottlenecks of deep neural networks executing on datacenter GPUs. By deploying standard architectures onto two distinctly different generations of NVIDIA hardware (Turing T4 vs. Ada Lovelace L4), we aim to map exactly how close modern PyTorch workloads can get to theoretical hardware limits, and determine if they are memory-bound or compute-bound.

**Hypotheses:**

1. **Hardware Generational Leap:** The L4 GPU, boasting significantly higher theoretical compute (TFLOPS) but similar memory bandwidth to the T4, will exhibit much higher absolute throughput, pushing models closer to the memory-bound regime.
2. **Architecture Influence:** Legacy, parameter-heavy architectures like VGG-16 will stress memory bandwidth far more than modern, matrix-multiplication-heavy attention architectures like ViT, which should exhibit high arithmetic intensity.
3. **Precision Scaling:** Utilizing FP16 mixed precision will activate specialized Tensor Cores, massively raising the theoretical and achieved compute ceiling compared to native FP32 operations.

Implementation limits: We conducted short, representative runs (40 iterations) on the `imagenet_subset` dataset. Training until full convergence was unnecessary as systems-level hardware utilization (throughput, not accuracy) is the focus.

---

## 2. Complexity Estimation

To understand the hardware pressure exerted by our workloads, we must firmly establish the theoretical complexity of our chosen Neural Network models prior to execution. We focus on the number of trainable parameters (which dictates memory footprint) and the Floating Point Operations (FLOPs) required per image (which dictates compute pressure).

Our workload selection was highly intentional, spanning three distinct architectural paradigms:

1. **ResNet-50 (The Baseline CNN):** A modern, balanced convolutional network utilizing residual connections.
2. **VGG-16 (The Memory Bottleneck):** An older CNN known for its extremely dense, parameter-heavy Fully-Connected layers at the tail end of the network. It serves as our worst-case scenario for memory pressure.
3. **Vision Transformer, ViT-B/16 (The Compute Bound Matrix):** A modern attention-based architecture. Rather than relying on convolutions, it breaks images into patches and performs massive dense matrix multiplications (self-attention) across them.

_Table 1: Estimated Complexity of the Workloads (Derived directly from PyTorch Profiler traces on batch size 32)_

| Model         | Parameter Count | Parameter Size (FP32) | Est. Compute Complexity (FLOPs / Image) | Architectural Paradigm      |
| :------------ | :-------------- | :-------------------- | :-------------------------------------- | :-------------------------- |
| **ResNet-50** | ~23.9 Million   | ~91.2 MB              | ~10.2 GFLOPs                            | Balanced Convolutional      |
| **ViT-B/16**  | ~85.9 Million   | ~327.8 MB             | ~125.8 GFLOPs                           | Dense Matrix/Self-Attention |
| **VGG-16**    | ~135.0 Million  | ~515.2 MB             | ~39.3 GFLOPs                            | Dense Fully-Connected       |

**Complexity Conclusion:** VGG-16 possesses the largest sheer memory footprint to load, while ViT requires vastly more mathematical operations to process a single image. Consequently, we expect ViT to yield high Operational Intensity (spending more time calculating than fetching data) and VGG-16 to yield the lowest.

---

## 3. Measurement

We constructed a modular benchmarking pipeline capable of hooking directly into the PyTorch Profiler (Kineto). To isolate hardware behavior from framework overhead, we logged specific hardware metrics via `nvidia-smi` async polling and captured absolute kernel traces:

**Data Collected:**

- **Achieved TFLOPS:** Measured directly by counting absolute floating-point operations executing in the PyTorch graph over time.
- **Data Transferred (Bytes):** Tracked by parsing the tensor memory accesses during the forward/backward passes.
- **Operational Intensity (FLOPs/Byte):** Calculated by dividing total FLOPs executed by total memory bytes accessed. This serves as the X-axis for our Roofline Models.
- **Hardware Telemetry:** Sideloaded captures of GPU Utilization (%), GPU Memory Utilization (%), and power throughput during the 40-iteration windows.

---

## 4. Roofline Modeling

The Roofline Model mathematically visualizes whether a workload is restricted by memory speed (the slanted ceiling) or pure processing speed (the flat ceiling). The exact point these ceilings intersect is the **Ridge Point**.

| Metric                 | **NVIDIA T4 (Turing)** | **NVIDIA L4 (Ada Lovelace)** |
| :--------------------- | :--------------------- | :--------------------------- |
| **DRAM Bandwidth**     | 320 GB/s (0.32 TB/s)   | 300 GB/s (0.30 TB/s)         |
| **Peak FP32 Compute**  | 8.1 TFLOPS             | 30.3 TFLOPS                  |
| **Peak FP16 (Tensor)** | 65.0 TFLOPS            | 120.0 TFLOPS                 |
| **Ridge Point (FP32)** | **25.31 FLOPs/Byte**   | **101.0 FLOPs/Byte**         |
| **Ridge Point (FP16)** | **203.12 FLOPs/Byte**  | **400.0 FLOPs/Byte**         |

Using our empirical measurements, we programmatically plotted the performance of all 12 experimental configurations against the theoretical limits of both the T4 and L4.

### 4.1 NVIDIA T4 Execution (Cloud Baseline)

![T4 Roofline Model](../outputs/T4_roofline.png)

### 4.2 NVIDIA L4 Execution (Ada Lovelace Upgrade)

![L4 Roofline Model](../outputs/L4_roofline.png)

_Note: All empirical test points (Operational Intensities strictly > 700 FLOPs/Byte) mathematically reside far to the right of the Ridge Points, placing our entire Deep Learning workload firmly in the **Compute-Bound** regime._

---

## 5. Analysis

### 5.1 Environmental Influence: T4 vs. L4 Hardware

Focusing on the Y-axis (Achieved TFLOPS), the generational leap from Turing to Ada Lovelace profoundly influences raw performance. When deploying the exact same PyTorch code (ViT-B/16, BS=128, FP16), the T4 capped out at roughly **13.88 TFLOPS**. In contrast, the L4 reached **20.29 TFLOPS** purely due to a higher Streaming Multiprocessor (SM) density.

However, raising the hardware ceiling exposes a severe **"Ceiling Gap"**. Despite being mathematically Compute-Bound (we are feeding the GPU data fast enough), neither card approaches its maximum theoretical limit (65 and 120 TFLOPS respectively). This indicates bottlenecking not in the hardware limits themselves, but rather in _software framework overhead_ and kernel launch latencies failing to fully saturate the massive GPU arrays.

### 5.2 The Influence of NN Model Choices on the Roofline

Our Complexity Estimation (Section 2) hypothesized that architectural choices fundamentally shift a model's operational intensity (X-axis) and achieved performance (Y-axis). The empirical plots prove this:

1.  **VGG-16 (The Memory Anchor):** As predicted, VGG-16 consistently plots the furthest left on the X-axis (lowest Operational Intensity at ~766 FLOP/Byte for FP32/BS=32). Its massive 135M parameter fully-connected layers force the GPU to spend exorbitant amounts of time fetching weights from DRAM rather than executing math, dragging it closer to the memory-bound slant.
2.  **ResNet-50 (The Middle Ground):** Maps perfectly in the middle (~1067 FLOP/Byte). Its efficient convolutional blocks allow higher data reuse in SRAM compared to VGG.
3.  **ViT-B/16 (The Compute Juggernaut):** Plots the furthest right (~3830 FLOP/Byte up to incredible ~29k values depending on batching). Attention mechanisms divide data into sequence patches and perform `(Q * K.T) * V` matrix multiplications, reusing the loaded block of data exponentially more times than a CNN. This results in the highest theoretical arithmetic intensity and consequently commands the highest absolute TFLOPS on both architectures.

### 5.3 Micro-Architectural Diagnostics (Nsight Compute Analysis)

To explicitly diagnose the theoretical "Ceiling Gap" visually prominent in our Roofline models, we conducted sub-millisecond hardware profiling using NVIDIA Nsight Compute (`.ncu-rep`).

By opening our trace in the Nsight GUI and inspecting the **GPU Speed Of Light (SOL)** and **Workload Analysis** panels, we proved exactly why PyTorch fails to reach theoretical max TFLOPS:

_(Nsight Compute showing the "GPU Speed of Light" panel overall summary, highlighting the low Compute and Memory %)_
![T4 Nsight Analysis Output](../outputs/T4_output/nsight/t4_nsight.png)

_(Nsight Compute showing the exact bottlenecked kernel warning about low occupancy/wavefronts)_
![L4 Nsight Analysis Output](../outputs/L4_output/nsight/l4_nsight.png)

The Nsight Trace unequivocally demonstrated that the execution timeline is saturated with over 9,000 invocations of microscopic `unrolled_elementwise_kernels`. The Nsight profiler explicitly flagged these with warnings stating: _"This kernel grid is too small to fill the available resources on this device."_

## 6. Final Conclusion

Our empirical Roofline Modeling successfully proved deep learning training is Compute-Bound. However, our Nsight analysis pinpointed that PyTorch's reliance on launching thousands of fragmented, sequential kernels fundamentally caps the achievable TFLOPS at a fraction of the hardware's theoretical peak, as the SM arrays spend disproportionally high amounts of time idle during kernel launch latencies.
