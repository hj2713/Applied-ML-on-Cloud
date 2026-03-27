# Project 1: Performance Modeling and Analysis
## Deadline: March 27, 11:59 PM

### 1. Data Collection & Setup (15% - Measurements)
- [x] Run PyTorch benchmark suite (`benchmark_roofline.py`) on L4.
- [x] Run PyTorch benchmark suite (`benchmark_roofline.py`) on T4.
- [x] Run Nsight Compute single-run proof on L4.
- [ ] Run Nsight Compute single-run proof on T4. (Currently Running)
- [ ] Aggregate average `images_per_s`, `achieved_tflops`, and `gpu_mem_util_mean_pct` from both GPUs.

### 2. Experiment Design (10%)
- [ ] Define the Objective: Compare the performance of cloud GPU architectures (Turing T4 vs. Ada Lovelace L4) across different Neural Network models.
- [ ] State the Hypothesis: Define how changes in hardware (higher compute on L4) and workload (ResNet vs ViT) shift the bottlenecks from compute-bound to memory-bound.

### 3. Complexity Estimation (20%)
- [ ] Calculate NN Complexity: Extract the parameter counts and sizes for the 2-3 chosen models (e.g., ResNet50, ViT).
- [ ] Estimate Arithmetic Intensity (FLOPs per Byte = Total FLOPs / Total Bytes transfered from memory per iteration).

### 4. Roofline Modeling (20%)
**CRITICAL:** The rubric states creating our own roofline model from measurements is *mandatory*, while the Nsight tool is *optional for comparison*.
- [ ] Research Hardware Theoretical Peaks: Get exact Peak TFLOPS and Peak Memory Bandwidth (GB/s) for NVIDIA T4 and NVIDIA L4.
- [ ] Write Python Plotting Script: Create a `plot_roofline.py` using `matplotlib` to draw the ridge point log-log graphs.
- [ ] Overlay Empirical Data: Plot the PyTorch measured data points (`achieved_tflops` vs `approx_operational_intensity`) from `metrics.csv` onto the drawn ceilings.
- [ ] Compare with Nsight: Use the Nsight CSV data to see if the hardware profiler matches our manual empirical plot.

### 5. Analysis & Final Report (35%)
- [ ] Hardware Analysis: Explain how the architecture (T4 vs L4) influenced the location of the model points on the graph. (Did the L4's massive compute ceiling make the models hit a memory wall?)
- [ ] Workload Analysis: Discuss how the choice of NN (e.g., CNNS vs Transformers) changed the arithmetic intensity and performance outcomes.
- [ ] Compile the final PDF/Docx report with graphs and tables.