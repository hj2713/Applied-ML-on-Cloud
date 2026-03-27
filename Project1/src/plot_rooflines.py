import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Hardware Specifications ---
# NVIDIA T4 Spec:
T4_PEAK_FP32_TFLOPS = 8.1
T4_PEAK_FP16_TFLOPS = 65.0
T4_BANDWIDTH_GB_S = 320.0

# NVIDIA L4 Spec:
L4_PEAK_FP32_TFLOPS = 30.3
L4_PEAK_FP16_TFLOPS = 120.0 # Tensor cores
L4_BANDWIDTH_GB_S = 300.0

def plot_roofline_for_gpu(gpu_name, metrics_file, peak_fp32, peak_fp16, bandwidth):
    """
    Draws a cleaner, visually clear Roofline plot with no overlapping labels.
    """
    x = np.logspace(-1, 5, 1000)
    bw_tb_s = bandwidth / 1000.0
    
    y_fp32 = np.minimum(peak_fp32, bw_tb_s * x)
    y_fp16 = np.minimum(peak_fp16, bw_tb_s * x)
    
    # Use larger figure space
    plt.figure(figsize=(14, 8))
    
    # Rooflines
    plt.plot(x, y_fp32, label=f'Theoretical Peak FP32 ({peak_fp32} TFLOPS)', color='#1f77b4', linewidth=2.5)
    plt.plot(x, y_fp16, label=f'Theoretical Peak FP16 Tensor ({peak_fp16} TFLOPS)', color='#d62728', linewidth=2.5, linestyle='--')
    
    # Diagonal bandwidth limit outlined
    plt.plot(x, bw_tb_s * x, label=f'Memory Bandwidth Limit ({bandwidth} GB/s)', color='gray', linewidth=1.5, linestyle=':')
    
    try:
        df = pd.read_csv(metrics_file)
        
        # Consistent shapes & colors
        markers = {'resnet50': 'o', 'vit_b_16': 's', 'vgg16': '^'}
        colors = {'fp32': '#1f77b4', 'fp16': '#d62728'}
        
        # Sort values so they display properly ordered in the legend by highest performance
        df = df.sort_values(by='achieved_tflops', ascending=False)
        
        for index, row in df.iterrows():
            oi = row['approx_operational_intensity_flops_per_byte']
            flops = row['achieved_tflops']
            model = row['model']
            precision = row['precision']
            batch_size = row['batch_size']
            
            marker = markers.get(model.lower(), 'D')
            color = colors.get(precision.lower(), 'black')
            
            # Format label mapping marker/color directly to performance
            label = f"{model.upper()} ({precision}, BS={batch_size}) ➔ {flops:.2f} TFLOPS"
            
            plt.scatter(oi, flops, color=color, marker=marker, s=180, edgecolors='black', 
                        linewidths=1.2, zorder=5, label=label, alpha=0.9)
            
    except Exception as e:
        print(f"Could not read empirical data from {metrics_file}: {e}")

    plt.xscale('log')
    plt.yscale('log')
    
    # Add good margin spacing around edges
    plt.xlim(0.1, 80000)
    plt.ylim(0.01, peak_fp16 * 2.5)
    
    plt.grid(True, which="both", ls="--", alpha=0.4)
    
    plt.title(f'Roofline Model - {gpu_name} (Target vs Achieved)', fontsize=20, fontweight='bold', pad=15)
    plt.xlabel('Operational Intensity (FLOPs / Byte)', fontsize=15)
    plt.ylabel('Performance (TFLOP / s)', fontsize=15)
    
    # Organize legend perfectly to the side so nothing overlaps chart
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.02, 1), 
               loc='upper left', fontsize=12, framealpha=1, title="Measured Performance Matrix", title_fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'{gpu_name}_roofline.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully generated clean {gpu_name}_roofline.png")

if __name__ == "__main__":
    plot_roofline_for_gpu("T4", "outputs/T4_output/metrics.csv", T4_PEAK_FP32_TFLOPS, T4_PEAK_FP16_TFLOPS, T4_BANDWIDTH_GB_S)
    plot_roofline_for_gpu("L4", "outputs/L4_output/metrics.csv", L4_PEAK_FP32_TFLOPS, L4_PEAK_FP16_TFLOPS, L4_BANDWIDTH_GB_S)