import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os

def smooth_results(csv_path, sigma=3):
    if not os.path.exists(csv_path):
        print(f"Error: File not found {csv_path}")
        return

    df = pd.read_csv(csv_path)
    k = df['k'].values
    r2 = df['r2'].values
    alpha = df['alpha'].values
    
    # Apply Gaussian smoothing
    r2_smooth = gaussian_filter1d(r2, sigma=sigma)
    
    # Smooth alpha in log space since it varies by orders of magnitude
    log_alpha = np.log10(alpha)
    log_alpha_smooth = gaussian_filter1d(log_alpha, sigma=sigma)
    alpha_smooth = 10**log_alpha_smooth
    
    # Find peaks
    peak_idx = np.argmax(r2_smooth)
    peak_k = k[peak_idx]
    peak_r2 = r2_smooth[peak_idx]
    
    print(f"Smoothed Peak for {csv_path}:")
    print(f"  K = {peak_k}")
    print(f"  R^2 = {peak_r2:.4f}")
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 7))

    color = 'tab:blue'
    ax1.set_xlabel('Dimensions (K)')
    ax1.set_ylabel('LOOCV R^2 Score', color=color)
    ax1.plot(k, r2, color=color, alpha=0.3, label='Raw R^2')
    ax1.plot(k, r2_smooth, color=color, linewidth=2, label=f'Smoothed R^2 (sigma={sigma})')
    ax1.scatter(peak_k, peak_r2, color='black', zorder=5, label=f'Peak: K={peak_k}')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Optimal Alpha (log scale)', color=color)
    ax2.plot(k, alpha, color=color, alpha=0.3, linestyle='--', label='Raw Alpha')
    ax2.plot(k, alpha_smooth, color=color, linewidth=2, linestyle='--', label='Smoothed Alpha')
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Smoothed DNA Parametric Study\nFile: {os.path.basename(csv_path)}')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    fig.tight_layout()
    output_png = csv_path.replace('.csv', '_smoothed.png')
    plt.savefig(output_png)
    print(f"Smoothed plot saved to {output_png}")
    plt.show()

if __name__ == "__main__":
    steamid = "76561198039155404"
    files = [
        f"research/dna_parametric_results_{steamid}_subsample_50.csv",
        f"research/dna_parametric_results_{steamid}_subsample_100.csv",
        f"research/dna_parametric_results_{steamid}_subsample_1000.csv"
    ]
    
    for f in files:
        if os.path.exists(f):
            smooth_results(f, sigma=5) # sigma=5 for smoother trends across 243 points
