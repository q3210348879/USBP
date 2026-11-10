"""
benchmark_analysis.py
=====================
Analyze and visualize the computational overhead comparison between
DivideMix (baseline) and DivideMix + USBP.

Outputs:
  1. A summary table printed to console and saved as CSV
     (benchmark_summary.csv)
  2. Publication-quality figures saved as PDF:
     - Fig A: Per-epoch training time comparison
     - Fig B: Per-epoch eval (selection) time comparison
     - Fig C: Peak GPU memory comparison
     - Fig D: Test accuracy trajectory (bonus)
     - Fig E: Combined 2x2 panel figure (paper-ready)
     - Fig F: Bar chart of average overhead (paper-ready)

Usage:
    python benchmark_analysis.py \
        --baseline benchmark_baseline_cifar100_sym_0.8.csv \
        --usbp     benchmark_USBP_cifar100_sym_0.8.csv \
        --outdir   ./benchmark_figs
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# =====================================================================
# Args
# =====================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--baseline', type=str,
                    default='benchmark_baseline_cifar100_sym_0.8.csv')
parser.add_argument('--usbp', type=str,
                    default='benchmark_USBP_cifar100_sym_0.8.csv')
parser.add_argument('--outdir', type=str, default='./benchmark_figs')
parser.add_argument('--warmup_epochs', type=int, default=30,
                    help='Number of warmup epochs to exclude from main-phase analysis')
parser.add_argument('--smooth_window', type=int, default=5,
                    help='Window size for moving average smoothing in plots')
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

# =====================================================================
# Load data
# =====================================================================
df_base = pd.read_csv(args.baseline)
df_usbp = pd.read_csv(args.usbp)

# Separate warmup and main training phases
base_warmup = df_base[df_base['phase'] == 'warmup']
base_train  = df_base[df_base['phase'] != 'warmup']

usbp_warmup = df_usbp[df_usbp['phase'] == 'warmup']
usbp_train  = df_usbp[df_usbp['phase'] != 'warmup']

# =====================================================================
# 1. Summary Statistics
# =====================================================================
def compute_stats(df, label):
    """Compute mean +/- std for main training phase metrics."""
    return {
        'Method': label,
        'Train Time (s/epoch)':    f"{df['train_time_s'].mean():.2f} ± {df['train_time_s'].std():.2f}",
        'Eval Time (s/epoch)':     f"{df['eval_time_s'].mean():.2f} ± {df['eval_time_s'].std():.2f}",
        'Total Time (s/epoch)':    f"{(df['train_time_s'] + df['eval_time_s']).mean():.2f} ± {(df['train_time_s'] + df['eval_time_s']).std():.2f}",
        'Test Time (s)':           f"{df['test_time_s'].mean():.2f} ± {df['test_time_s'].std():.2f}",
        'Test Latency (ms/img)':   f"{df['test_latency_ms_per_img'].mean():.4f} ± {df['test_latency_ms_per_img'].std():.4f}",
        'Peak GPU (MB)':           f"{df['peak_gpu_mb'].mean():.1f} ± {df['peak_gpu_mb'].std():.1f}",
        'Best Acc (%)':            f"{df['test_acc'].max():.2f}",
        'Last 10 Avg Acc (%)':     f"{df['test_acc'].tail(10).mean():.2f}",
    }

stats_base = compute_stats(base_train, 'DivideMix')
stats_usbp = compute_stats(usbp_train, 'DivideMix + USBP')

summary_df = pd.DataFrame([stats_base, stats_usbp])
summary_df = summary_df.set_index('Method')
print("\n" + "=" * 80)
print("BENCHMARK SUMMARY  (CIFAR-100, 80% symmetric noise, main training phase)")
print("=" * 80)
print(summary_df.to_string())
print("=" * 80)

# Compute overhead ratios
base_total_mean = (base_train['train_time_s'] + base_train['eval_time_s']).mean()
usbp_total_mean = (usbp_train['train_time_s'] + usbp_train['eval_time_s']).mean()
overhead_ratio = usbp_total_mean / base_total_mean

base_gpu_mean = base_train['peak_gpu_mb'].mean()
usbp_gpu_mean = usbp_train['peak_gpu_mb'].mean()
gpu_overhead = (usbp_gpu_mean - base_gpu_mean) / base_gpu_mean * 100

print(f"\nTime overhead ratio (USBP / Baseline): {overhead_ratio:.2f}x")
print(f"GPU memory overhead: +{gpu_overhead:.1f}%")
print(f"Accuracy improvement: +{usbp_train['test_acc'].tail(10).mean() - base_train['test_acc'].tail(10).mean():.2f}%")

summary_csv_path = os.path.join(args.outdir, 'benchmark_summary.csv')
summary_df.to_csv(summary_csv_path)
print(f"\nSummary saved to: {summary_csv_path}")

# =====================================================================
# Helper: smoothing
# =====================================================================
def smooth(y, window=5):
    """Simple moving average smoothing."""
    if len(y) < window:
        return y
    return pd.Series(y).rolling(window=window, center=True, min_periods=1).mean().values

# =====================================================================
# Plot style setup (publication quality)
# =====================================================================
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

COLOR_BASE = '#2196F3'   # Blue
COLOR_USBP = '#F44336'   # Red
ALPHA_RAW = 0.15
LW = 1.8
W = args.smooth_window

# =====================================================================
# 2. Individual Figures
# =====================================================================

# --- Fig A: Per-epoch Training Time ---
fig, ax = plt.subplots(figsize=(6, 3.5))
epochs_b = base_train['epoch'].values
epochs_u = usbp_train['epoch'].values

ax.fill_between(epochs_b, base_train['train_time_s'].values, alpha=ALPHA_RAW, color=COLOR_BASE)
ax.plot(epochs_b, smooth(base_train['train_time_s'].values, W), color=COLOR_BASE, lw=LW, label='DivideMix')

ax.fill_between(epochs_u, usbp_train['train_time_s'].values, alpha=ALPHA_RAW, color=COLOR_USBP)
ax.plot(epochs_u, smooth(usbp_train['train_time_s'].values, W), color=COLOR_USBP, lw=LW, label='DivideMix + USBP')

ax.set_xlabel('Epoch')
ax.set_ylabel('Training Time (s / epoch)')
ax.set_title('Per-Epoch Training Time')
ax.legend()
ax.grid(True, alpha=0.3)
fig.savefig(os.path.join(args.outdir, 'fig_train_time.pdf'))
fig.savefig(os.path.join(args.outdir, 'fig_train_time.png'))
plt.close(fig)

# --- Fig B: Per-epoch Eval Time ---
fig, ax = plt.subplots(figsize=(6, 3.5))

ax.fill_between(epochs_b, base_train['eval_time_s'].values, alpha=ALPHA_RAW, color=COLOR_BASE)
ax.plot(epochs_b, smooth(base_train['eval_time_s'].values, W), color=COLOR_BASE, lw=LW, label='DivideMix')

ax.fill_between(epochs_u, usbp_train['eval_time_s'].values, alpha=ALPHA_RAW, color=COLOR_USBP)
ax.plot(epochs_u, smooth(usbp_train['eval_time_s'].values, W), color=COLOR_USBP, lw=LW, label='DivideMix + USBP')

ax.set_xlabel('Epoch')
ax.set_ylabel('Eval / Selection Time (s / epoch)')
ax.set_title('Per-Epoch Evaluation Time (GMM + MC Dropout)')
ax.legend()
ax.grid(True, alpha=0.3)
fig.savefig(os.path.join(args.outdir, 'fig_eval_time.pdf'))
fig.savefig(os.path.join(args.outdir, 'fig_eval_time.png'))
plt.close(fig)

# --- Fig C: Peak GPU Memory ---
fig, ax = plt.subplots(figsize=(6, 3.5))

ax.fill_between(epochs_b, base_train['peak_gpu_mb'].values, alpha=ALPHA_RAW, color=COLOR_BASE)
ax.plot(epochs_b, smooth(base_train['peak_gpu_mb'].values, W), color=COLOR_BASE, lw=LW, label='DivideMix')

ax.fill_between(epochs_u, usbp_train['peak_gpu_mb'].values, alpha=ALPHA_RAW, color=COLOR_USBP)
ax.plot(epochs_u, smooth(usbp_train['peak_gpu_mb'].values, W), color=COLOR_USBP, lw=LW, label='DivideMix + USBP')

ax.set_xlabel('Epoch')
ax.set_ylabel('Peak GPU Memory (MB)')
ax.set_title('Peak GPU Memory per Epoch')
ax.legend()
ax.grid(True, alpha=0.3)
fig.savefig(os.path.join(args.outdir, 'fig_gpu_memory.pdf'))
fig.savefig(os.path.join(args.outdir, 'fig_gpu_memory.png'))
plt.close(fig)

# --- Fig D: Test Accuracy ---
fig, ax = plt.subplots(figsize=(6, 3.5))

# Include warmup epochs too
all_epochs_b = df_base['epoch'].values
all_epochs_u = df_usbp['epoch'].values

ax.plot(all_epochs_b, smooth(df_base['test_acc'].values, W), color=COLOR_BASE, lw=LW, label='DivideMix')
ax.plot(all_epochs_u, smooth(df_usbp['test_acc'].values, W), color=COLOR_USBP, lw=LW, label='DivideMix + USBP')

ax.set_xlabel('Epoch')
ax.set_ylabel('Test Accuracy (%)')
ax.set_title('Test Accuracy Trajectory')
ax.legend()
ax.grid(True, alpha=0.3)
fig.savefig(os.path.join(args.outdir, 'fig_test_accuracy.pdf'))
fig.savefig(os.path.join(args.outdir, 'fig_test_accuracy.png'))
plt.close(fig)

# =====================================================================
# 3. Combined 2x2 Panel Figure (paper-ready)
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(10, 7))

# (a) Training Time
ax = axes[0, 0]
ax.fill_between(epochs_b, base_train['train_time_s'].values, alpha=ALPHA_RAW, color=COLOR_BASE)
ax.plot(epochs_b, smooth(base_train['train_time_s'].values, W), color=COLOR_BASE, lw=LW, label='DivideMix')
ax.fill_between(epochs_u, usbp_train['train_time_s'].values, alpha=ALPHA_RAW, color=COLOR_USBP)
ax.plot(epochs_u, smooth(usbp_train['train_time_s'].values, W), color=COLOR_USBP, lw=LW, label='DivideMix + USBP')
ax.set_ylabel('Time (s)')
ax.set_title('(a) Training Time per Epoch')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (b) Eval Time
ax = axes[0, 1]
ax.fill_between(epochs_b, base_train['eval_time_s'].values, alpha=ALPHA_RAW, color=COLOR_BASE)
ax.plot(epochs_b, smooth(base_train['eval_time_s'].values, W), color=COLOR_BASE, lw=LW, label='DivideMix')
ax.fill_between(epochs_u, usbp_train['eval_time_s'].values, alpha=ALPHA_RAW, color=COLOR_USBP)
ax.plot(epochs_u, smooth(usbp_train['eval_time_s'].values, W), color=COLOR_USBP, lw=LW, label='DivideMix + USBP')
ax.set_ylabel('Time (s)')
ax.set_title('(b) Eval / Selection Time per Epoch')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (c) Peak GPU Memory
ax = axes[1, 0]
ax.fill_between(epochs_b, base_train['peak_gpu_mb'].values, alpha=ALPHA_RAW, color=COLOR_BASE)
ax.plot(epochs_b, smooth(base_train['peak_gpu_mb'].values, W), color=COLOR_BASE, lw=LW, label='DivideMix')
ax.fill_between(epochs_u, usbp_train['peak_gpu_mb'].values, alpha=ALPHA_RAW, color=COLOR_USBP)
ax.plot(epochs_u, smooth(usbp_train['peak_gpu_mb'].values, W), color=COLOR_USBP, lw=LW, label='DivideMix + USBP')
ax.set_xlabel('Epoch')
ax.set_ylabel('Memory (MB)')
ax.set_title('(c) Peak GPU Memory')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (d) Test Accuracy
ax = axes[1, 1]
ax.plot(all_epochs_b, smooth(df_base['test_acc'].values, W), color=COLOR_BASE, lw=LW, label='DivideMix')
ax.plot(all_epochs_u, smooth(df_usbp['test_acc'].values, W), color=COLOR_USBP, lw=LW, label='DivideMix + USBP')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy (%)')
ax.set_title('(d) Test Accuracy')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(args.outdir, 'fig_combined_2x2.pdf'))
fig.savefig(os.path.join(args.outdir, 'fig_combined_2x2.png'))
plt.close(fig)

# =====================================================================
# 4. Bar Chart: Average Overhead Summary (paper-ready)
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

bar_width = 0.35
methods = ['DivideMix', 'DivideMix\n+ USBP']
x = np.arange(len(methods))

# (a) Total time per epoch (train + eval)
base_total = base_train['train_time_s'] + base_train['eval_time_s']
usbp_total = usbp_train['train_time_s'] + usbp_train['eval_time_s']
means = [base_total.mean(), usbp_total.mean()]
stds  = [base_total.std(), usbp_total.std()]

ax = axes[0]
bars = ax.bar(x, means, bar_width * 2, yerr=stds, capsize=4,
              color=[COLOR_BASE, COLOR_USBP], alpha=0.85, edgecolor='black', linewidth=0.5)
ax.set_ylabel('Time (s / epoch)')
ax.set_title('(a) Total Time per Epoch')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=9)
ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=9)
ax.set_ylim(0, max(means) * 1.3)
ax.grid(axis='y', alpha=0.3)

# (b) Peak GPU Memory
means_gpu = [base_train['peak_gpu_mb'].mean(), usbp_train['peak_gpu_mb'].mean()]
stds_gpu  = [base_train['peak_gpu_mb'].std(), usbp_train['peak_gpu_mb'].std()]

ax = axes[1]
bars = ax.bar(x, means_gpu, bar_width * 2, yerr=stds_gpu, capsize=4,
              color=[COLOR_BASE, COLOR_USBP], alpha=0.85, edgecolor='black', linewidth=0.5)
ax.set_ylabel('Memory (MB)')
ax.set_title('(b) Peak GPU Memory')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=9)
ax.bar_label(bars, fmt='%.0f', padding=3, fontsize=9)
ax.set_ylim(0, max(means_gpu) * 1.2)
ax.grid(axis='y', alpha=0.3)

# (c) Test Latency
means_lat = [base_train['test_latency_ms_per_img'].mean(), usbp_train['test_latency_ms_per_img'].mean()]
stds_lat  = [base_train['test_latency_ms_per_img'].std(), usbp_train['test_latency_ms_per_img'].std()]

ax = axes[2]
bars = ax.bar(x, means_lat, bar_width * 2, yerr=stds_lat, capsize=4,
              color=[COLOR_BASE, COLOR_USBP], alpha=0.85, edgecolor='black', linewidth=0.5)
ax.set_ylabel('Latency (ms / image)')
ax.set_title('(c) Test Latency')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=9)
ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
ax.set_ylim(0, max(means_lat) * 1.4)
ax.grid(axis='y', alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(args.outdir, 'fig_overhead_bars.pdf'))
fig.savefig(os.path.join(args.outdir, 'fig_overhead_bars.png'))
plt.close(fig)

# =====================================================================
# 5. Overhead breakdown table (for paper)
# =====================================================================
print("\n" + "=" * 80)
print("OVERHEAD BREAKDOWN (suitable for paper Table)")
print("=" * 80)

breakdown_data = {
    'Metric': [
        'Train time (s/epoch)',
        'Eval time (s/epoch)',
        'Total time (s/epoch)',
        'Test latency (ms/img)',
        'Peak GPU memory (MB)',
        'Best test acc (%)',
    ],
    'DivideMix': [
        f"{base_train['train_time_s'].mean():.2f}",
        f"{base_train['eval_time_s'].mean():.2f}",
        f"{base_total.mean():.2f}",
        f"{base_train['test_latency_ms_per_img'].mean():.4f}",
        f"{base_train['peak_gpu_mb'].mean():.0f}",
        f"{base_train['test_acc'].max():.2f}",
    ],
    'DivideMix + USBP': [
        f"{usbp_train['train_time_s'].mean():.2f}",
        f"{usbp_train['eval_time_s'].mean():.2f}",
        f"{usbp_total.mean():.2f}",
        f"{usbp_train['test_latency_ms_per_img'].mean():.4f}",
        f"{usbp_train['peak_gpu_mb'].mean():.0f}",
        f"{usbp_train['test_acc'].max():.2f}",
    ],
    'Overhead': [
        f"{(usbp_train['train_time_s'].mean() / base_train['train_time_s'].mean()):.2f}x",
        f"{(usbp_train['eval_time_s'].mean() / base_train['eval_time_s'].mean()):.2f}x",
        f"{overhead_ratio:.2f}x",
        f"{(usbp_train['test_latency_ms_per_img'].mean() / base_train['test_latency_ms_per_img'].mean()):.2f}x",
        f"+{gpu_overhead:.1f}%",
        f"+{usbp_train['test_acc'].max() - base_train['test_acc'].max():.2f}%",
    ]
}

breakdown_df = pd.DataFrame(breakdown_data)
print(breakdown_df.to_string(index=False))

breakdown_csv_path = os.path.join(args.outdir, 'benchmark_breakdown.csv')
breakdown_df.to_csv(breakdown_csv_path, index=False)
print(f"\nBreakdown table saved to: {breakdown_csv_path}")

print(f"\nAll figures saved to: {args.outdir}/")
print("  - fig_train_time.pdf / .png")
print("  - fig_eval_time.pdf / .png")
print("  - fig_gpu_memory.pdf / .png")
print("  - fig_test_accuracy.pdf / .png")
print("  - fig_combined_2x2.pdf / .png   <-- recommended for paper")
print("  - fig_overhead_bars.pdf / .png   <-- recommended for paper")
print("\nDone!")
