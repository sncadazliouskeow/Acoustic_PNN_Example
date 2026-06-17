"""
data_inspector.py — Physical Data Distribution Inspector

Purpose:
    Diagnose whether extreme outliers in physical experiment data are
    compressing the useful signal into a narrow range after normalization.

    Key questions answered:
      1. Are there a few huge values dominating the max?
      2. Which (channel, frequency) combinations are abnormally strong?
      3. After global-max normalization, how much of the range is actually used?
      4. Would per-dimension normalization help?
      5. Is the distribution consistent between train and test sets?
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.cm as cm

DATA_DIR         = 'data'
BASE_FREQUENCIES = [300, 450, 600, 750, 900]
NUM_CHANNELS     = 4
VOWEL_NAMES      = ['ae', 'ah', 'aw', 'uw', 'er', 'iy', 'ih']


# ==========================================
# Helpers
# ==========================================

def pick_files() -> dict[str, np.ndarray]:
    """
    Scan data/ and let user pick one or more files to inspect.
    Returns {name: ndarray}.
    """
    files = sorted([f[:-4] for f in os.listdir(DATA_DIR) if f.endswith('.csv')])
    if not files:
        print(f"[Error] No CSV files in '{DATA_DIR}/'.")
        sys.exit(1)

    print("\n  Available CSV files:")
    for i, f in enumerate(files):
        df = pd.read_csv(os.path.join(DATA_DIR, f"{f}.csv"), header=None)
        print(f"  [{i+1:>2}]  {f:<45}  ({len(df)} rows x {df.shape[1]} cols)")

    raw = input(
        "\n  Enter file numbers to inspect (comma-separated, e.g. 1,3,5): "
    ).strip()
    chosen = {}
    for tok in raw.split(','):
        try:
            idx = int(tok.strip()) - 1
            if 0 <= idx < len(files):
                name = files[idx]
                chosen[name] = pd.read_csv(
                    os.path.join(DATA_DIR, f"{name}.csv"), header=None
                ).values.astype(np.float32)
        except ValueError:
            pass
    if not chosen:
        print("[Error] No valid files selected.")
        sys.exit(1)
    return chosen


def dim_label(d: int) -> str:
    ch   = d // len(BASE_FREQUENCIES) + 1
    freq = BASE_FREQUENCIES[d % len(BASE_FREQUENCIES)]
    return f"ch{ch}\n{freq}Hz"


# ==========================================
# Module A: Global value distribution
# ==========================================

def plot_global_distribution(data_dict: dict[str, np.ndarray]):
    """
    For each file: histogram of ALL values + cumulative distribution.
    This reveals whether a handful of large values are pulling the max up.
    """
    n_files = len(data_dict)
    fig, axes = plt.subplots(n_files, 2,
                              figsize=(13, 3.5 * n_files + 1))
    if n_files == 1:
        axes = [axes]

    for row, (name, data) in enumerate(data_dict.items()):
        flat = data.flatten()
        ax_hist, ax_cdf = axes[row]

        # Histogram
        ax_hist.hist(flat, bins=200, color='steelblue', alpha=0.8,
                     edgecolor='none', density=True)
        ax_hist.axvline(flat.mean(), color='tomato', linestyle='--',
                        linewidth=1.5, label=f'mean={flat.mean():.4f}')
        ax_hist.axvline(np.percentile(flat, 99), color='orange',
                        linestyle=':', linewidth=1.5,
                        label=f'p99={np.percentile(flat,99):.4f}')
        ax_hist.axvline(flat.max(), color='red', linestyle='-',
                        linewidth=1.2, label=f'max={flat.max():.4f}')
        ax_hist.set_title(f'{name}  —  Value Histogram  '
                          f'(n={len(flat):,} values)')
        ax_hist.set_xlabel('Value')
        ax_hist.set_ylabel('Density')
        ax_hist.legend(fontsize=8)
        ax_hist.grid(True, alpha=0.25)

        # CDF (log x-scale to reveal what fraction sits below 0.1)
        sorted_vals = np.sort(flat)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax_cdf.plot(sorted_vals, cdf * 100,
                    color='steelblue', linewidth=1.5)
        for thr in [0.05, 0.1, 0.2, 0.5]:
            frac = (flat < thr).mean() * 100
            ax_cdf.axvline(thr, color='gray', linestyle=':', alpha=0.6)
            ax_cdf.text(thr, 5, f'{thr:.2f}\n({frac:.0f}%)',
                        fontsize=7, ha='center', color='gray')
        ax_cdf.set_title(f'{name}  —  Cumulative Distribution (CDF)')
        ax_cdf.set_xlabel('Value threshold')
        ax_cdf.set_ylabel('% of values below threshold')
        ax_cdf.set_xlim(left=0)
        ax_cdf.grid(True, alpha=0.25)

        # Print percentile summary
        print(f"\n  {name}:")
        print(f"    shape={data.shape}  "
              f"max={flat.max():.5f}  mean={flat.mean():.5f}  std={flat.std():.5f}")
        for p in [50, 75, 90, 95, 99, 99.9]:
            print(f"    p{p:>5.1f}: {np.percentile(flat, p):.5f}  "
                  f"({(flat > np.percentile(flat,p)).mean()*100:.2f}% above)")

    fig.suptitle('Global Value Distribution', fontsize=13)
    plt.tight_layout()
    _save_show(fig, 'inspect_global_distribution.png')


# ==========================================
# Module B: Per-dimension heatmap
# ==========================================

def plot_per_dim_heatmap(data_dict: dict[str, np.ndarray]):
    """
    For each file: heatmap of per-dimension stats (mean, std, max).
    Arranged as (n_channels x n_freqs) grid so spatial layout matches
    the physical microphone × frequency structure.

    This immediately shows which (microphone, frequency) pairs are anomalous.
    """
    stats_names = ['Mean', 'Std', 'Max']
    n_files = len(data_dict)
    n_stats = len(stats_names)

    fig, axes = plt.subplots(n_files * n_stats, 1,
                              figsize=(11, 2.8 * n_files * n_stats + 1))
    if n_files * n_stats == 1:
        axes = [axes]
    ax_iter = iter(axes)

    for name, data in data_dict.items():
        if data.shape[1] != NUM_CHANNELS * len(BASE_FREQUENCIES):
            print(f"  [Note] {name} has {data.shape[1]} cols, "
                  f"expected {NUM_CHANNELS * len(BASE_FREQUENCIES)}. "
                  f"Skipping per-dim heatmap.")
            continue

        stats = {
            'Mean': data.mean(axis=0),
            'Std' : data.std(axis=0),
            'Max' : data.max(axis=0),
        }

        for stat_name, vals in stats.items():
            mat = vals.reshape(NUM_CHANNELS, len(BASE_FREQUENCIES))
            ax  = next(ax_iter)

            im = ax.imshow(mat, cmap='YlOrRd', aspect='auto',
                           vmin=0, vmax=vals.max())
            plt.colorbar(im, ax=ax, fraction=0.03, pad=0.01)

            ax.set_xticks(range(len(BASE_FREQUENCIES)))
            ax.set_xticklabels([f'{f}Hz' for f in BASE_FREQUENCIES])
            ax.set_yticks(range(NUM_CHANNELS))
            ax.set_yticklabels([f'Mic {i+1}' for i in range(NUM_CHANNELS)])
            ax.set_title(f'{name}  —  Per (Mic × Freq)  [{stat_name}]',
                         fontsize=10)

            # Annotate each cell
            for i in range(NUM_CHANNELS):
                for j in range(len(BASE_FREQUENCIES)):
                    v = mat[i, j]
                    ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                            fontsize=8,
                            color='white' if v > vals.max() * 0.6 else 'black')

        # Print the dominant (mic, freq) combo
        max_dim = int(data.max(axis=0).argmax())
        ch = max_dim // len(BASE_FREQUENCIES) + 1
        fr = BASE_FREQUENCIES[max_dim % len(BASE_FREQUENCIES)]
        print(f"\n  {name}: dominant output = Mic {ch}, {fr} Hz  "
              f"(dim {max_dim}, max={data[:,max_dim].max():.5f})")

    fig.suptitle('Per (Microphone × Frequency) Statistics', fontsize=13)
    plt.tight_layout()
    _save_show(fig, 'inspect_per_dim_heatmap.png')


# ==========================================
# Module C: Outlier impact on normalization
# ==========================================

def plot_normalization_impact(data_dict: dict[str, np.ndarray]):
    """
    Show what happens to data distribution after global-max normalization
    vs per-dimension normalization.
    This quantifies how much useful range is lost to outliers.
    """
    n_files = len(data_dict)
    fig, axes = plt.subplots(n_files, 3,
                              figsize=(15, 4 * n_files + 1))
    if n_files == 1:
        axes = [axes]

    for row, (name, data) in enumerate(data_dict.items()):
        ax1, ax2, ax3 = axes[row]
        flat = data.flatten()

        # Global-max normalized
        g_max    = flat.max()
        g_normed = flat / g_max

        # Per-dim normalized (each dim: divide by its own max)
        dim_max  = data.max(axis=0, keepdims=True)
        dim_max  = np.where(dim_max < 1e-10, 1.0, dim_max)
        d_normed = (data / dim_max).flatten()

        # Panel 1: raw vs global-max
        bins = np.linspace(0, 1, 80)
        ax1.hist(g_normed, bins=bins, color='steelblue',
                 alpha=0.8, edgecolor='none', density=True)
        p90_g = np.percentile(g_normed, 90)
        ax1.axvline(p90_g, color='tomato', linestyle='--',
                    label=f'p90={p90_g:.3f}')
        frac_low = (g_normed < 0.1).mean() * 100
        ax1.set_title(
            f'{name}\nGlobal-max norm  '
            f'({frac_low:.0f}% of values < 0.10)',
            fontsize=9
        )
        ax1.set_xlabel('Normalized value')
        ax1.set_ylabel('Density')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.25)

        # Panel 2: per-dim normalized
        ax2.hist(d_normed, bins=bins, color='mediumseagreen',
                 alpha=0.8, edgecolor='none', density=True)
        p90_d = np.percentile(d_normed, 90)
        ax2.axvline(p90_d, color='tomato', linestyle='--',
                    label=f'p90={p90_d:.3f}')
        frac_low_d = (d_normed < 0.1).mean() * 100
        ax2.set_title(
            f'{name}\nPer-dim norm  '
            f'({frac_low_d:.0f}% of values < 0.10)',
            fontsize=9
        )
        ax2.set_xlabel('Normalized value')
        ax2.set_ylabel('Density')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.25)

        # Panel 3: per-dimension max bar chart (shows which dims are huge)
        dim_maxes = data.max(axis=0)
        colors    = cm.RdYlGn_r(Normalize()(dim_maxes))
        ax3.bar(range(len(dim_maxes)), dim_maxes,
                color=colors, edgecolor='none')
        ax3.set_title(f'{name}\nPer-dimension Max  '
                      f'(red = outlier driving global norm)',
                      fontsize=9)
        ax3.set_xlabel('Dimension  (ch1×freqs, ch2×freqs, ...)')
        ax3.set_ylabel('Max value across all rows')
        ax3.set_xticks(range(len(dim_maxes)))
        ax3.set_xticklabels(
            [dim_label(d) for d in range(len(dim_maxes))],
            fontsize=6, rotation=0
        )
        ax3.grid(True, alpha=0.25, axis='y')

        # Annotate the most extreme dim
        top3 = np.argsort(dim_maxes)[::-1][:3]
        print(f"\n  {name} — top 3 dims by max value:")
        for d in top3:
            ch = d // len(BASE_FREQUENCIES) + 1
            fr = BASE_FREQUENCIES[d % len(BASE_FREQUENCIES)]
            frac_of_max = dim_maxes[d] / dim_maxes.max()
            print(f"    dim {d:>2}  Mic {ch}  {fr} Hz: "
                  f"max={dim_maxes[d]:.5f}  "
                  f"({frac_of_max*100:.1f}% of global max)")

    fig.suptitle('Normalization Strategy Comparison', fontsize=13)
    plt.tight_layout()
    _save_show(fig, 'inspect_normalization_impact.png')


# ==========================================
# Module D: Train vs Test consistency
# ==========================================

def plot_train_test_consistency(data_dict: dict[str, np.ndarray]):
    """
    Compare per-dimension mean and std between train and test files.
    Large discrepancies explain train/test accuracy gaps.
    Requires at least 2 files to compare.
    """
    if len(data_dict) < 2:
        print("  [Note] Train-vs-test comparison needs >= 2 files. Skipping.")
        return

    names = list(data_dict.keys())
    datas = list(data_dict.values())

    # Use first two files for comparison
    n1, d1 = names[0], datas[0]
    n2, d2 = names[1], datas[1]

    n_dims = min(d1.shape[1], d2.shape[1])
    dims   = np.arange(n_dims)

    m1 = d1[:, :n_dims].mean(axis=0)
    s1 = d1[:, :n_dims].std(axis=0)
    m2 = d2[:, :n_dims].mean(axis=0)
    s2 = d2[:, :n_dims].std(axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))

    for ax, stat1, stat2, stat_name in [
        (axes[0], m1, m2, 'Mean'),
        (axes[1], s1, s2, 'Std'),
    ]:
        width = 0.4
        ax.bar(dims - width/2, stat1, width, label=n1,
               color='steelblue', alpha=0.8, edgecolor='none')
        ax.bar(dims + width/2, stat2, width, label=n2,
               color='tomato', alpha=0.8, edgecolor='none')
        ax.set_xlabel('Dimension  (ch1×freqs, ch2×freqs, ...)')
        ax.set_ylabel(stat_name)
        ax.set_title(f'Per-dimension {stat_name} Comparison:  {n1}  vs  {n2}')
        ax.set_xticks(dims)
        ax.set_xticklabels([dim_label(d) for d in dims], fontsize=6)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25, axis='y')

        # Highlight dims with large discrepancy
        ratio = np.abs(stat1 - stat2) / (np.maximum(stat1, stat2) + 1e-8)
        big   = np.where(ratio > 0.3)[0]
        for d in big:
            ax.axvspan(d - 0.5, d + 0.5, alpha=0.12, color='gold')

        if len(big) > 0:
            print(f"\n  Dims with >30% discrepancy in {stat_name}: {list(big)}")
            for d in big:
                print(f"    dim {d:>2}  Mic {d//5+1}  "
                      f"{BASE_FREQUENCIES[d%5]}Hz:  "
                      f"{n1}={stat1[d]:.4f}  {n2}={stat2[d]:.4f}  "
                      f"ratio={ratio[d]:.2f}")

    fig.suptitle(f'Train vs Test Consistency:  {n1}  vs  {n2}',
                 fontsize=13)
    plt.tight_layout()
    _save_show(fig, f'inspect_train_test_{n1[:12]}_vs_{n2[:12]}.png')


# ==========================================
# Util
# ==========================================

def _save_show(fig, fname: str):
    path = os.path.join(DATA_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.show()


# ==========================================
# Main
# ==========================================

def main():
    print("=" * 64)
    print("  Data Inspector  —  Physical Experiment Distribution Analyzer")
    print("=" * 64)

    data_dict = pick_files()

    while True:
        print("\n" + "-" * 64)
        print("  Select analysis:")
        print("  [1] Global value distribution  (histogram + CDF)")
        print("  [2] Per (Mic x Freq) heatmap   (find dominant dimensions)")
        print("  [3] Normalization impact        (global vs per-dim norm)")
        print("  [4] Train vs Test consistency   (requires 2 files selected)")
        print("  [5] Run all four")
        print("  [0] Exit")
        print("-" * 64)

        try:
            c = int(input("  Enter: ").strip())
        except ValueError:
            continue

        if c == 0:
            break
        elif c == 1:
            plot_global_distribution(data_dict)
        elif c == 2:
            plot_per_dim_heatmap(data_dict)
        elif c == 3:
            plot_normalization_impact(data_dict)
        elif c == 4:
            plot_train_test_consistency(data_dict)
        elif c == 5:
            plot_global_distribution(data_dict)
            plot_per_dim_heatmap(data_dict)
            plot_normalization_impact(data_dict)
            plot_train_test_consistency(data_dict)
        else:
            print("  Please enter 0~5.")


if __name__ == "__main__":
    main()
