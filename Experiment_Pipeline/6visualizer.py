import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.cm as cm

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ==========================================
# Global Config
# ==========================================
DATA_DIR    = 'data'
NUM_CLASSES = 7
VOWEL_NAMES = ['ae', 'ah', 'aw', 'uw', 'er', 'iy', 'ih']

# Color palette for 7 vowel classes
CLASS_COLORS = [
    '#E63946', '#2196F3', '#4CAF50', '#FF9800',
    '#9C27B0', '#00BCD4', '#795548'
]

# All known data files (for user selection)
KNOWN_FILES = [
    'train_pos', 'train_neg',
    'test_embedded_inputs', 'test_true_labels',
    'phys_train_pos',      'phys_train_pos_norm',
    'phys_train_neg',      'phys_train_neg_norm',
    'phys_test_embedded_inputs', 'phys_test_embedded_inputs_norm',
    'split_index',
]


# ==========================================
# 0. Utility
# ==========================================

def list_available_files():
    """扫描 data/ 目录下所有 CSV 文件供用户选择。"""
    files = [f[:-4] for f in os.listdir(DATA_DIR)
             if f.endswith('.csv') and not f.endswith('_matrix.csv')]
    files.sort()
    return files


def load_file(name: str) -> np.ndarray:
    path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        print(f"[Error] File not found: {path}")
        sys.exit(1)
    return pd.read_csv(path, header=None).values.astype(np.float32)


def select_file_interactively(prompt: str = "Select a file") -> tuple[str, np.ndarray]:
    files = list_available_files()
    print(f"\n{prompt}:")
    for i, f in enumerate(files):
        data = pd.read_csv(os.path.join(DATA_DIR, f"{f}.csv"), header=None)
        print(f"  [{i+1:>2}] {f:<40}  ({len(data)} rows x {data.shape[1]} cols)")
    while True:
        try:
            idx = int(input("  Enter number: ").strip()) - 1
            if 0 <= idx < len(files):
                name = files[idx]
                data = load_file(name)
                return name, data
            print(f"  Please enter 1 to {len(files)}.")
        except ValueError:
            print("  Please enter a valid integer.")


def select_row_range(n_rows: int, label: str = "") -> tuple[int, int, np.ndarray | None]:
    """
    选择行范围或自定义行列表（1-based，闭区间）。
    返回 (start_0based, end_0based_exclusive, custom_indices_or_None)
    custom_indices 不为 None 时，调用方应使用它而非连续切片。
    """
    print(f"  '{label}' has {n_rows} rows total.")
    print(f"  [1] Continuous range  (e.g. rows 1 ~ 100)")
    print(f"  [2] Custom row list   (e.g. 1,5,10,23-30,50)")
    while True:
        try:
            mode = int(input("  Select input mode (1/2): ").strip())
            if mode in [1, 2]:
                break
            print("  Please enter 1 or 2.")
        except ValueError:
            print("  Please enter a valid integer.")

    if mode == 1:
        while True:
            try:
                s = int(input(f"  Start row (1 ~ {n_rows}): ").strip())
                e = int(input(f"  End   row ({s} ~ {n_rows}): ").strip())
                if 1 <= s <= e <= n_rows:
                    return s - 1, e, None
                print(f"  Invalid range.")
            except ValueError:
                print("  Please enter valid integers.")
    else:
        # 解析自定义列表，支持 "1,5,10,23-30,50" 格式
        while True:
            raw = input(
                f"  Enter rows (1-based), e.g. '1,5,10,23-30,50': "
            ).strip()
            try:
                indices = set()
                for part in raw.split(','):
                    part = part.strip()
                    if '-' in part:
                        lo, hi = part.split('-')
                        indices.update(range(int(lo), int(hi) + 1))
                    else:
                        indices.add(int(part))
                indices = sorted(indices)
                # 校验范围
                invalid = [i for i in indices if not (1 <= i <= n_rows)]
                if invalid:
                    print(f"  Out-of-range rows: {invalid[:10]}. Max={n_rows}.")
                    continue
                if len(indices) == 0:
                    print("  No valid rows entered.")
                    continue
                # 转为 0-based
                idx_0 = np.array([i - 1 for i in indices], dtype=int)
                print(f"  Selected {len(idx_0)} rows: {indices[:5]}"
                      f"{'...' if len(indices) > 5 else ''}")
                # 返回首尾范围（仅作参考）和精确索引数组
                return idx_0[0], idx_0[-1] + 1, idx_0
            except ValueError:
                print("  Parse error. Use format like '1,5,10,23-30,50'.")


def _get_labels_custom(file_name: str, custom_idx: np.ndarray) -> np.ndarray | None:
    """为自定义行索引数组获取类别标签。"""
    idx_path = os.path.join(DATA_DIR, 'split_index.csv')
    if not os.path.exists(idx_path):
        return None
    idx_df = pd.read_csv(idx_path)
    train_lbl = idx_df[idx_df['split'] == 'train']['true_label'].values
    test_lbl  = idx_df[idx_df['split'] == 'test' ]['true_label'].values

    if 'train_pos' in file_name or 'train_neg' in file_name:
        row_labels = np.repeat(train_lbl, 6)
    elif 'test_embedded' in file_name:
        row_labels = np.repeat(test_lbl, 7)
    elif 'test_true' in file_name:
        row_labels = test_lbl
    else:
        return None

    valid = custom_idx[custom_idx < len(row_labels)]
    if len(valid) < len(custom_idx):
        print(f"  [Warning] {len(custom_idx)-len(valid)} indices out of label range, skipped.")
    return row_labels[valid]


def get_labels_for_rows(file_name: str, row_start_0: int, row_end_0: int) -> np.ndarray | None:
    """
    尝试为指定文件的指定行范围获取类别标签，供聚类图着色。
    优先从 split_index.csv 推断；若无法获取则返回 None。
    """
    # split_index 有完整的训练/测试集标签
    idx_path = os.path.join(DATA_DIR, 'split_index.csv')
    if not os.path.exists(idx_path):
        return None

    idx_df = pd.read_csv(idx_path)
    n = row_end_0 - row_start_0

    if 'train_pos' in file_name:
        # train_pos 每6行对应同一个样本
        train_lbl = idx_df[idx_df['split']=='train']['true_label'].values
        row_labels = np.repeat(train_lbl, 6)
        return row_labels[row_start_0:row_end_0]

    if 'train_neg' in file_name:
        train_lbl = idx_df[idx_df['split']=='train']['true_label'].values
        row_labels = np.repeat(train_lbl, 6)
        return row_labels[row_start_0:row_end_0]

    if 'test_embedded' in file_name:
        test_lbl = idx_df[idx_df['split']=='test']['true_label'].values
        row_labels = np.repeat(test_lbl, 7)
        return row_labels[row_start_0:row_end_0]

    if 'test_true_labels' in file_name:
        test_lbl = idx_df[idx_df['split']=='test']['true_label'].values
        return test_lbl[row_start_0:row_end_0]

    return None


def save_and_show(fig, fname: str):
    path = os.path.join(DATA_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Figure saved to {path}")
    plt.show()


# ==========================================
# Module 1: Clustering Analysis
# ==========================================

def run_clustering():
    print("\n" + "="*62)
    print("  Module 1: Clustering Analysis")
    print("="*62)

    # 选择文件
    file_name, data = select_file_interactively("Select file for clustering")
    n_rows = len(data)

    # 选择行范围
    r0, r1, custom_idx = select_row_range(n_rows, file_name)
    subset = data[custom_idx] if custom_idx is not None else data[r0:r1]
    n = len(subset)
    row_desc = f"custom {n} rows" if custom_idx is not None else f"rows {r0+1}~{r1}"
    print(f"  Selected {n} rows ({row_desc})")

    # 获取标签（用于着色）
    labels = (get_labels_for_rows(file_name, r0, r1)
              if custom_idx is None
              else _get_labels_custom(file_name, custom_idx))
    has_labels = labels is not None and len(labels) == n
    if not has_labels:
        print("  [Note] Could not retrieve class labels for this file/range.")
        print("         Points will be colored uniformly.")

    # 选择降维方法
    print("\n  Select dimensionality reduction method:")
    print("  [1] PCA   (fast, linear, no hyperparameters)")
    print("  [2] t-SNE (slow, nonlinear, best for visualization)")
    print("  [3] LDA   (supervised, requires class labels)")
    print("  [4] All three (side by side)")
    while True:
        try:
            m = int(input("  Enter (1/2/3/4): ").strip())
            if m in [1, 2, 3, 4]:
                break
            print("  Please enter 1, 2, 3, or 4.")
        except ValueError:
            print("  Please enter a valid integer.")

    if m == 3 and not has_labels:
        print("  [Warning] LDA requires class labels; switching to PCA.")
        m = 1

    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(subset)

    # 降维
    methods_to_run = {1: 'PCA', 2: 't-SNE', 3: 'LDA', 4: 'all'}
    method_list = ['PCA', 't-SNE', 'LDA'] if m == 4 else [methods_to_run[m]]
    embeddings  = {}

    for method in method_list:
        print(f"  Running {method} ...")
        if method == 'PCA':
            reducer = PCA(n_components=2, random_state=42)
            emb = reducer.fit_transform(X)
            explained = reducer.explained_variance_ratio_
            embeddings['PCA'] = (emb, f"PCA  (var: {explained[0]*100:.1f}%+{explained[1]*100:.1f}%)")
        elif method == 't-SNE':
            perp = min(30, max(5, n // 10))
            reducer = TSNE(n_components=2, perplexity=perp,
                           random_state=42, max_iter=1000)
            emb = reducer.fit_transform(X)
            embeddings['t-SNE'] = (emb, f"t-SNE  (perplexity={perp})")
        elif method == 'LDA':
            reducer = LDA(n_components=2)
            emb = reducer.fit_transform(X, labels)
            embeddings['LDA'] = (emb, "LDA  (supervised)")

    # 绘图
    n_plots = len(embeddings)
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for ax, (method_key, (emb, subtitle)) in zip(axes, embeddings.items()):
        if has_labels:
            for cls in range(NUM_CLASSES):
                mask = labels == cls
                if mask.sum() == 0:
                    continue
                ax.scatter(emb[mask, 0], emb[mask, 1],
                           c=CLASS_COLORS[cls], label=VOWEL_NAMES[cls],
                           alpha=0.7, s=25, edgecolors='none')
            ax.legend(fontsize=8, markerscale=1.5,
                      loc='best', framealpha=0.7)
        else:
            ax.scatter(emb[:, 0], emb[:, 1],
                       c='steelblue', alpha=0.6, s=25, edgecolors='none')
        ax.set_title(subtitle, fontsize=10)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        f'Clustering Analysis  |  {file_name}  {row_desc}  (n={n})',
        fontsize=12
    )
    plt.tight_layout()
    fname_out = f"cluster_{file_name}_{row_desc.replace(' ','_')}.png"
    save_and_show(fig, fname_out)


# ==========================================
# Module 2: Heatmap Comparison
# ==========================================

def run_heatmap_comparison():
    print("\n" + "="*62)
    print("  Module 2: Heatmap Comparison")
    print("="*62)
    print("  You will select two files and a row range for each.")
    print("  A stacked heatmap will be generated for comparison.\n")

    # 选择两个文件及行范围
    file_configs = []
    for i in [1, 2]:
        print(f"--- File {i} ---")
        fname, fdata = select_file_interactively(f"Select File {i}")
        r0, r1, custom_idx = select_row_range(len(fdata), fname)
        subset   = fdata[custom_idx] if custom_idx is not None else fdata[r0:r1]
        row_desc = f"custom_{len(subset)}rows" if custom_idx is not None else f"r{r0+1}-{r1}"
        file_configs.append((fname, subset, r0, r1, row_desc))
        print()

    # 计算两个文件的共同颜色范围（vmin/vmax 相同，便于对比）
    all_vals = np.concatenate([fc[1].flatten() for fc in file_configs])
    vmin, vmax = float(all_vals.min()), float(all_vals.max())
    print(f"  Shared color scale: [{vmin:.4f}, {vmax:.4f}]")

    # 绘图
    fig = plt.figure(figsize=(max(14, file_configs[0][1].shape[1]*0.4),
                               max(8, sum(fc[1].shape[0]*0.15 for fc in file_configs) + 3)))

    n_subplots = len(file_configs)
    gs = gridspec.GridSpec(n_subplots, 1, hspace=0.5)

    cmap = 'RdYlBu_r'
    norm = Normalize(vmin=vmin, vmax=vmax)

    for plot_idx, (fname, subset, r0, r1, row_desc) in enumerate(file_configs):
        ax = fig.add_subplot(gs[plot_idx])
        n_rows_shown = subset.shape[0]
        n_cols       = subset.shape[1]

        im = ax.imshow(subset, aspect='auto', cmap=cmap,
                       norm=norm, interpolation='nearest')

        tick_step = max(1, n_rows_shown // 20)
        row_ticks = np.arange(0, n_rows_shown, tick_step)
        ax.set_yticks(row_ticks)
        ax.set_yticklabels([str(r0 + t + 1) for t in row_ticks], fontsize=7)

        col_step = max(1, n_cols // 20)
        col_ticks = np.arange(0, n_cols, col_step)
        ax.set_xticks(col_ticks)
        ax.set_xticklabels([str(c) for c in col_ticks], fontsize=7)

        ax.set_ylabel('Row index', fontsize=9)
        ax.set_xlabel('Dimension', fontsize=9)
        ax.set_title(
            f'File {plot_idx+1}: {fname}  |  {row_desc}  '
            f'({n_rows_shown} rows x {n_cols} cols)',
            fontsize=10, pad=6
        )

        cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f'Heatmap Comparison  |  '
        f'{file_configs[0][0]}  vs  {file_configs[1][0]}',
        fontsize=12, y=1.01
    )

    out_fname = (f"heatmap_{file_configs[0][0]}_{file_configs[0][4]}"
                 f"__vs__{file_configs[1][0]}_{file_configs[1][4]}.png")
    save_and_show(fig, out_fname)


# ==========================================
# Module 3: Per-dimension Statistics
# ==========================================

def run_dimension_statistics():
    """
    对选定文件的指定行范围，绘制每个维度的均值±std箱线图，
    并可对比两个文件，直观判断数据在各维度上的分布是否合理。
    """
    print("\n" + "="*62)
    print("  Module 3: Per-dimension Statistics")
    print("="*62)

    compare = input("  Compare two files? (y/n, default n): ").strip().lower() == 'y'
    n_files = 2 if compare else 1

    configs = []
    for i in range(n_files):
        tag = f"File {i+1}" if compare else "File"
        fname, fdata = select_file_interactively(f"Select {tag}")
        r0, r1, custom_idx = select_row_range(len(fdata), fname)
        subset   = fdata[custom_idx] if custom_idx is not None else fdata[r0:r1]
        row_desc = f"custom_{len(subset)}rows" if custom_idx is not None else f"r{r0+1}-{r1}"
        configs.append((fname, subset, r0, r1, row_desc))

    fig, axes = plt.subplots(n_files, 1,
                              figsize=(max(12, configs[0][1].shape[1]*0.45),
                                       4.5*n_files))
    if n_files == 1:
        axes = [axes]

    for ax, (fname, subset, r0, r1, row_desc) in zip(axes, configs):
        means  = subset.mean(axis=0)
        stds   = subset.std(axis=0)
        mins   = subset.min(axis=0)
        maxs   = subset.max(axis=0)
        dims   = np.arange(len(means))

        ax.fill_between(dims, mins, maxs,
                        alpha=0.15, color='steelblue', label='Min-Max range')
        ax.fill_between(dims, means - stds, means + stds,
                        alpha=0.35, color='steelblue', label='Mean ± Std')
        ax.plot(dims, means, color='steelblue', linewidth=1.8,
                marker='o', markersize=3, label='Mean')

        ax.set_xlabel('Dimension Index')
        ax.set_ylabel('Value')
        ax.set_title(f'{fname}  |  {row_desc}  ({len(subset)} rows)', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(dims)
        ax.set_xticklabels([str(d) for d in dims], fontsize=7)

    fig.suptitle('Per-dimension Statistics', fontsize=12)
    plt.tight_layout()
    tag_str = configs[0][0]
    if compare:
        tag_str += f"__vs__{configs[1][0]}"
    save_and_show(fig, f"dim_stats_{tag_str}.png")


# ==========================================
# Module 4: Class Separation Score
# ==========================================

def run_class_separation():
    """
    计算并可视化各类别在每个维度上的均值，
    以及类间距离矩阵，直观量化不同映射对可分度的影响。
    """
    print("\n" + "="*62)
    print("  Module 4: Class Separation Analysis")
    print("="*62)
    print("  Requires a file with retrievable class labels.")

    fname, fdata = select_file_interactively("Select file")
    r0, r1, custom_idx = select_row_range(len(fdata), fname)
    subset   = fdata[custom_idx] if custom_idx is not None else fdata[r0:r1]
    row_desc = f"custom_{len(subset)}rows" if custom_idx is not None else f"r{r0+1}-{r1}"
    labels   = (_get_labels_custom(fname, custom_idx) if custom_idx is not None
                else get_labels_for_rows(fname, r0, r1))

    if labels is None or len(labels) != len(subset):
        print("  [Error] Cannot retrieve labels for this file/range.")
        print("  Supported files: train_pos, train_neg, test_embedded_inputs")
        return

    # 类别均值矩阵  (7, n_dims)
    class_means = np.array([
        subset[labels == c].mean(axis=0) if (labels == c).sum() > 0
        else np.zeros(subset.shape[1])
        for c in range(NUM_CLASSES)
    ])

    # 类间欧氏距离矩阵
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(class_means, class_means, metric='euclidean')

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # 1. 各类别均值折线图
    ax = axes[0]
    for c in range(NUM_CLASSES):
        if (labels == c).sum() == 0:
            continue
        ax.plot(class_means[c], color=CLASS_COLORS[c],
                label=VOWEL_NAMES[c], linewidth=1.5, marker='o', markersize=3)
    ax.set_xlabel('Dimension Index')
    ax.set_ylabel('Mean Value')
    ax.set_title('Per-class Mean per Dimension')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. 类间距离热图
    ax2 = axes[1]
    im = ax2.imshow(dist_matrix, cmap='YlOrRd', aspect='auto')
    ax2.set_xticks(range(NUM_CLASSES))
    ax2.set_yticks(range(NUM_CLASSES))
    ax2.set_xticklabels(VOWEL_NAMES)
    ax2.set_yticklabels(VOWEL_NAMES)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax2.text(j, i, f"{dist_matrix[i,j]:.2f}",
                     ha='center', va='center', fontsize=7,
                     color='white' if dist_matrix[i,j] > dist_matrix.max()*0.6 else 'black')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_title('Inter-class Euclidean Distance Matrix')

    # 3. 每个维度的类内/类间方差比（Fisher criterion per dim）
    ax3 = axes[2]
    overall_mean = subset.mean(axis=0)
    between_var = np.array([
        np.mean([(class_means[c][d] - overall_mean[d])**2
                 for c in range(NUM_CLASSES)
                 if (labels==c).sum()>0])
        for d in range(subset.shape[1])
    ])
    within_var = np.array([
        np.mean([subset[labels==c, d].var()
                 for c in range(NUM_CLASSES)
                 if (labels==c).sum()>0])
        for d in range(subset.shape[1])
    ])
    fisher = between_var / (within_var + 1e-12)
    dims = np.arange(subset.shape[1])
    colors_fisher = cm.RdYlGn(Normalize()(fisher))
    ax3.bar(dims, fisher, color=colors_fisher, edgecolor='none')
    ax3.set_xlabel('Dimension Index')
    ax3.set_ylabel('Fisher Criterion  (between / within variance)')
    ax3.set_title('Per-dimension Discriminability')
    ax3.set_xticks(dims)
    ax3.set_xticklabels([str(d) for d in dims], fontsize=7)
    ax3.grid(True, alpha=0.3, axis='y')
    top3 = np.argsort(fisher)[::-1][:3]
    print(f"  Top 3 most discriminative dimensions: {list(top3)}  "
          f"(Fisher={fisher[top3].round(3)})")

    fig.suptitle(
        f'Class Separation Analysis  |  {fname}  {row_desc}',
        fontsize=12
    )
    plt.tight_layout()
    save_and_show(fig, f"separation_{fname}_{row_desc}.png")


# ==========================================
# Main Menu
# ==========================================

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 62)
    print("   Acoustic PNN Experiment Visualizer")
    print("=" * 62)

    # 检查 data/ 目录
    available = list_available_files()
    if not available:
        print(f"\n[Warning] No CSV files found in '{DATA_DIR}/'.")
        print("  Please run input_generator.py first.")
    else:
        print(f"\n  Available files in '{DATA_DIR}/': {len(available)} CSV files")

    while True:
        print("\n" + "-"*62)
        print("  Select analysis module:")
        print("  [1] Clustering Analysis      (PCA / t-SNE / LDA)")
        print("  [2] Heatmap Comparison       (two files, row ranges)")
        print("  [3] Per-dimension Statistics (mean, std, min-max)")
        print("  [4] Class Separation Analysis (Fisher criterion)")
        print("  [0] Exit")
        print("-"*62)

        try:
            choice = int(input("  Enter module number: ").strip())
        except ValueError:
            print("  Please enter a valid integer.")
            continue

        if choice == 0:
            print("  Exiting visualizer.")
            break
        elif choice == 1:
            run_clustering()
        elif choice == 2:
            run_heatmap_comparison()
        elif choice == 3:
            run_dimension_statistics()
        elif choice == 4:
            run_class_separation()
        else:
            print("  Please enter 0, 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
