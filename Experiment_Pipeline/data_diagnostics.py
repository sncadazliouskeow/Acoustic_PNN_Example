"""
data_diagnostics.py
====================
Automated diagnostic tool for the acoustic PhyLL experiment pipeline.
Checks data consistency across all pipeline stages and reports issues.

Usage: python data_diagnostics.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DATA_DIR    = 'data'
NUM_CLASSES = 7
N_TRAIN     = 206
N_TEST      = 52
VOWEL_NAMES = ['ae', 'ah', 'aw', 'uw', 'er', 'iy', 'ih']


# ==========================================
# Utility
# ==========================================

def try_load(name):
    path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, header=None).values.astype(np.float32)


def status(ok, msg):
    tag = "[  OK  ]" if ok else "[FAILED]"
    print(f"  {tag}  {msg}")
    return ok


def warn(msg):
    print(f"  [ WARN ]  {msg}")


# ==========================================
# Check 1: File existence
# ==========================================

def check_files():
    print("\n" + "="*64)
    print("  CHECK 1: File Existence")
    print("="*64)

    required = ['train_pos', 'train_neg', 'test_embedded_inputs',
                'test_true_labels', 'split_index']
    optional = ['phys_train_pos', 'phys_train_neg', 'phys_test_embedded_inputs',
                'phys_train_pos_norm', 'phys_train_neg_norm',
                'phys_test_embedded_inputs_norm', 'phys_norm_factor']

    all_ok = True
    for name in required:
        ok = os.path.exists(os.path.join(DATA_DIR, f"{name}.csv"))
        all_ok &= status(ok, f"{name}.csv")

    print()
    for name in optional:
        ext = '.txt' if name == 'phys_norm_factor' else '.csv'
        exists = os.path.exists(os.path.join(DATA_DIR, f"{name}{ext}"))
        tag = "[ INFO ]" if exists else "[ MISS ]"
        print(f"  {tag}  {name}{ext}  (optional)")

    return all_ok


# ==========================================
# Check 2: Shape consistency
# ==========================================

def check_shapes():
    print("\n" + "="*64)
    print("  CHECK 2: Shape Consistency")
    print("="*64)

    specs = {
        'train_pos':            (N_TRAIN * (NUM_CLASSES-1), 20),
        'train_neg':            (N_TRAIN * (NUM_CLASSES-1), 20),
        'test_embedded_inputs': (N_TEST  * NUM_CLASSES,     20),
        'test_true_labels':     (N_TEST,                     1),
    }
    all_ok = True
    for name, (exp_r, exp_c) in specs.items():
        d = try_load(name)
        if d is None:
            status(False, f"{name}: file missing")
            all_ok = False
            continue
        r, c = d.shape[0], (d.shape[1] if d.ndim > 1 else 1)
        ok = (r == exp_r and c == exp_c)
        all_ok &= status(ok, f"{name}: shape=({r},{c})  expect=({exp_r},{exp_c})")

    # Physical files
    for name in ['phys_train_pos', 'phys_train_neg',
                 'phys_train_pos_norm', 'phys_train_neg_norm']:
        d = try_load(name)
        if d is None:
            continue
        exp_r = N_TRAIN * (NUM_CLASSES-1)
        ok = d.shape[0] == exp_r and d.shape[1] == 20
        all_ok &= status(ok, f"{name}: shape={d.shape}  expect=({exp_r},20)")

    for name in ['phys_test_embedded_inputs', 'phys_test_embedded_inputs_norm']:
        d = try_load(name)
        if d is None:
            continue
        exp_r = N_TEST * NUM_CLASSES
        ok = d.shape[0] == exp_r and d.shape[1] == 20
        all_ok &= status(ok, f"{name}: shape={d.shape}  expect=({exp_r},20)")

    return all_ok


# ==========================================
# Check 3: Value range and scale
# ==========================================

def check_scale():
    print("\n" + "="*64)
    print("  CHECK 3: Value Range & Scale Consistency")
    print("="*64)

    all_ok = True
    groups = [
        # (label, files that must share the same scale)
        ("Raw encoded",
         ['train_pos', 'train_neg', 'test_embedded_inputs']),
        ("Physical (raw)",
         ['phys_train_pos', 'phys_train_neg', 'phys_test_embedded_inputs']),
        ("Physical (normalized)",
         ['phys_train_pos_norm', 'phys_train_neg_norm',
          'phys_test_embedded_inputs_norm']),
    ]

    for group_name, names in groups:
        loaded = {n: try_load(n) for n in names if try_load(n) is not None}
        if not loaded:
            print(f"  [ SKIP ]  {group_name}: no files available")
            continue

        print(f"\n  --- {group_name} ---")
        maxvals = {}
        for name, d in loaded.items():
            mn, mx, me = d.min(), d.max(), d.mean()
            maxvals[name] = mx
            print(f"    {name:<42}  min={mn:.5f}  max={mx:.5f}  mean={me:.5f}")

        if len(maxvals) > 1:
            max_of_maxes = max(maxvals.values())
            min_of_maxes = min(maxvals.values())
            ratio = max_of_maxes / (min_of_maxes + 1e-12)
            ok = ratio < 2.0   # 允许2倍以内的偏差
            all_ok &= status(ok,
                f"Scale ratio across files: {ratio:.2f}x  "
                f"(max={max_of_maxes:.5f}, min={min_of_maxes:.5f})"
                + ("" if ok else "  <- CRITICAL: files not on same scale!"))
            if not ok:
                worst = max(maxvals, key=maxvals.get)
                best  = min(maxvals, key=maxvals.get)
                warn(f"  {worst} (max={maxvals[worst]:.5f}) is {ratio:.1f}x larger than "
                     f"{best} (max={maxvals[best]:.5f})")
                warn(f"  -> Run phys_normalizer.py to re-normalize ALL three files together")

    return all_ok


# ==========================================
# Check 4: Within-sample consistency
# ==========================================

def check_within_sample_consistency():
    print("\n" + "="*64)
    print("  CHECK 4: Within-sample Consistency")
    print("="*64)
    print("  Same sample repeated across rows should have near-zero variance.")
    print()

    all_ok = True

    checks = [
        # (file, rows_per_sample, description, ok_threshold)
        ('train_pos',            6, 'Same sample, same label (should be identical)', 1e-5),
        ('train_neg',            6, 'Same sample, diff wrong labels (OK to vary)', None),
        ('test_embedded_inputs', 7, 'Same sample, diff label embed (OK to vary)',   None),
        ('phys_train_pos',       6, 'Physical: same sample, same label (should be identical)', 1e-4),
        ('phys_train_pos_norm',  6, 'Physical norm: same sample, same label', 1e-4),
        ('phys_test_embedded_inputs_norm', 7, 'Physical test: diff label embed (OK to vary)', None),
    ]

    for fname, rps, desc, threshold in checks:
        d = try_load(fname)
        if d is None:
            continue
        n_samples = d.shape[0] // rps
        within_vars = []
        for i in range(n_samples):
            rows = d[i*rps:(i+1)*rps]
            within_vars.append(rows.var(axis=0).mean())
        mean_wv = np.mean(within_vars)

        if threshold is not None:
            ok = mean_wv < threshold
            all_ok &= status(ok,
                f"{fname}: within-sample var={mean_wv:.2e}  threshold={threshold:.0e}"
                + (f"  <- rows are NOT identical!" if not ok else ""))
        else:
            print(f"  [ INFO ]  {fname}: within-sample var={mean_wv:.2e}  ({desc})")

    # Special check: phys_test within-sample var vs total var
    pt = try_load('phys_test_embedded_inputs_norm')
    if pt is None:
        pt = try_load('phys_test_embedded_inputs')
        
    if pt is not None:
        total_var = pt.var(axis=0).mean()
        within_vars = [pt[i*7:(i+1)*7].var(axis=0).mean() for i in range(N_TEST)]
        mean_wv = np.mean(within_vars)
        ratio = mean_wv / (total_var + 1e-12)
        ok = ratio < 0.5   # within-sample should be < 50% of total
        all_ok &= status(ok,
            f"phys_test: within/total variance ratio = {ratio*100:.1f}%"
            + ("  <- CRITICAL: test rows recorded independently per label!"
               if not ok else "  (label embedding effect dominates, expected)"))
        if not ok:
            warn("  Expected behavior: each test sample's 7 rows should have")
            warn("  the SAME physical audio input -> near-identical cavity output")
            warn("  High within-sample variance means each label was recorded separately")
            warn("  -> Review your test data collection procedure in audio_analyzer.py")

    return all_ok


# ==========================================
# Check 5: Label consistency
# ==========================================

def check_labels():
    print("\n" + "="*64)
    print("  CHECK 5: Label Consistency")
    print("="*64)

    all_ok = True

    idx_path = os.path.join(DATA_DIR, 'split_index.csv')
    if not os.path.exists(idx_path):
        status(False, "split_index.csv missing, cannot check labels")
        return False

    idx_df = pd.read_csv(idx_path)
    train_lbl = idx_df[idx_df['split']=='train']['true_label'].values
    test_lbl  = idx_df[idx_df['split']=='test' ]['true_label'].values

    # 训练/测试集数量
    status(len(train_lbl)==N_TRAIN,
           f"Train labels: {len(train_lbl)}  (expect {N_TRAIN})")
    status(len(test_lbl)==N_TEST,
           f"Test labels:  {len(test_lbl)}  (expect {N_TEST})")

    # 类别分布
    print()
    print("  Class distribution:")
    print(f"  {'Class':<6} {'Vowel':<6} {'Train':>7} {'Test':>7}")
    print(f"  {'-'*30}")
    for c in range(NUM_CLASSES):
        nt = (train_lbl==c).sum()
        ne = (test_lbl ==c).sum()
        flag = "  <- low!" if ne < 3 else ""
        print(f"  {c:<6} {VOWEL_NAMES[c]:<6} {nt:>7} {ne:>7}{flag}")

    # test_true_labels 与 split_index 是否一致
    ttl = try_load('test_true_labels')
    if ttl is not None:
        ttl_flat = ttl.flatten().astype(int)
        ok = np.array_equal(ttl_flat, test_lbl)
        all_ok &= status(ok,
            "test_true_labels.csv matches split_index.csv"
            + ("" if ok else "  <- MISMATCH!"))

    # 验证 train_pos 的 one-hot 部分与 split_index 是否一致
    # 仅对原始编码有效（线性变换后 one-hot 已经混合了）
    tp = try_load('train_pos')
    if tp is not None:
        tp_lbl = np.argmax(tp[::6, :7], axis=1)
        ok_onehot = np.array_equal(tp_lbl, train_lbl)
        if ok_onehot:
            status(True, "train_pos one-hot labels match split_index (no linear transform)")
        else:
            warn("train_pos one-hot does NOT match split_index")
            warn("  -> Linear transform was applied (expected), labels read from split_index")

    return all_ok


# ==========================================
# Check 6: Physical data informativeness
# ==========================================

def check_informativeness():
    print("\n" + "="*64)
    print("  CHECK 6: Physical Data Informativeness")
    print("="*64)

    # 优先使用归一化版本
    for fname in ['phys_train_pos_norm', 'phys_train_pos']:
        d = try_load(fname)
        if d is not None:
            break
    if d is None:
        print("  [ SKIP ]  No physical training data available.")
        return True

    idx_path = os.path.join(DATA_DIR, 'split_index.csv')
    if not os.path.exists(idx_path):
        print("  [ SKIP ]  split_index.csv missing.")
        return True

    train_lbl = pd.read_csv(idx_path)
    train_lbl = train_lbl[train_lbl['split']=='train']['true_label'].values
    labels_per_row = np.repeat(train_lbl, 6)
    d_rep = d[:len(labels_per_row)]

    # 类间方差 vs 类内方差（Fisher criterion）
    class_means = np.array([d_rep[labels_per_row==c].mean(axis=0)
                             for c in range(NUM_CLASSES)
                             if (labels_per_row==c).sum()>0])
    overall_mean = d_rep.mean(axis=0)
    between_var  = np.mean((class_means - overall_mean)**2, axis=0)
    within_var   = np.array([d_rep[labels_per_row==c].var(axis=0).mean()
                              for c in range(NUM_CLASSES)
                              if (labels_per_row==c).sum()>0]).mean(axis=0)

    fisher_mean = float(np.mean(between_var / (within_var + 1e-12)))
    fisher_max  = float(np.max( between_var / (within_var + 1e-12)))
    best_dim    = int(np.argmax(between_var / (within_var + 1e-12)))

    print(f"  Fisher criterion (between / within class variance):")
    print(f"    Mean over dims : {fisher_mean:.4f}")
    print(f"    Max  over dims : {fisher_max:.4f}  (dim {best_dim})")
    print(f"    Baseline (random): ~1.0")

    ok_fisher = fisher_max > 1.0
    status(ok_fisher,
           f"At least one dim has Fisher > 1.0"
           + ("" if ok_fisher
              else "  <- POOR: cavity output not separating classes!"))

    # SNR 分析：信号（类间差异）vs 噪声（类内差异）
    signal_power = float(np.mean(between_var))
    noise_power  = float(np.mean(within_var))
    snr_db = 10 * np.log10((signal_power + 1e-12) / (noise_power + 1e-12))
    ok_snr = snr_db > 0
    status(ok_snr,
           f"Class-discriminative SNR: {snr_db:.1f} dB"
           + ("" if ok_snr else "  <- POOR: noise > signal"))

    return ok_fisher and ok_snr


# ==========================================
# Check 7: Normalization factor file
# ==========================================

def check_norm_factor():
    print("\n" + "="*64)
    print("  CHECK 7: Normalization Factor Record")
    print("="*64)

    factor_path = os.path.join(DATA_DIR, 'phys_norm_factor.txt')
    if not os.path.exists(factor_path):
        warn("phys_norm_factor.txt not found (run phys_normalizer.py)")
        return True

    with open(factor_path) as f:
        lines = {l.split('=')[0]: l.strip().split('=')[1]
                 for l in f if '=' in l}

    gmax = float(lines.get('global_max', 0))
    sf   = float(lines.get('scale_factor', 0))
    print(f"  Recorded global_max  : {gmax:.8f}")
    print(f"  Recorded scale_factor: {sf:.6f}  (x{sf:.2f})")

    # 验证：norm 文件的最大值应该 ≈ 1.0
    all_ok = True
    for fname in ['phys_train_pos_norm', 'phys_train_neg_norm',
                  'phys_test_embedded_inputs_norm']:
        d = try_load(fname)
        if d is None:
            continue
        ok = abs(d.max() * (1.0/sf) / gmax - 1.0) < 0.01 or d.max() <= 1.0 + 1e-3
        all_ok &= status(ok,
            f"{fname}: max={d.max():.6f}"
            + ("" if ok else "  <- unexpected max after normalization"))

    return all_ok


# ==========================================
# Summary Plot
# ==========================================

def plot_summary():
    """生成一张综合概览图，横向对比各数据文件的分布。"""
    print("\n  Generating summary plot ...")

    files_to_plot = []
    for name in ['train_pos', 'phys_train_pos', 'phys_train_pos_norm',
                 'train_neg', 'phys_train_neg_norm',
                 'test_embedded_inputs', 'phys_test_embedded_inputs_norm']:
        d = try_load(name)
        if d is not None:
            files_to_plot.append((name, d))

    if not files_to_plot:
        print("  No data files available for plotting.")
        return

    n = len(files_to_plot)
    fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, (name, d) in enumerate(files_to_plot):
        # 上图：每维均值
        means = d.mean(axis=0)
        stds  = d.std(axis=0)
        dims  = np.arange(len(means))
        axes[0, col].bar(dims, means, color='steelblue', alpha=0.7)
        axes[0, col].errorbar(dims, means, yerr=stds, fmt='none',
                               color='navy', capsize=2, linewidth=0.8)
        axes[0, col].set_title(name, fontsize=8)
        axes[0, col].set_xlabel('Dim', fontsize=7)
        axes[0, col].set_ylabel('Mean ± Std', fontsize=7)
        axes[0, col].tick_params(labelsize=6)

        # 下图：值分布直方图
        axes[1, col].hist(d.flatten(), bins=50, color='steelblue',
                          alpha=0.7, edgecolor='none')
        axes[1, col].set_xlabel('Value', fontsize=7)
        axes[1, col].set_ylabel('Count', fontsize=7)
        axes[1, col].set_title(f'Distribution  max={d.max():.4f}', fontsize=8)
        axes[1, col].tick_params(labelsize=6)

    fig.suptitle('Data Pipeline Overview: Per-file Mean and Distribution',
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(DATA_DIR, 'diagnostics_summary.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Summary plot saved to {out}")


# ==========================================
# Main
# ==========================================

def main():
    print("=" * 64)
    print("  Acoustic PhyLL Experiment Data Diagnostics")
    print("=" * 64)

    results = {}
    results['files']       = check_files()
    results['shapes']      = check_shapes()
    results['scale']       = check_scale()
    results['consistency'] = check_within_sample_consistency()
    results['labels']      = check_labels()
    results['informative'] = check_informativeness()
    results['norm_factor'] = check_norm_factor()

    print("\n" + "="*64)
    print("  DIAGNOSTIC SUMMARY")
    print("="*64)
    all_passed = True
    for name, ok in results.items():
        tag = "[  OK  ]" if ok else "[FAILED]"
        print(f"  {tag}  {name}")
        if not ok:
            all_passed = False

    if all_passed:
        print("\n  All checks passed.")
    else:
        print("\n  One or more checks failed. Please fix issues above before training.")

    # 生成概览图
    ans = input("\nGenerate summary overview plot? (y/n, default y): ").strip().lower()
    if ans != 'n':
        plot_summary()


if __name__ == "__main__":
    main()
