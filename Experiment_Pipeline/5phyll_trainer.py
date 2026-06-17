import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ==========================================
# Global Config
# ==========================================
DATA_DIR     = 'data'
NUM_CLASSES  = 7
NUM_FEATURES = 20
N_TRAIN      = 206
N_TEST       = 52
N_PAIRS      = N_TRAIN * (NUM_CLASSES - 1)   # 1236

VOWEL_NAMES  = ['ae', 'ah', 'aw', 'uw', 'er', 'iy', 'ih']

# PhyLL hyperparameters
THETA    = 4.0
N_INNER  = 10
LR       = 0.0005
N_EPOCHS = 200

TORCH_SEED = 2


# ==========================================
# 1. Data Loading
# ==========================================

def load_csv(name: str, required: bool = True):
    path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        if required:
            print(f"[Error] Cannot find {path}")
            print("  Please run the preceding program to generate this file.")
            sys.exit(1)
        return None
    return pd.read_csv(path, header=None).values.astype(np.float32)


def _prefer_norm(name: str) -> str:
    """
    优先使用归一化版本（phys_*_norm.csv），若不存在则回退到原始文件并给出提示。
    归一化文件由 phys_normalizer.py 生成。
    """
    norm_name = f"{name}_norm"
    norm_path = os.path.join(DATA_DIR, f"{norm_name}.csv")
    if os.path.exists(norm_path):
        return norm_name
    print(f"  [Note] {norm_name}.csv not found, using {name}.csv instead.")
    print(f"         Run phys_normalizer.py to generate normalized files.")
    return name


def load_true_labels_from_index():
    """
    从 split_index.csv 中读取训练集和测试集的真实标签。
    这是唯一可靠的标签来源：无论是否应用了线性变换，
    split_index.csv 中存储的都是原始分配的类别标签。

    返回:
        true_labels_train : (206,) int，训练样本的真实标签
        true_labels_test  : (52,)  int，测试样本的真实标签（用于校验）
    """
    idx_path = os.path.join(DATA_DIR, 'split_index.csv')
    if not os.path.exists(idx_path):
        print(f"[Error] Cannot find {idx_path}")
        print("  Please re-run input_generator.py to regenerate split_index.csv.")
        sys.exit(1)

    df = pd.read_csv(idx_path)
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    test_df  = df[df['split'] == 'test'].reset_index(drop=True)

    true_labels_train = train_df['true_label'].values.astype(int)   # (206,)
    true_labels_test  = test_df['true_label'].values.astype(int)    # (52,)

    return true_labels_train, true_labels_test


def build_wrong_labels_mat(true_labels_train: np.ndarray) -> np.ndarray:
    """
    根据真实标签构建错误标签矩阵。
    对每个样本，列出除自身以外的所有类别（升序排列），共6个。

    返回:
        wrong_labels_mat : (206, 6) int
    """
    mat = np.zeros((N_TRAIN, NUM_CLASSES - 1), dtype=int)
    for i, tl in enumerate(true_labels_train):
        mat[i] = [l for l in range(NUM_CLASSES) if l != tl]
    return mat


def select_dataset():
    print("=" * 62)
    print("     PhyLL Physical Neural Network Trainer")
    print("=" * 62)
    print("\nSelect the dataset for this training run:")
    print("  [1] Raw encoded data  (digital baseline, no physical cavity)")
    print("      -> train_pos / train_neg / test_embedded_inputs")
    print("  [2] Physical experiment data  (acoustic cavity output)")
    print("      -> phys_train_pos / phys_train_neg / phys_test_embedded_inputs")
    print("  [3] Physical + skip connection  (concat physical & raw, dim=40)")
    print("      -> matches the original paper architecture")

    while True:
        try:
            choice = int(input("\nEnter choice (1/2/3): ").strip())
            if choice in [1, 2, 3]:
                break
            print("  Please enter 1, 2, or 3.")
        except ValueError:
            print("  Please enter a valid number.")
    return choice


def load_all_data(mode: int, true_labels_train: np.ndarray):
    """
    Load training and test tensors according to mode.

    Returns:
        h_train_pos : (1236, in_dim)
        h_train_neg : (1236, in_dim)
        h_test      : (364,  in_dim)
        test_labels : (52,)  int
        in_dim      : int
    """
    raw_pos  = load_csv('train_pos')
    raw_neg  = load_csv('train_neg')
    raw_test = load_csv('test_embedded_inputs')
    test_labels = load_csv('test_true_labels').flatten().astype(int)

    if mode == 1:
        h_pos, h_neg, h_test = raw_pos, raw_neg, raw_test
        in_dim = NUM_FEATURES
        print("\n  Mode: Digital baseline (raw encoding, dim=20)")

    elif mode == 2:
        h_pos  = load_csv(_prefer_norm('phys_train_pos'))
        h_neg  = load_csv(_prefer_norm('phys_train_neg'))
        h_test = load_csv(_prefer_norm('phys_test_embedded_inputs'))
        in_dim = NUM_FEATURES
        print("\n  Mode: Physical experiment data (acoustic cavity output, dim=20)")

    else:
        phys_pos  = load_csv(_prefer_norm('phys_train_pos'))
        phys_neg  = load_csv(_prefer_norm('phys_train_neg'))
        phys_test = load_csv(_prefer_norm('phys_test_embedded_inputs'))
        h_pos  = np.concatenate([phys_pos,  raw_pos],  axis=1)
        h_neg  = np.concatenate([phys_neg,  raw_neg],  axis=1)
        h_test = np.concatenate([phys_test, raw_test], axis=1)
        in_dim = NUM_FEATURES * 2
        print("\n  Mode: Physical + skip connection (dim=40)")

    print(f"  Training pairs : {len(h_pos)}  ({N_TRAIN} samples x 6 wrong labels)")
    print(f"  Test samples   : {N_TEST}")
    print(f"  Input dim      : {in_dim}")

    return (
        torch.tensor(h_pos,  dtype=torch.float32),
        torch.tensor(h_neg,  dtype=torch.float32),
        torch.tensor(h_test, dtype=torch.float32),
        test_labels,
        in_dim
    )


# ==========================================
# 2. Anti-overfitting Options
# ==========================================

def select_regularization(h_train_pos, h_train_neg,
                           true_labels_train, wrong_labels_mat):
    """
    让用户选择是否启用防过拟合选项：
      A. 无（使用全部数据，全部参数参与训练）
      B. 减少训练数据比例（随机子采样）
      C. 冻结权重矩阵的一部分（梯度掩码）
      D. 同时启用 B + C

    返回:
        h_pos_out, h_neg_out : 处理后的训练张量
        tl_out, wl_out       : 对应的标签（如果子采样则同步缩减）
        freeze_ratio         : float，0.0 表示不冻结
    """
    print("\n[Regularization Options]")
    print("  [A] None  (use full data, all parameters trainable)")
    print("  [B] Subsample training data  (reduce dataset size)")
    print("  [C] Partial weight freeze    (gradient mask on W_t)")
    print("  [D] Both B and C")

    while True:
        opt = input("  Enter option (A/B/C/D, default A): ").strip().upper()
        if opt in ['', 'A', 'B', 'C', 'D']:
            break
        print("  Please enter A, B, C, or D.")
    if opt == '':
        opt = 'A'

    h_pos_out = h_train_pos
    h_neg_out = h_train_neg
    tl_out    = true_labels_train
    wl_out    = wrong_labels_mat
    freeze_ratio = 0.0

    # --- 子采样 ---
    if opt in ['B', 'D']:
        while True:
            try:
                ratio = float(input(
                    "  Subsample ratio (e.g. 0.5 = keep 50% of training samples): "
                ).strip())
                if 0.0 < ratio < 1.0:
                    break
                print("  Please enter a value between 0 and 1 (exclusive).")
            except ValueError:
                print("  Please enter a valid float.")

        n_keep  = max(1, int(N_TRAIN * ratio))
        rng     = np.random.default_rng(TORCH_SEED)
        keep_idx = rng.choice(N_TRAIN, n_keep, replace=False)
        keep_idx.sort()

        # 每个样本对应连续6行
        pair_idx = np.concatenate([np.arange(i*6, i*6+6) for i in keep_idx])
        h_pos_out = h_train_pos[pair_idx]
        h_neg_out = h_train_neg[pair_idx]
        tl_out    = true_labels_train[keep_idx]
        wl_out    = wrong_labels_mat[keep_idx]

        print(f"  Subsampled: {n_keep}/{N_TRAIN} samples "
              f"({n_keep * (NUM_CLASSES-1)} pairs retained)")

    # --- 梯度冻结比例 ---
    if opt in ['C', 'D']:
        while True:
            try:
                freeze_ratio = float(input(
                    "  Freeze ratio for W_t (e.g. 0.5 = freeze 50% of weights): "
                ).strip())
                if 0.0 < freeze_ratio < 1.0:
                    break
                print("  Please enter a value between 0 and 1 (exclusive).")
            except ValueError:
                print("  Please enter a valid float.")
        print(f"  Gradient mask: {freeze_ratio*100:.0f}% of W_t will be frozen")

    return h_pos_out, h_neg_out, tl_out, wl_out, freeze_ratio


# ==========================================
# 3. PhyLL Layer
# ==========================================

class PhyLLLayer(nn.Module):
    """
    Trainable weight matrix W_t updated by the PhyLL algorithm.

    Forward : y = abs(W_t @ h)
    Loss    : L = mean(log(1 + exp(-θ * (cossim(y_pos, ξ) - cossim(y_neg, ξ)))))
    ξ is a fixed random unit vector generated once before training.
    """
    def __init__(self, in_features: int, out_features: int = NUM_FEATURES,
                 freeze_ratio: float = 0.0):
        super().__init__()
        self.out_features = out_features
        self.freeze_ratio = freeze_ratio

        self.W   = nn.Linear(in_features, out_features, bias=False)
        self.opt = Adam(self.W.parameters(), lr=LR, weight_decay=0)

        # Fixed reference vector ξ — same for training and inference
        xi = torch.randn(out_features)
        xi = xi / (xi.norm() + 1e-8)
        self.register_buffer('xi', xi)

        # Gradient mask for partial freeze (1 = trainable, 0 = frozen)
        if freeze_ratio > 0.0:
            mask = torch.ones_like(self.W.weight)
            n_freeze = int(mask.numel() * freeze_ratio)
            flat_idx = torch.randperm(mask.numel())[:n_freeze]
            mask.view(-1)[flat_idx] = 0.0
            self.register_buffer('grad_mask', mask)
            # Apply hook to zero out frozen gradients after each backward
            self.W.weight.register_hook(lambda grad: grad * self.grad_mask)
        else:
            self.grad_mask = None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.abs(self.W(h))
    
    def goodness(self, h: torch.Tensor) -> torch.Tensor:
        y      = self.forward(h)
        xi_exp = self.xi.unsqueeze(0).expand_as(y)
        return F.cosine_similarity(y, xi_exp, dim=1)

    def train_step(self, h_pos: torch.Tensor, h_neg: torch.Tensor) -> float:
        total_loss = 0.0
        for _ in range(N_INNER):
            g_pos = self.goodness(h_pos)
            g_neg = self.goodness(h_neg)
            loss  = torch.log(
                1 + torch.exp(-THETA * (g_pos - g_neg))
            ).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            total_loss += loss.item()
        return total_loss / N_INNER


# ==========================================
# 4. Reference Vector ξ Visualization  [New]
# ==========================================

def visualize_xi(layer: PhyLLLayer, seed: int, mode_name: str):
    """
    打印并可视化参考向量 ξ 的数值分布。
    ξ 的方向决定了网络学习"goodness"的方向，
    对结果影响较大，在此展示以便实验记录。
    """
    xi = layer.xi.cpu().numpy()
    print(f"\n  Reference vector ξ  (seed={seed}, dim={len(xi)}):")
    print(f"  Values: {np.round(xi, 4)}")
    print(f"  Norm  : {np.linalg.norm(xi):.6f}  (should be ~1.0)")
    print(f"  Mean  : {xi.mean():.4f}   Std: {xi.std():.4f}")
    print(f"  Range : [{xi.min():.4f}, {xi.max():.4f}]")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 柱状图
    ax = axes[0]
    colors = ['steelblue' if v >= 0 else 'tomato' for v in xi]
    ax.bar(np.arange(len(xi)), xi, color=colors, edgecolor='white', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Dimension Index')
    ax.set_ylabel('Value')
    ax.set_title(f'Reference Vector ξ  [seed={seed}]')
    ax.grid(True, alpha=0.3, axis='y')

    # 极坐标雷达图（直观感受向量方向）
    ax2 = axes[1]
    angles = np.linspace(0, 2 * np.pi, len(xi), endpoint=False)
    radii  = np.abs(xi)
    signs  = np.sign(xi)
    bar_colors = ['steelblue' if s >= 0 else 'tomato' for s in signs]
    ax2 = plt.subplot(1, 2, 2, projection='polar')
    ax2.bar(angles, radii, width=2*np.pi/len(xi)*0.85,
            color=bar_colors, alpha=0.75, edgecolor='white')
    ax2.set_title('ξ Polar View\n(blue=positive, red=negative)',
                  pad=15, fontsize=9)
    ax2.set_xticks(angles)
    ax2.set_xticklabels([str(i) for i in range(len(xi))], fontsize=7)

    plt.suptitle(f'Reference Vector ξ  [{mode_name}]', fontsize=12)
    plt.tight_layout()
    fname = os.path.join(DATA_DIR, f'xi_vector_{mode_name}.png')
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"  ξ visualization saved to {fname}")


# ==========================================
# 5. Accuracy Evaluation
# ==========================================

def evaluate_train_accuracy(layer: PhyLLLayer,
                             h_pos: torch.Tensor,
                             h_neg: torch.Tensor,
                             true_labels_train: np.ndarray,
                             wrong_labels_mat: np.ndarray):
    """
    Training set accuracy evaluation.

    For each sample i:
      - pos input  = h_pos[i*6]          (correct label embedding)
      - neg inputs = h_neg[i*6 : i*6+6]  (6 wrong label embeddings)
    Stack to (7,) goodness vector; argmax=0 means correct label wins.

    True labels are always read from split_index.csv (via true_labels_train),
    which is valid regardless of whether a linear transform was applied.
    """
    n_samples = len(true_labels_train)
    layer.eval()
    with torch.no_grad():
        pos_per_sample = h_pos.reshape(n_samples, 6, -1)[:, 0, :]   # (N, dim)
        neg_per_sample = h_neg.reshape(n_samples, 6, -1)             # (N, 6, dim)

        all_7      = torch.cat([pos_per_sample.unsqueeze(1), neg_per_sample], dim=1)
        all_7_flat = all_7.reshape(-1, all_7.shape[-1])
        g_flat     = layer.goodness(all_7_flat)
        g_grouped  = g_flat.reshape(n_samples, 7)
        argmax_idx = g_grouped.argmax(dim=1).cpu().numpy()

    preds = np.where(
        argmax_idx == 0,
        true_labels_train,
        wrong_labels_mat[np.arange(n_samples), argmax_idx - 1]
    )
    accuracy = float(np.mean(preds == true_labels_train))
    return accuracy, preds, true_labels_train


def evaluate_test_accuracy(layer: PhyLLLayer,
                            h_test: torch.Tensor,
                            test_labels: np.ndarray):
    """
    Test set accuracy evaluation.

    test_embedded rows are ordered as: sample_0_label_0, sample_0_label_1, ...,
    sample_0_label_6, sample_1_label_0, ...
    So argmax position in each group of 7 directly equals the predicted label.
    test_labels (loaded from test_true_labels.csv) provides ground truth.
    """
    layer.eval()
    with torch.no_grad():
        g_all     = layer.goodness(h_test)
        g_grouped = g_all.reshape(N_TEST, NUM_CLASSES)
        preds     = g_grouped.argmax(dim=1).cpu().numpy()
    accuracy = float(np.mean(preds == test_labels))
    return accuracy, preds


def save_goodness_to_csv(layer: PhyLLLayer, h_test: torch.Tensor, test_labels: np.ndarray, filename='data/test_goodness_scores.csv'):
    """
    将测试集的 Goodness 打分矩阵导出为 CSV 文件。
    """
    layer.eval()
    with torch.no_grad():
        # 计算所有 364 行的 goodness
        g_all = layer.goodness(h_test)
        # 重组为 (52个样本, 7个候选标签) 的打分矩阵
        g_matrix = g_all.reshape(N_TEST, NUM_CLASSES).cpu().numpy()

    # 转换为 DataFrame 并添加表头
    columns = [f"Label_{i}_Goodness" for i in range(NUM_CLASSES)]
    df = pd.DataFrame(g_matrix, columns=columns)
    
    # 附加上真实标签和预测结果，方便对比
    df['True_Label'] = test_labels
    df['Predicted_Label'] = g_matrix.argmax(axis=1)
    
    # 判断是否预测正确 (1为正确，0为错误)
    df['Is_Correct'] = (df['True_Label'] == df['Predicted_Label']).astype(int)

    # 导出文件
    df.to_csv(filename, index=False)
    print(f"\n  测试集 Goodness 矩阵已保存至 {filename}")


# ==========================================
# 6. Goodness Distribution Plot  [New]
# ==========================================

def plot_goodness_distribution(layer: PhyLLLayer,
                                h_pos: torch.Tensor,
                                h_neg: torch.Tensor,
                                epoch: int,
                                mode_name: str,
                                save: bool = True):
    """
    绘制正/负样本的 goodness 分布直方图。
    理想情况下：正样本 goodness 高，负样本 goodness 低，两峰分离。
    """
    layer.eval()
    with torch.no_grad():
        g_pos = layer.goodness(h_pos).cpu().numpy()
        g_neg = layer.goodness(h_neg).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # 直方图叠加
    ax = axes[0]
    bins = np.linspace(
        min(g_pos.min(), g_neg.min()) - 0.05,
        max(g_pos.max(), g_neg.max()) + 0.05,
        50
    )
    ax.hist(g_pos, bins=bins, alpha=0.6, color='steelblue', label='Positive')
    ax.hist(g_neg, bins=bins, alpha=0.6, color='tomato',    label='Negative')
    ax.axvline(g_pos.mean(), color='steelblue', linestyle='--', linewidth=1.5,
               label=f'Pos mean={g_pos.mean():.3f}')
    ax.axvline(g_neg.mean(), color='tomato',    linestyle='--', linewidth=1.5,
               label=f'Neg mean={g_neg.mean():.3f}')
    ax.set_xlabel('Goodness (cosine similarity with ξ)')
    ax.set_ylabel('Count')
    ax.set_title(f'Goodness Distribution  [epoch={epoch}]')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 差值分布（g_pos - g_neg for aligned pairs）
    ax2 = axes[1]
    delta = g_pos - g_neg   # 每对正负样本的 goodness 差
    ax2.hist(delta, bins=40, color='mediumpurple', alpha=0.8, edgecolor='white')
    ax2.axvline(0, color='black', linewidth=1.2, linestyle='--', label='Δ=0')
    ax2.axvline(delta.mean(), color='mediumpurple', linewidth=1.5,
                label=f'mean Δ={delta.mean():.3f}')
    frac_pos = (delta > 0).mean()
    ax2.set_xlabel('Goodness Difference  (Pos - Neg)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Goodness Gap Distribution  '
                  f'[{frac_pos*100:.1f}% pairs: pos > neg]')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Goodness Analysis  [{mode_name}]', fontsize=12)
    plt.tight_layout()

    if save:
        fname = os.path.join(DATA_DIR, f'goodness_dist_{mode_name}_ep{epoch}.png')
        plt.savefig(fname, dpi=150)
        print(f"  Goodness distribution saved to {fname}")
    plt.show()

    # 打印统计摘要
    print(f"\n  Goodness summary (epoch {epoch}):")
    print(f"    Positive: mean={g_pos.mean():.4f}  std={g_pos.std():.4f}  "
          f"range=[{g_pos.min():.3f}, {g_pos.max():.3f}]")
    print(f"    Negative: mean={g_neg.mean():.4f}  std={g_neg.std():.4f}  "
          f"range=[{g_neg.min():.3f}, {g_neg.max():.3f}]")
    print(f"    Gap (pos-neg): mean={delta.mean():.4f}  "
          f"{frac_pos*100:.1f}% pairs correctly ordered")


# ==========================================
# 7. Visualization Helpers
# ==========================================

def plot_accuracy_curves(train_accs, test_accs, mode_name):
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = np.arange(1, len(train_accs) + 1)
    ax.plot(epochs, [a*100 for a in train_accs],
            label='Train Accuracy', color='steelblue', linewidth=1.5)
    ax.plot(epochs, [a*100 for a in test_accs],
            label='Test Accuracy',  color='tomato',    linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'PhyLL Training Curve  [{mode_name}]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    ax.annotate(f"{train_accs[-1]*100:.2f}%",
                xy=(len(train_accs), train_accs[-1]*100),
                xytext=(-35, 8), textcoords='offset points',
                color='steelblue', fontsize=9)
    ax.annotate(f"{test_accs[-1]*100:.2f}%",
                xy=(len(test_accs), test_accs[-1]*100),
                xytext=(-35, -14), textcoords='offset points',
                color='tomato', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, f'accuracy_curve_{mode_name}.png'), dpi=150)
    plt.show()
    print(f"  Accuracy curve saved to data/accuracy_curve_{mode_name}.png")


def plot_confusion_matrix(trues, preds, title, mode_name, split):
    cm      = confusion_matrix(trues, preds, labels=list(range(NUM_CLASSES)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, subtitle in zip(
        axes, [cm, cm_norm], ['d', '.2f'], ['Count', 'Normalized']
    ):
        disp = ConfusionMatrixDisplay(data, display_labels=VOWEL_NAMES)
        disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format=fmt)
        ax.set_title(subtitle)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    fig.suptitle(f'{title}  [{mode_name}]', fontsize=13)
    plt.tight_layout()
    fname = os.path.join(DATA_DIR, f'confusion_{split}_{mode_name}.png')
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"  Confusion matrix saved to {fname}")


def get_mode_name(mode):
    return {1: 'digital_baseline', 2: 'physical', 3: 'physical_skip'}[mode]


# ==========================================
# 8. Main
# ==========================================

def main():
    # --- Step 0: 选择数据集 ---
    mode      = select_dataset()
    mode_name = get_mode_name(mode)

    # --- Step 1: 加载标签（始终从 split_index.csv，与线性变换无关）---
    print("\n[1/5] Loading labels from split_index.csv ...")
    true_labels_train, true_labels_test = load_true_labels_from_index()
    wrong_labels_mat = build_wrong_labels_mat(true_labels_train)
    print(f"  Train labels loaded: {len(true_labels_train)} samples")
    print(f"  Class distribution : "
          f"{[int((true_labels_train==i).sum()) for i in range(NUM_CLASSES)]}")

    # --- Step 2: 加载特征数据 ---
    print("\n[2/5] Loading feature data ...")
    (h_train_pos, h_train_neg, h_test,
     test_labels, in_dim) = load_all_data(mode, true_labels_train)

    # --- Step 3: 防过拟合选项 ---
    print("\n[3/5] Regularization setup ...")
    (h_train_pos, h_train_neg,
     true_labels_train, wrong_labels_mat,
     freeze_ratio) = select_regularization(
        h_train_pos, h_train_neg,
        true_labels_train, wrong_labels_mat
    )

    # --- Step 4: 初始化模型 ---
    print("\n[4/5] Initializing PhyLL layer ...")
    torch.manual_seed(TORCH_SEED)
    layer = PhyLLLayer(
        in_features  = in_dim,
        out_features = NUM_FEATURES,
        freeze_ratio = freeze_ratio
    )

    total_params  = in_dim * NUM_FEATURES
    frozen_params = int(total_params * freeze_ratio)
    print(f"  W_t shape     : {in_dim} -> {NUM_FEATURES}")
    print(f"  Total params  : {total_params}")
    print(f"  Frozen params : {frozen_params}  "
          f"({freeze_ratio*100:.0f}% of W_t)")
    print(f"\n  Hyperparameters:")
    print(f"    theta (scale)   = {THETA}")
    print(f"    inner steps     = {N_INNER}")
    print(f"    learning rate   = {LR}")
    print(f"    epochs          = {N_EPOCHS}")

    # ξ 可视化  [New]
    print(f"\n  Visualizing reference vector ξ ...")
    visualize_xi(layer, TORCH_SEED, mode_name)

    # --- Step 5: 训练 ---
    print(f"\n[5/5] Training ({N_EPOCHS} epochs) ...")
    print(f"  Press Ctrl+C to stop early and view results\n")
    print(f"  {'Epoch':>6}  {'Loss':>8}  {'Train%':>8}  {'Test%':>8}")
    print(f"  {'-'*38}")

    train_accs, test_accs = [], []

    try:
        for epoch in range(1, N_EPOCHS + 1):
            layer.train()
            loss = layer.train_step(h_train_pos, h_train_neg)

            tr_acc, _, _ = evaluate_train_accuracy(
                layer, h_train_pos, h_train_neg,
                true_labels_train, wrong_labels_mat
            )
            te_acc, _ = evaluate_test_accuracy(layer, h_test, test_labels)

            train_accs.append(tr_acc)
            test_accs.append(te_acc)
            print(f"  {epoch:>6}  {loss:>8.4f}  {tr_acc*100:>7.2f}%  {te_acc*100:>7.2f}%")

    except KeyboardInterrupt:
        print(f"\n  [Early stop] Completed {len(train_accs)} epochs.")

    if not train_accs:
        print("No epochs completed. Exiting.")
        return

    # --- 最终结果 ---
    best_ep = int(np.argmax(test_accs)) + 1
    print(f"\n{'='*62}")
    print(f"  Results  [{mode_name}]")
    print(f"{'='*62}")
    print(f"  Final train accuracy : {train_accs[-1]*100:.2f}%")
    print(f"  Final test  accuracy : {test_accs[-1]*100:.2f}%")
    print(f"  Best  test  accuracy : {max(test_accs)*100:.2f}%  (epoch {best_ep})")

    # --- 可视化 ---
    print("\nGenerating accuracy curve ...")
    plot_accuracy_curves(train_accs, test_accs, mode_name)

    # Goodness distribution  [New]
    print("\nGenerating goodness distribution ...")
    plot_goodness_distribution(layer, h_train_pos, h_train_neg,
                                epoch=len(train_accs),
                                mode_name=mode_name)

    # Confusion matrices
    print("\nGenerating confusion matrices ...")
    _, test_preds = evaluate_test_accuracy(layer, h_test, test_labels)
    plot_confusion_matrix(test_labels, test_preds,
                          'Test Set Confusion Matrix', mode_name, 'test')

    _, train_preds, train_trues = evaluate_train_accuracy(
        layer, h_train_pos, h_train_neg,
        true_labels_train, wrong_labels_mat
    )
    plot_confusion_matrix(train_trues, train_preds,
                          'Train Set Confusion Matrix', mode_name, 'train')

    # --- 分类报告 ---
    print(f"\n{'='*62}")
    print(f"  Per-class Report  (test set, {mode_name})")
    print(f"{'='*62}")
    print(f"  {'Vowel':<6}  {'Correct':>7}  {'Total':>7}  {'Accuracy':>9}")
    print(f"  {'-'*36}")
    for cls in range(NUM_CLASSES):
        mask   = test_labels == cls
        n_tot  = mask.sum()
        n_corr = (test_preds[mask] == cls).sum()
        pct    = n_corr / n_tot * 100 if n_tot > 0 else 0.0
        print(f"  {VOWEL_NAMES[cls]:<6}  {n_corr:>7}  {n_tot:>7}  {pct:>8.1f}%")
    total_corr = (test_preds == test_labels).sum()
    print(f"  {'Total':<6}  {total_corr:>7}  {len(test_labels):>7}  "
          f"{test_accs[-1]*100:>8.2f}%")

    # --- 保存模型 ---
    wt_path = os.path.join(DATA_DIR, f'Wt_{mode_name}.pt')
    torch.save(layer.state_dict(), wt_path)
    print(f"\n  Model weights saved to {wt_path}")
    print()

    save_goodness_to_csv(layer, h_test, test_labels)


if __name__ == "__main__":
    main()
