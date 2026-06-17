import os
import numpy as np
import pandas as pd

# ==========================================
# 全局配置
# ==========================================
VOWEL_DATASET  = 'Vowel_dataset.xlsx'
DATA_DIR       = 'data'

NUM_CLASSES    = 7    # 元音类别数
NUM_FEATURES   = 12   # 每个样本的特征维度
LABEL_DIMS     = 7    # one-hot 标签占用维度
RESERVED_DIMS  = 1    # 第8位预留（固定为0）
TOTAL_DIMS     = LABEL_DIMS + RESERVED_DIMS + NUM_FEATURES   # = 20

TRAIN_RATIO    = 0.8  # 训练集比例 → 206训练 / 52测试
RANDOM_SEED    = 42   # 固定随机种子，保证可复现

# 原始数据集的类别边界（0-indexed，左闭右开）
CLASS_BOUNDARIES = [0, 37, 74, 111, 148, 185, 222, 258]


# ==========================================
# 1. 加载原始数据集
# ==========================================
def load_vowel_dataset():
    """
    读取 Vowel_dataset.xlsx。
    默认结构：第一行为表头，第0列为名称列（忽略），第1~12列为12维LPC特征。
    类别标签根据行范围分配：行0-36→类0，行37-73→类1，……
    """
    if not os.path.exists(VOWEL_DATASET):
        raise FileNotFoundError(
            f"找不到数据集文件 '{VOWEL_DATASET}'，请将其放在程序同目录下。"
        )

    print(f"  正在读取 {VOWEL_DATASET} ...")
    df = pd.read_excel(VOWEL_DATASET, header=0)

    features = df.iloc[:, 1 : NUM_FEATURES + 1].values.astype(np.float64)

    n_samples = features.shape[0]
    labels    = np.zeros(n_samples, dtype=int)
    for cls in range(NUM_CLASSES):
        labels[CLASS_BOUNDARIES[cls] : CLASS_BOUNDARIES[cls + 1]] = cls

    print(f"  总样本数: {n_samples}")
    counts = [int(np.sum(labels == i)) for i in range(NUM_CLASSES)]
    print(f"  各类样本数: {counts}  (合计 {sum(counts)})")
    return features, labels


# ==========================================
# 2. 全局最大值归一化
# ==========================================
def normalize_global_max(features):
    """
    将所有特征值除以全局最大值，缩放到 [0, 1]。
    保留各频率幅值的相对大小，且数值落在扬声器可播放范围内。
    """
    global_max = np.max(features)
    if global_max <= 0:
        raise ValueError("特征全局最大值 ≤ 0，请检查数据集内容。")
    normalized = features / global_max
    print(f"  全局最大值: {global_max:.6f}")
    print(f"  归一化后范围: [{np.min(normalized):.4f}, {np.max(normalized):.4f}]")
    return normalized, global_max


# ==========================================
# 3. 单样本编码（12维特征 → 20维向量）
# ==========================================
def encode_sample(label: int, features_12d: np.ndarray) -> np.ndarray:
    """
    将一个样本编码为20维向量：
      列  0~6  → one-hot 标签（第 label 位 = 1，其余 = 0）
      列  7    → 预留位（固定 = 0）
      列  8~19 → 12维归一化特征
    """
    one_hot = np.zeros(LABEL_DIMS, dtype=np.float32)
    one_hot[label] = 1.0
    return np.concatenate([one_hot, [0.0], features_12d]).astype(np.float32)


# ==========================================
# 4. 生成训练正/负样本
# ==========================================
def generate_train_data(train_features, train_labels):
    """
    对每个训练样本，与其6个错误标签逐一配对，顺序严格对齐：
        pos[i*6 + k]  <->  neg[i*6 + k]
        来自同一原始样本 i，第 k 个错误标签

    train_pos: 206 x 6 = 1236 行（每6行特征相同，标签相同）
    train_neg: 1236 行（每6行特征相同，标签为6个不同错误类别）
    """
    pos_rows = []
    neg_rows = []

    for i in range(len(train_labels)):
        true_label   = int(train_labels[i])
        feat         = train_features[i]
        wrong_labels = [l for l in range(NUM_CLASSES) if l != true_label]

        for wrong_label in wrong_labels:
            pos_rows.append(encode_sample(true_label,  feat))
            neg_rows.append(encode_sample(wrong_label, feat))

    return np.array(pos_rows, dtype=np.float32), np.array(neg_rows, dtype=np.float32)


# ==========================================
# 5. 生成测试嵌入输入与真实标签
# ==========================================
def generate_test_data(test_features, test_labels):
    """
    对每个测试样本嵌入全部7个候选标签：
        rows[i*7 : i*7+7] → 测试样本 i 分别嵌入标签 0,1,...,6

    推理时：取这7行中 goodness 最大的行对应的标签作为预测结果。

    test_embedded_inputs : 52 x 7 = 364 行
    test_true_labels     : 52 行
    """
    embedded_rows = []
    for i in range(len(test_labels)):
        feat = test_features[i]
        for label in range(NUM_CLASSES):
            embedded_rows.append(encode_sample(label, feat))

    return (
        np.array(embedded_rows, dtype=np.float32),
        np.array(test_labels,   dtype=np.int32)
    )


# ==========================================
# 6. 可选：20x20 线性变换（作用于完整编码矩阵）
# ==========================================
def apply_linear_transform(encoded_matrix: np.ndarray):
    """
    对完整的 20 维编码向量乘以随机 20x20 矩阵，
    将标签信息弥散到整个向量，扩大输入空间。

    变换后重新归一化到 [0, 1]，保证幅值可播放。
    变换矩阵保存到 data/linear_transform_matrix.csv。

    参数:
        encoded_matrix : shape (N, 20)
    返回:
        result : shape (N, 20)，归一化后
        W      : 20x20 变换矩阵
    """
    np.random.seed(RANDOM_SEED)
    W = np.random.randn(TOTAL_DIMS, TOTAL_DIMS).astype(np.float64)   # 20x20

    # (N, 20) @ (20, 20) -> (N, 20)
    transformed = encoded_matrix.astype(np.float64) @ W

    t_min  = np.min(transformed)
    t_max  = np.max(transformed)
    result = (transformed - t_min) / (t_max - t_min + 1e-12)

    os.makedirs(DATA_DIR, exist_ok=True)
    W_path = os.path.join(DATA_DIR, 'linear_transform_matrix.csv')
    pd.DataFrame(W).to_csv(W_path, index=False, header=False)
    print(f"  线性变换矩阵 (20x20) 已保存至: {W_path}")

    return result.astype(np.float32), W


# ==========================================
# 7. 保存 CSV
# ==========================================
def save_csv(name: str, data: np.ndarray):
    path = os.path.join(DATA_DIR, f"{name}.csv")
    pd.DataFrame(data).to_csv(path, index=False, header=False)
    cols = data.shape[1] if data.ndim > 1 else 1
    print(f"  [OK] {name}.csv  ->  {data.shape[0]} 行 x {cols} 列")
    return path


# ==========================================
# 主流程
# ==========================================
def main():
    print("=" * 58)
    print("       元音数据编码器  -  Input Generator")
    print("=" * 58)

    # --- 用户选项 ---
    print("\n是否对完整20维编码向量应用随机 20x20 线性变换？")
    print("  作用：将标签信息弥散至整个向量（扩大输入空间）")
    print("  两种模式建议分别进行独立实验对比")
    ans = input("输入选择 (y/n，默认 n): ").strip().lower()
    use_linear = (ans == 'y')

    # --- Step 1: 加载 ---
    print("\n[1/5] 加载原始数据集...")
    features, labels = load_vowel_dataset()

    # --- Step 2: 归一化 ---
    print("\n[2/5] 全局最大值归一化...")
    norm_features, global_max = normalize_global_max(features)

    # --- Step 3: 划分训练/测试集 ---
    print("\n[3/5] 划分训练/测试集...")
    np.random.seed(RANDOM_SEED)
    n_total = len(labels)
    n_train = int(n_total * TRAIN_RATIO)   # 206
    n_test  = n_total - n_train             # 52

    shuffled_idx   = np.random.permutation(n_total)
    train_idx      = shuffled_idx[:n_train]
    test_idx       = shuffled_idx[n_train:]

    train_features = norm_features[train_idx]
    train_labels   = labels[train_idx]
    test_features  = norm_features[test_idx]
    test_labels    = labels[test_idx]

    print(f"  训练样本: {n_train} 个  |  测试样本: {n_test} 个")
    print(f"  训练集各类分布: {[int(np.sum(train_labels==i)) for i in range(NUM_CLASSES)]}")
    print(f"  测试集各类分布: {[int(np.sum(test_labels==i))  for i in range(NUM_CLASSES)]}")

    # --- Step 4: 编码（12维特征 -> 20维向量）---
    print("\n[4/5] 生成20维编码矩阵...")
    train_pos, train_neg     = generate_train_data(train_features, train_labels)
    test_embedded, test_true = generate_test_data(test_features,  test_labels)
    print(f"  train_pos:            {train_pos.shape}")
    print(f"  train_neg:            {train_neg.shape}")
    print(f"  test_embedded_inputs: {test_embedded.shape}")

    # --- Step 5: 可选线性变换（作用于完整20维编码向量）---
    if use_linear:
        print("\n[5/5] 应用随机 20x20 线性变换...")
        # 三个文件联合归一化，保证处于同一幅值空间
        all_encoded        = np.vstack([train_pos, train_neg, test_embedded])
        all_transformed, _ = apply_linear_transform(all_encoded)

        n_pos = len(train_pos)
        n_neg = len(train_neg)
        train_pos     = all_transformed[: n_pos]
        train_neg     = all_transformed[n_pos : n_pos + n_neg]
        test_embedded = all_transformed[n_pos + n_neg :]
        print(f"  变换后幅值范围: [{np.min(all_transformed):.4f}, {np.max(all_transformed):.4f}]")
    else:
        print("\n[5/5] 跳过线性变换（使用原始20维编码）")

    # --- 保存 ---
    print("\n正在保存文件到 ./data/ ...")
    os.makedirs(DATA_DIR, exist_ok=True)
    save_csv('train_pos',            train_pos)
    save_csv('train_neg',            train_neg)
    save_csv('test_embedded_inputs', test_embedded)
    save_csv('test_true_labels',     test_true.reshape(-1, 1))

    # 保存索引映射（供追溯）
    index_map = pd.DataFrame({
        'shuffled_position'   : np.arange(n_total),
        'original_row_in_xlsx': shuffled_idx,
        'split'               : ['train'] * n_train + ['test'] * n_test,
        'true_label'          : labels[shuffled_idx]
    })
    index_map.to_csv(os.path.join(DATA_DIR, 'split_index.csv'), index=False)
    print(f"  [OK] split_index.csv  ->  索引映射（供追溯用）")

    # --- 最终摘要 ---
    print("\n" + "=" * 58)
    print("  全部完成！文件保存于 ./data/ 目录")
    print("=" * 58)
    print(f"  train_pos            : {train_pos.shape[0]:>5} 行 x {train_pos.shape[1]} 列")
    print(f"  train_neg            : {train_neg.shape[0]:>5} 行 x {train_neg.shape[1]} 列")
    print(f"  test_embedded_inputs :  {test_embedded.shape[0]} 行 x {test_embedded.shape[1]} 列")
    print(f"  test_true_labels     :    {test_true.shape[0]} 行 x 1 列")
    print()
    print(f"  编码说明：")
    print(f"    列  0~6  -> one-hot 标签（第i位=1代表第i类元音）")
    print(f"    列  7    -> 预留位（固定=0）")
    print(f"    列  8~19 -> 归一化12维元音特征（幅值在[0,1]之间）")
    print(f"\n  全局归一化基准: 原始最大值 = {global_max:.6f}")
    if use_linear:
        print(f"  本次使用线性变换模式，已生成 linear_transform_matrix.csv")
    else:
        print(f"  本次使用原始编码模式（无线性变换）")
    print()


if __name__ == "__main__":
    main()
