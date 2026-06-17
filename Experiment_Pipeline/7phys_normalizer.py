import os
import sys
import numpy as np
import pandas as pd

# ==========================================
# Global Config
# ==========================================
DATA_DIR = 'data'

# 必须联合归一化的物理实验数据文件
PHYS_FILES = [
    'phys_train_pos',
    'phys_train_neg',
    'phys_test_embedded_inputs',
]

NORM_SUFFIX = '_norm'
PERCENTILE_VAL = 94  # 剔除最高的全局离群极大值


def main():
    print("=" * 62)
    print("  Physical Data Normalizer (Global Robust Scaling)")
    print("=" * 62)
    
    # --- Step 1: 读取数据 ---
    data_dict = {}
    missing   = []
    for name in PHYS_FILES:
        path = os.path.join(DATA_DIR, f"{name}.csv")
        if not os.path.exists(path):
            missing.append(path)
        else:
            df = pd.read_csv(path, header=None)
            data_dict[name] = df.values.astype(np.float32)
            
    if missing:
        print("[Error] 缺少文件，请先运行 audio_analyzer.py")
        sys.exit(1)

    # --- Step 2: 计算分位数 ---
    all_values = np.concatenate([d.flatten() for d in data_dict.values()])
    
    global_max = float(np.max(all_values))
    global_p_val = float(np.percentile(all_values, PERCENTILE_VAL))
    
    # 保护性处理，防止全零数据
    global_p_val = max(global_p_val, 1e-5)
    scale_factor = 1.0 / global_p_val

    print(f"\n[Statistics]")
    print(f"  Total values evaluated : {len(all_values)}")
    print(f"  Global absolute MAX    : {global_max:.6f} (This outlier was ruining the network)")
    print(f"  Global {PERCENTILE_VAL}% Percentile : {global_p_val:.6f} (This is our new robust ceiling)")
    print(f"  Global Scale Factor    : 1 / {global_p_val:.6f} = {scale_factor:.4f}")

    # --- Step 3: 执行缩放并截断 ---
    print(f"\n[Normalizing & Clipping]")
    for name, data in data_dict.items():
        # 所有维度乘以同一个缩放因子，完美保留相对物理结构
        scaled_data = data * scale_factor
        
        # 超过 1.0 的部分（即那 0.5% 的极值噪声）被削平
        normed = np.clip(scaled_data, 0.0, 1.0)

        out_name = f"{name}{NORM_SUFFIX}"
        out_path = os.path.join(DATA_DIR, f"{out_name}.csv")
        pd.DataFrame(normed).to_csv(out_path, index=False, header=False)
        
        print(f"  Saved  {out_name}.csv  -> max={normed.max():.4f}, mean={normed.mean():.4f}")

    # 保存记录
    factor_path = os.path.join(DATA_DIR, 'phys_norm_factor.txt')
    with open(factor_path, 'w') as f:
        f.write(f"method=global_robust_scaling (P{PERCENTILE_VAL})\n")
        f.write(f"global_p_val={global_p_val}\n")
        f.write(f"scale_factor={scale_factor}\n")
    print(f"\n  Scale factor recorded in {factor_path}")
    
    
if __name__ == "__main__":
    main()