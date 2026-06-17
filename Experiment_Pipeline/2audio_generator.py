import os
import numpy as np
import pandas as pd
import soundfile as sf

# ==========================================
# 全局配置
# ==========================================
DATA_DIR       = 'data'
AUDIO_DIR      = 'audio'

BASE_FREQUENCIES = [300, 450, 600, 750, 900]   # Hz，5个基频
NUM_CHANNELS     = 4                             # 扬声器数量
NUM_FREQ         = len(BASE_FREQUENCIES)         # 5
TOTAL_DIMS       = NUM_CHANNELS * NUM_FREQ       # 20

SAMPLE_RATE      = 48000    # Hz，标准音频采样率
AUDIO_DURATION   = 0.5      # 秒，每段音频时长
SILENCE_DURATION = 0.5      # 秒，段间静音时长（播放程序负责插入，此处仅生成纯音频）
FADE_DURATION    = 0.025    # 秒，淡入淡出时长（消除咔哒声）

VALID_FILES = ['train_pos', 'train_neg', 'test_embedded_inputs']


# ==========================================
# 1. 单声道组合音合成
# ==========================================
def synthesize_channel(amplitudes_5: np.ndarray, sample_rate: int, duration: float) -> np.ndarray:
    """
    将5个基频的幅值合成为一段单声道音频。

    参数:
        amplitudes_5: shape (5,)，对应 BASE_FREQUENCIES 各频率的幅值，范围 [0, 1]
        sample_rate : 采样率 (Hz)
        duration    : 时长 (秒)

    返回:
        audio: shape (N,)，float32，范围约 [-1, 1]
    """
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    signal = np.zeros(n_samples, dtype=np.float64)
    for amp, freq in zip(amplitudes_5, BASE_FREQUENCIES):
        signal += amp * np.sin(2 * np.pi * freq * t)

    # 淡入淡出：消除起止时刻的不连续咔哒声
    fade_samples = int(sample_rate * FADE_DURATION)
    if fade_samples > 0:
        # 抛弃线性的 linspace，使用正弦平滑曲线
        t_fade = np.linspace(0, np.pi / 2, fade_samples)
        fade_in  = np.sin(t_fade) ** 2
        fade_out = np.cos(t_fade) ** 2
        
        signal[:fade_samples]  *= fade_in
        signal[-fade_samples:] *= fade_out

    # 归一化到 [-1, 1]（防止多频叠加超出范围）
    max_val = np.max(np.abs(signal))
    if max_val > 1e-8:
        signal /= max_val

    return signal.astype(np.float32)


# ==========================================
# 2. 单行数据 → 4路单声道 + 1路4声道
# ==========================================
def generate_audio_for_row(
    row_vector: np.ndarray,
    row_number: int,         # 1-based，用于文件命名
    out_dir: str,
    sample_rate: int = SAMPLE_RATE,
    duration: float = AUDIO_DURATION
):
    """
    将一行20维向量拆分为4组，每组合成单声道音频，
    再合并为4声道音频。

    文件命名：
        ch1_row_{row_number:04d}.wav   ~ ch4_row_{row_number:04d}.wav
        combined_row_{row_number:04d}.wav

    参数:
        row_vector : shape (20,)，已归一化到 [0,1]
        row_number : 对应原始数据文件中的行号（1-based）
        out_dir    : 保存目录
    """
    if len(row_vector) != TOTAL_DIMS:
        raise ValueError(f"行向量维度应为 {TOTAL_DIMS}，实际为 {len(row_vector)}")

    channels = []
    for ch in range(NUM_CHANNELS):
        amps = row_vector[ch * NUM_FREQ : (ch + 1) * NUM_FREQ]   # 取5个幅值
        audio_ch = synthesize_channel(amps, sample_rate, duration)
        channels.append(audio_ch)

        # 保存单声道文件
        ch_path = os.path.join(out_dir, f"ch{ch+1}_row_{row_number:04d}.wav")
        sf.write(ch_path, audio_ch, sample_rate, subtype='PCM_16')

    # 合并为4声道（每列=一个声道）
    combined = np.stack(channels, axis=1)   # shape (N_samples, 4)
    combined_path = os.path.join(out_dir, f"combined_row_{row_number:04d}.wav")
    sf.write(combined_path, combined, sample_rate, subtype='PCM_16')


# ==========================================
# 3. 用户交互
# ==========================================
def get_user_inputs():
    print("=" * 58)
    print("        音频生成器  —  Audio Generator")
    print("=" * 58)

    # 选择目标文件
    print("\n请选择要生成音频的数据文件：")
    for i, name in enumerate(VALID_FILES):
        print(f"  [{i+1}] {name}.csv")
    while True:
        try:
            choice = int(input("输入编号 (1/2/3): ").strip())
            if 1 <= choice <= len(VALID_FILES):
                target_name = VALID_FILES[choice - 1]
                break
            print(f"  请输入 1 到 {len(VALID_FILES)} 之间的数字。")
        except ValueError:
            print("  请输入有效数字。")

    # 读取对应 CSV，获取总行数
    csv_path = os.path.join(DATA_DIR, f"{target_name}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"找不到 {csv_path}，请先运行 input_generator.py 生成数据文件。"
        )
    df = pd.read_csv(csv_path, header=None)
    total_rows = len(df)
    print(f"\n  文件共 {total_rows} 行数据。")

    # 选择行范围（1-based，闭区间）
    while True:
        try:
            start = int(input(f"请输入起始行号 (1 ~ {total_rows}): ").strip())
            end   = int(input(f"请输入结束行号 ({start} ~ {total_rows}): ").strip())
            if 1 <= start <= end <= total_rows:
                break
            print(f"  行号无效，请确保 1 ≤ 起始行 ≤ 结束行 ≤ {total_rows}。")
        except ValueError:
            print("  请输入有效整数。")

    return target_name, df, start, end


# ==========================================
# 主流程
# ==========================================
def main():
    target_name, df, start_row, end_row = get_user_inputs()

    # 构建输出目录
    out_dir = os.path.join(AUDIO_DIR, target_name)
    os.makedirs(out_dir, exist_ok=True)

    n_rows = end_row - start_row + 1
    print(f"\n[开始生成]  {target_name}  行 {start_row} → {end_row}  共 {n_rows} 条")
    print(f"输出目录: {out_dir}")
    print(f"音频参数: {NUM_CHANNELS} 声道 | "
          f"{BASE_FREQUENCIES} Hz | "
          f"{AUDIO_DURATION}s / 段 | "
          f"{SAMPLE_RATE} Hz 采样率\n")

    # 核心循环：注意 1-based 到 0-based 的索引转换
    for row_1based in range(start_row, end_row + 1):
        row_0based  = row_1based - 1                      # DataFrame 索引（0-based）
        row_vector  = df.iloc[row_0based].values.astype(np.float32)

        generate_audio_for_row(
            row_vector  = row_vector,
            row_number  = row_1based,                     # 文件名保留 1-based 行号
            out_dir     = out_dir,
            sample_rate = SAMPLE_RATE,
            duration    = AUDIO_DURATION
        )

        # 进度显示
        done = row_1based - start_row + 1
        bar  = '█' * (done * 30 // n_rows) + '░' * (30 - done * 30 // n_rows)
        print(f"\r  [{bar}] {done}/{n_rows}  行 {row_1based}", end='', flush=True)

    print(f"\n\n✅ 完成！共生成 {n_rows * (NUM_CHANNELS + 1)} 个音频文件：")
    print(f"   • {n_rows * NUM_CHANNELS} 个单声道文件  (ch1~ch4_row_XXXX.wav)")
    print(f"   • {n_rows} 个四声道合并文件  (combined_row_XXXX.wav)")
    print(f"\n文件结构示例（行 {start_row}）：")
    for ch in range(1, NUM_CHANNELS + 1):
        print(f"   {out_dir}/ch{ch}_row_{start_row:04d}.wav")
    print(f"   {out_dir}/combined_row_{start_row:04d}.wav")
    print()


if __name__ == "__main__":
    main()
