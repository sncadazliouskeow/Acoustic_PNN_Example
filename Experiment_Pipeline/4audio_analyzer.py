import os
import sys
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import get_window
import matplotlib.pyplot as plt

# ==========================================
# Global Configuration
# ==========================================
DATA_DIR         = 'data'
BASE_FREQUENCIES = [331, 487, 677, 863, 947]     # Hz
NUM_CHANNELS     = 4                               # 轨道数量
NUM_FREQ         = len(BASE_FREQUENCIES)           # 5
TOTAL_DIMS       = NUM_CHANNELS * NUM_FREQ         # 20
FREQ_TOLERANCE   = 20                              # Hz，在目标频率 ±20Hz 范围内寻找真实峰值

VALID_FILES      = ['train_pos', 'train_neg', 'test_embedded_inputs']

# 切割物理参数
AUDIO_DURATION   = 0.5   # 秒，每段音频时长
SILENCE_DURATION = 0.5   # 秒，段间静音


# ==========================================
# 1. Audio Segmentation (Steady-State Detection)
# ==========================================
def _plot_steady_segmentation(envelope, triggers, segments, threshold, expected, sr, zoom_indices=None):
    """
    Plot the segmentation preview for steady-state extraction.
    Generates two subplots: a Full View and a Zoomed View for detail inspection.
    
    Args:
        zoom_indices: list of ints (0-based). Specify which pulses to zoom in on.
                      If None, defaults to the first 3 pulses.
    """
    time_axis = np.arange(len(envelope)) / sr
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # ==========================================
    # 1. Top Plot: Full View (全局图)
    # ==========================================
    ax1.plot(time_axis, envelope, color='steelblue', linewidth=0.8, label='Energy Envelope')
    ax1.axhline(threshold, color='green', linestyle='--', linewidth=1.2, label=f'Trigger Threshold: {threshold:.5f}')
    
    for i, trig in enumerate(triggers):
        ax1.axvline(trig / sr, color='red', linestyle=':', label='Detected Start' if i==0 else "")
        
    for i, (s, e) in enumerate(segments):
        ax1.axvspan(s / sr, e / sr, color='orange', alpha=0.3, label='Steady-State Window' if i==0 else "")
        
    status = 'OK' if len(segments) == expected else 'MISMATCH'
    ax1.set_title(f'Steady-State Segmentation [FULL VIEW]  Status: {status} | Found={len(segments)} / Expected={expected}')
    ax1.set_ylabel('Energy')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.25)

    # ==========================================
    # 2. Bottom Plot: Zoomed View (局部放大图)
    # ==========================================
    ax2.plot(time_axis, envelope, color='steelblue', linewidth=1.5)
    ax2.axhline(threshold, color='green', linestyle='--', linewidth=1.5)
    
    for trig in triggers:
        ax2.axvline(trig / sr, color='red', linestyle=':', linewidth=2.0)
        
    for s, e in segments:
        ax2.axvspan(s / sr, e / sr, color='orange', alpha=0.4)
        
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Energy')
    ax2.grid(True, alpha=0.5)
    
    # 动态计算放大图的 X 轴显示范围
    if len(segments) > 0:
        if zoom_indices is not None and len(zoom_indices) > 0:
            # 过滤掉非法的索引
            valid_indices = [i for i in zoom_indices if 0 <= i < len(segments)]
            if valid_indices:
                start_idx = min(valid_indices)
                end_idx = max(valid_indices)
                # 向前留出 0.2 秒，向后留出 0.5 秒的裕量
                zoom_start_time = max(0, (segments[start_idx][0] / sr) - 0.2)
                zoom_end_time = (segments[end_idx][1] / sr) + 0.5
                ax2.set_xlim(zoom_start_time, zoom_end_time)
                ax2.set_title(f'Steady-State Segmentation [ZOOMED VIEW] - Inspecting Pulses: {[i+1 for i in valid_indices]} (1-based)')
            else:
                ax2.set_title('Steady-State Segmentation [ZOOMED VIEW] - Invalid indices provided')
        else:
            # 默认放大前 3 个脉冲
            zoom_target = min(3, len(segments)) - 1
            zoom_start_time = 0
            zoom_end_time = (segments[zoom_target][1] / sr) + 0.5
            ax2.set_xlim(zoom_start_time, zoom_end_time)
            ax2.set_title('Steady-State Segmentation [ZOOMED VIEW] - Inspecting the first few pulses')
    else:
        ax2.set_xlim(0, 4.0)
        ax2.set_title('Steady-State Segmentation [ZOOMED VIEW]')

    plt.tight_layout()
    plt.show()


def segment_audio_steady_state(audio_data, sample_rate, expected_pulses, debug_plot=False, zoom_indices=None):
    """
    Locate each audio pulse by detecting the rising edge of the energy envelope,
    then skip the transient phase to capture only the steady-state window.
    This approach is specifically designed to eliminate startup transients
    and pop noises from physical acoustic cavities, ensuring high fidelity
    for physical neural network datasets.
    
    Returns:
        list of (start_idx, end_idx)  on success
        None                          if user aborts
    """
    cycle_s = AUDIO_DURATION + SILENCE_DURATION
    
    # --- 稳态截取核心参数 ---
    transient_skip_s = 0.15   # 延迟150ms，避开物理腔体和扬声器的起振瞬态
    steady_window_s  = 0.20   # 提取中间极其稳定的200ms用于计算
    
    transient_skip_samples = int(sample_rate * transient_skip_s)
    steady_window_samples  = int(sample_rate * steady_window_s)
    
    # 计算平滑能量包络（20ms窗口对上升沿敏感）
    smooth_window = int(sample_rate * 0.02) 
    envelope = np.convolve(np.abs(audio_data), np.ones(smooth_window)/smooth_window, mode='same')
    
    # 动态阈值计算（取前10%计算底噪）
    noise_floor = np.mean(envelope[:int(sample_rate * 0.1)]) 
    max_energy  = np.max(envelope)
    
    trigger_threshold = noise_floor + 0.15 * (max_energy - noise_floor)
    
    triggers = []
    cooldown = 0
    cooldown_samples = int(sample_rate * (cycle_s * 0.8)) # 冷却期，防止同一段声音内多次触发
    
    # 寻找越过阈值的上升沿
    for i in range(len(envelope)):
        if cooldown > 0:
            cooldown -= 1
            continue
            
        if i > 0 and envelope[i] > trigger_threshold and envelope[i-1] <= trigger_threshold:
            triggers.append(i)
            cooldown = cooldown_samples
            
    # 基于触发点，计算稳态窗口索引
    segments = []
    for trig in triggers:
        start_idx = trig + transient_skip_samples
        end_idx   = start_idx + steady_window_samples
        
        if end_idx <= len(audio_data):
            segments.append((start_idx, end_idx))
            
    # 数量校验与画图
    if len(segments) != expected_pulses:
        print(f"\n[Error] Segmentation mismatch. Found {len(segments)} pulses, expected {expected_pulses}.")
        print("  Hint: Check recording noise floor or adjust 'trigger_threshold'.")
        if debug_plot:
            _plot_steady_segmentation(envelope, triggers, segments, trigger_threshold, expected_pulses, sample_rate, zoom_indices)
        return None
        
    if debug_plot:
         _plot_steady_segmentation(envelope, triggers, segments, trigger_threshold, expected_pulses, sample_rate, zoom_indices)
         
    return segments


# ==========================================
# 2. High-Precision Frequency Feature Extraction
# ==========================================
def extract_frequency_features_high_precision(segment_data, sample_rate):
    """
    Extract frequency features using a Flattop window and Parabolic Sub-bin Interpolation 
    to eliminate scalloping loss and find the true physical peak. 
    This provides highly accurate linear amplitudes suitable for matrix multiplication 
    in physical neural network models.
    """
    n = len(segment_data)
    
    # 1. 物理级要求：使用平顶窗保证幅值绝对精度
    window = get_window('flattop', n)
    windowed = segment_data * window
    
    # 2. rFFT
    fft_result = np.fft.rfft(windowed)
    freqs      = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    
    # 3. 幅值归一化（修正平顶窗带来的能量衰减）
    amplitude_correction = 2.0 / np.sum(window) 
    magnitudes = np.abs(fft_result) * amplitude_correction
    
    linear_amps   = []
    analysis_detail = []
    
    for target_f in BASE_FREQUENCIES:
        idx_lo = np.searchsorted(freqs, target_f - FREQ_TOLERANCE)
        idx_hi = np.searchsorted(freqs, target_f + FREQ_TOLERANCE)
        
        if idx_lo >= idx_hi:
            idx_hi = idx_lo + 1
            
        # 找到容差范围内的最高离散频点
        local_peak_idx = idx_lo + np.argmax(magnitudes[idx_lo:idx_hi])
        
        # 4. 抛物线亚像素插值，寻找偏离离散刻度的真实峰值
        if 0 < local_peak_idx < len(magnitudes) - 1:
            alpha = magnitudes[local_peak_idx - 1]
            beta  = magnitudes[local_peak_idx]
            gamma = magnitudes[local_peak_idx + 1]
            
            # 抛物线顶点偏移量
            denom = alpha - 2*beta + gamma
            p = 0.5 * (alpha - gamma) / denom if denom != 0 else 0
            
            # 亚像素级实际物理幅值和频率
            true_amp = beta - 0.25 * (alpha - gamma) * p
            bin_resolution = freqs[1] - freqs[0]
            true_freq = freqs[local_peak_idx] + p * bin_resolution
            
        else:
            true_amp  = magnitudes[local_peak_idx]
            true_freq = freqs[local_peak_idx]
            
        # 修正计算误差可能导致的微小负数
        true_amp = max(0.0, float(true_amp))
        db_val   = 20 * np.log10(true_amp + 1e-12)
        
        linear_amps.append(true_amp)
        analysis_detail.append({
            'target_hz': target_f,
            'actual_hz': round(float(true_freq), 2),
            'linear_amp': true_amp,
            'db': round(float(db_val), 2)
        })
        
    return linear_amps, analysis_detail


# ==========================================
# 3. User Inputs
# ==========================================
def get_user_inputs():
    """
    Handle command-line interactive inputs to locate recordings and target dataset rows.
    """
    print("=" * 62)
    print("    Physical Neural Network  -  Audio Analyzer V2")
    print("=" * 62)

    # 录音文件夹路径
    raw = input(
        "\n请输入包含录音文件的 *_Recorded 文件夹路径\n"
        "(支持绝对路径或相对路径，拖拽进来的引号会自动去除):\n"
    ).strip().strip("\"'")

    if not os.path.isdir(raw):
        print(f"[错误] 找不到路径: {raw}")
        sys.exit(1)
    session_path = raw

    # 录音批次索引 y
    while True:
        try:
            y_index = int(input("\n请输入本次录音的批次序号 y（如输入 1 代表 _001.wav）: ").strip())
            if y_index >= 1:
                break
            print("  请输入大于 0 的整数。")
        except ValueError:
            print("  请输入有效整数。")
    y_str = f"{y_index:03d}"

    # 确认文件是否存在
    print(f"\n正在查找轨道文件（批次 {y_str}）：")
    track_paths = []
    for x in range(1, NUM_CHANNELS + 1):
        candidates = [
            os.path.join(session_path, f"轨道{x}_{y_str}.wav"),
            os.path.join(session_path, f"轨道 {x}_{y_str}.wav"),
        ]
        found = None
        for c in candidates:
            if os.path.exists(c):
                found = c
                break
        if found is None:
            print(f"  [错误] 找不到轨道 {x} 的文件，尝试过：")
            for c in candidates:
                print(f"    {c}")
            print("  请检查文件名格式后重试。")
            sys.exit(1)
        track_paths.append(found)
        print(f"  轨道 {x}: {os.path.basename(found)}")

    # 选择数据集文件
    print("\n请选择这段录音对应的原始数据集：")
    for i, name in enumerate(VALID_FILES):
        print(f"  [{i+1}] {name}.csv")
    while True:
        try:
            choice = int(input("输入编号 (1/2/3): ").strip()) - 1
            if 0 <= choice < len(VALID_FILES):
                target_name = VALID_FILES[choice]
                break
            print(f"  请输入 1 到 {len(VALID_FILES)} 之间的数字。")
        except ValueError:
            print("  请输入有效数字。")

    # 从原始 CSV 获取总行数
    source_csv = os.path.join(DATA_DIR, f"{target_name}.csv")
    if not os.path.exists(source_csv):
        print(f"[错误] 找不到 {source_csv}，请先运行 input_generator.py。")
        sys.exit(1)
    total_rows = len(pd.read_csv(source_csv, header=None))

    # 输入行号
    while True:
        try:
            start_row = int(input(f"\n请输入对应的起始行号（1 ~ {total_rows}）: ").strip())
            end_row   = int(input(f"请输入对应的结束行号（{start_row} ~ {total_rows}）: ").strip())
            if 1 <= start_row <= end_row <= total_rows:
                break
            print(f"  行号无效，请确保 1 <= 起始 <= 结束 <= {total_rows}。")
        except ValueError:
            print("  请输入有效整数。")

    expected_pulses = end_row - start_row + 1
    return session_path, track_paths, target_name, start_row, end_row, expected_pulses


# ==========================================
# 4. Load WAV Tracks
# ==========================================
def load_tracks(track_paths):
    """
    Read 4 channel WAVs, standardize to float32, and ensure sample rate consistency.
    """
    tracks_data = []
    sample_rate = None

    for i, fpath in enumerate(track_paths):
        sr, data = wavfile.read(fpath)

        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            print(f"[错误] 轨道 {i+1} 的采样率 {sr} Hz 与轨道 1 的 {sample_rate} Hz 不一致！")
            sys.exit(1)

        # 统一转为 float32
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            data = (data.astype(np.float32) - 128.0) / 128.0
        else:
            data = data.astype(np.float32)

        # 降维处理
        if data.ndim > 1:
            print(f"  [提示] 轨道 {i+1} 为 {data.shape[1]} 声道，自动取第一声道分析。")
            data = data[:, 0]

        tracks_data.append(data)
        print(f"  轨道 {i+1}：{len(data)} 采样点，{len(data)/sr:.2f}s，{sr} Hz")

    return tracks_data, sample_rate


# ==========================================
# 5. Main Processing Flow
# ==========================================
def process_recorded_session():
    """
    Core pipeline: Load audio -> Segment steady state -> Extract physical features -> Write CSV
    """
    session_path, track_paths, target_name, start_row, end_row, expected_pulses = get_user_inputs()

    print("\n--- 正在加载音频文件 ---")
    tracks_data, sample_rate = load_tracks(track_paths)

    # 交互询问是否需要查看特定片段
    zoom_input = input("\n[可选] 是否查看特定片段的放大图？\n直接回车默认看前3段，或输入逗号分隔的序号(例如 '10,11' 看第10和11段): ").strip()
    zoom_list = None
    if zoom_input:
        try:
            # 将用户输入的 1-based 序号转换为 0-based 索引
            zoom_list = [int(x.strip()) - 1 for x in zoom_input.split(',')]
        except ValueError:
            print("  输入格式有误，将使用默认视图。")
            zoom_list = None

    # 切割音频（强制开启 debug_plot 以确认切片准确性）
    print(f"\n--- 正在执行稳态能量截取算法（期望 {expected_pulses} 段）---")
    segments = segment_audio_steady_state(
        tracks_data[0], sample_rate, expected_pulses, debug_plot=True, zoom_indices=zoom_list
    )
    
    if segments is None:
        print("\n请检查切片图表，修改参数或重新选取音频段落后重试。")
        sys.exit(1)

    print(f"  成功定位 {expected_pulses} 段高保真稳态发声区间。")

    print(f"\n--- 高精度频域特征提取（4 轨道 × {expected_pulses} 段）---")
    final_matrix = np.zeros((expected_pulses, TOTAL_DIMS), dtype=np.float32)

    for seg_idx, (s_idx, e_idx) in enumerate(segments):
        row_vector = []

        if seg_idx == 0:
            print(f"\n[第一段详细报告]（对应数据行 {start_row}）：")

        for ch_idx in range(NUM_CHANNELS):
            seg_data = tracks_data[ch_idx][s_idx:e_idx]
            # 调用全新的高精度提取算法
            amps, details = extract_frequency_features_high_precision(seg_data, sample_rate)
            row_vector.extend(amps)

            if seg_idx == 0:
                print(f"  轨道 {ch_idx + 1}（→ 第 {ch_idx*NUM_FREQ+1}~{(ch_idx+1)*NUM_FREQ} 维）：")
                for d in details:
                    print(f"    目标 {d['target_hz']:>4} Hz  →  "
                          f"实际 {d['actual_hz']:>7.2f} Hz  |  "
                          f"幅值: {d['linear_amp']:.5f}  |  "
                          f"{d['db']:>8.2f} dB")

        final_matrix[seg_idx] = row_vector

        # 进度条
        done = seg_idx + 1
        bar  = '█' * (done * 30 // expected_pulses) + '░' * (30 - done * 30 // expected_pulses)
        print(f"\r  [{bar}] {done}/{expected_pulses}", end='', flush=True)

    print()

    # 更新物理数据集
    print(f"\n--- 正在更新物理数据集 ---")
    os.makedirs(DATA_DIR, exist_ok=True)

    source_csv = os.path.join(DATA_DIR, f"{target_name}.csv")
    phys_csv   = os.path.join(DATA_DIR, f"phys_{target_name}.csv")

    source_df = pd.read_csv(source_csv, header=None)
    total_rows = len(source_df)

    if os.path.exists(phys_csv):
        phys_df = pd.read_csv(phys_csv, header=None)
        if len(phys_df) != total_rows:
            print(f"  [警告] {phys_csv} 行数 ({len(phys_df)}) 与原始文件不符 ({total_rows})，重建。")
            phys_df = pd.DataFrame(np.zeros((total_rows, TOTAL_DIMS), dtype=np.float32))
    else:
        print(f"  首次创建: {phys_csv}")
        phys_df = pd.DataFrame(np.zeros((total_rows, TOTAL_DIMS), dtype=np.float32))

    # 1-based 转 0-based 写入
    py_start = start_row - 1
    py_end   = end_row       
    phys_df.iloc[py_start:py_end] = final_matrix

    phys_df.to_csv(phys_csv, index=False, header=False)

    print(f"\n{'='*62}")
    print(f"  分析完成！")
    print(f"{'='*62}")
    print(f"  数据集     : {target_name}")
    print(f"  写入行范围 : {start_row} ~ {end_row}（共 {expected_pulses} 行）")
    print(f"  输出文件   : {phys_csv}")
    print()


if __name__ == "__main__":
    process_recorded_session()