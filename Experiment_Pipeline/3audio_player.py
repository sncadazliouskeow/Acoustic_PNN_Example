import os
import sys
import time
import threading
import sounddevice as sd
import soundfile as sf
import numpy as np

# ==========================================
# 全局配置
# ==========================================
AUDIO_DIR        = 'audio'
VALID_FILES      = ['train_pos', 'train_neg', 'test_embedded_inputs']
NUM_CHANNELS     = 4          # 扬声器数量（音频声道数）
SILENCE_DURATION = 0.5        # 段间静音时长（秒）
CONFIG_FILE      = 'audio_device_config.txt'   # 设备配置缓存

# 播放控制标志（跨线程共享）
_stop_flag = threading.Event()


# ==========================================
# 1. 设备管理
# ==========================================
def list_output_devices():
    """列出所有支持多声道输出的音频设备。"""
    devices = sd.query_devices()
    output_devices = []
    for i, dev in enumerate(devices):
        if dev['max_output_channels'] >= NUM_CHANNELS:
            output_devices.append((i, dev))
    return output_devices


def select_device():
    """
    交互式选择播放设备。
    如果存在缓存配置且用户确认，直接复用；否则重新选择并保存。
    """
    # 尝试读取缓存
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            lines = f.read().strip().splitlines()
        if lines:
            try:
                cached_idx  = int(lines[0])
                cached_name = lines[1] if len(lines) > 1 else ''
                print(f"\n[缓存配置] 上次使用的播放设备：")
                print(f"  设备索引: {cached_idx}")
                print(f"  设备名称: {cached_name}")
                ans = input("是否继续使用此设备？(y/n，默认 y): ").strip().lower()
                if ans != 'n':
                    print(f"  使用缓存设备: [{cached_idx}] {cached_name}")
                    return cached_idx
            except (ValueError, IndexError):
                pass

    # 重新选择
    print("\n正在扫描支持 4 声道输出的音频设备...")
    output_devices = list_output_devices()

    if not output_devices:
        print("[错误] 未找到支持 4 声道输出的设备。")
        print("  请检查：")
        print("  1. 音频接口/声卡是否已连接并安装驱动")
        print("  2. 设备是否支持多声道输出（至少4通道）")
        sys.exit(1)

    print(f"\n找到 {len(output_devices)} 个可用设备：\n")
    print(f"  {'编号':<6} {'设备索引':<10} {'最大输出声道':<14} {'设备名称'}")
    print(f"  {'-'*60}")
    for i, (dev_idx, dev) in enumerate(output_devices):
        print(f"  [{i+1:<4}] {dev_idx:<10} {dev['max_output_channels']:<14} {dev['name']}")

    while True:
        try:
            choice = int(input("\n请选择设备编号: ").strip())
            if 1 <= choice <= len(output_devices):
                dev_idx, dev = output_devices[choice - 1]
                break
            print(f"  请输入 1 到 {len(output_devices)} 之间的数字。")
        except ValueError:
            print("  请输入有效数字。")

    # 声道映射说明
    print(f"\n  已选择: [{dev_idx}] {dev['name']}")
    print(f"  声道映射（请确认物理接线）：")
    for ch in range(NUM_CHANNELS):
        print(f"    音频声道 {ch+1}  →  扬声器 {ch+1}")
    confirm = input("\n接线确认无误？(y/n): ").strip().lower()
    if confirm != 'y':
        print("请检查接线后重新运行程序。")
        sys.exit(0)

    # 保存缓存
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"{dev_idx}\n{dev['name']}\n")
    print(f"  设备配置已保存至 {CONFIG_FILE}")

    return dev_idx


# ==========================================
# 2. 设备原生采样率查询
# ==========================================
def get_device_sample_rate(device_idx: int) -> int:
    """
    查询设备支持的原生采样率。
    若设备的 default_samplerate 不在常见列表中，则依次尝试候选值。
    返回设备实际可用的采样率。
    """
    dev_info = sd.query_devices(device_idx)
    native_sr = int(dev_info['default_samplerate'])

    # 常见采样率候选列表（从高到低）
    candidates = [native_sr, 48000, 44100, 96000, 192000, 32000, 22050, 16000]
    # 去重保序
    seen = set()
    candidates = [x for x in candidates if not (x in seen or seen.add(x))]

    for sr in candidates:
        try:
            sd.check_output_settings(device=device_idx,
                                     channels=NUM_CHANNELS,
                                     dtype='float32',
                                     samplerate=sr)
            if sr != native_sr:
                print(f"  [提示] 设备原生采样率 {native_sr} Hz 不可用，"
                      f"改用 {sr} Hz。")
            return sr
        except Exception:
            continue

    # 所有候选都失败，抛出明确错误
    raise RuntimeError(
        f"找不到设备 [{device_idx}] 支持的采样率。\n"
        f"  设备信息: {dev_info['name']}\n"
        f"  已尝试: {candidates}\n"
        f"  请检查驱动或更换音频设备。"
    )


# ==========================================
# 3. 设备测试（可选）
# ==========================================
def test_device(device_idx: int):
    """
    依次在4个声道单独发出短促测试音，帮助用户确认扬声器对应关系。
    自动使用设备原生采样率，避免 paInvalidSampleRate 错误。
    """
    print("\n[设备测试] 将依次测试 4 个声道（每个 0.3 秒）...")
    print("  请确认各扬声器发声顺序与预期一致。\n")

    # 使用设备原生采样率
    sample_rate = get_device_sample_rate(device_idx)
    print(f"  使用采样率: {sample_rate} Hz\n")

    duration  = 0.3
    freq      = 1000   # 1kHz 测试音
    n_samples = int(sample_rate * duration)
    t         = np.linspace(0, duration, n_samples, endpoint=False)
    tone      = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    for ch in range(NUM_CHANNELS):
        print(f"  测试声道 {ch+1} / 扬声器 {ch+1} ...", end='', flush=True)
        audio = np.zeros((n_samples, NUM_CHANNELS), dtype=np.float32)
        audio[:, ch] = tone
        sd.play(audio, samplerate=sample_rate, device=device_idx)
        sd.wait()
        time.sleep(0.2)
        print(" 完成")

    print("\n  测试完成。")
    confirm = input("各扬声器对应关系是否正确？(y/n): ").strip().lower()
    return confirm == 'y'


# ==========================================
# 3. 用户交互：选择播放范围
# ==========================================
def get_play_range():
    """选择目标数据文件及播放行范围（1-based 闭区间）。"""
    print("\n请选择要播放的数据文件：")
    for i, name in enumerate(VALID_FILES):
        print(f"  [{i+1}] {name}")
    while True:
        try:
            choice = int(input("输入编号 (1/2/3): ").strip())
            if 1 <= choice <= len(VALID_FILES):
                target_name = VALID_FILES[choice - 1]
                break
            print(f"  请输入 1 到 {len(VALID_FILES)} 之间的数字。")
        except ValueError:
            print("  请输入有效数字。")

    # 扫描该目录下有多少个 combined_row_XXXX.wav
    audio_subdir = os.path.join(AUDIO_DIR, target_name)
    if not os.path.isdir(audio_subdir):
        print(f"[错误] 目录不存在: {audio_subdir}")
        print("  请先运行 audio_generator.py 生成音频文件。")
        sys.exit(1)

    combined_files = sorted([
        f for f in os.listdir(audio_subdir)
        if f.startswith('combined_row_') and f.endswith('.wav')
    ])
    if not combined_files:
        print(f"[错误] 在 {audio_subdir} 中未找到任何 combined_row_XXXX.wav 文件。")
        sys.exit(1)

    # 从文件名解析已有的行号范围
    available_rows = []
    for fname in combined_files:
        try:
            row_num = int(fname.replace('combined_row_', '').replace('.wav', ''))
            available_rows.append(row_num)
        except ValueError:
            pass
    available_rows.sort()
    min_row, max_row = available_rows[0], available_rows[-1]

    print(f"\n  目录: {audio_subdir}")
    print(f"  已找到 {len(available_rows)} 个音频文件，行号范围: {min_row} ~ {max_row}")

    while True:
        try:
            start = int(input(f"请输入起始行号 ({min_row} ~ {max_row}): ").strip())
            end   = int(input(f"请输入结束行号 ({start} ~ {max_row}): ").strip())
            if min_row <= start <= end <= max_row:
                # 检查范围内的文件是否都存在
                missing = [r for r in range(start, end + 1) if r not in available_rows]
                if missing:
                    print(f"  [警告] 以下行号的音频文件缺失: {missing[:10]}{'...' if len(missing)>10 else ''}")
                    ans = input("  是否跳过缺失文件继续播放？(y/n): ").strip().lower()
                    if ans != 'y':
                        continue
                break
            print(f"  行号无效，请确保 {min_row} <= 起始行 <= 结束行 <= {max_row}。")
        except ValueError:
            print("  请输入有效整数。")

    return target_name, audio_subdir, start, end, available_rows


# ==========================================
# 4. 停止监听线程
# ==========================================
def _listen_for_stop():
    """后台线程：等待用户按 Enter 设置停止标志。"""
    input()   # 阻塞直到用户按 Enter
    _stop_flag.set()


# ==========================================
# 5. 播放主循环
# ==========================================
def play_sequence(audio_subdir: str, start_row: int, end_row: int,
                  available_rows: list, device_idx: int):
    """
    按顺序播放指定行范围的 combined_row_XXXX.wav。
    采用全量内存预加载 + 连续音频流直通方案。
    彻底解决硬盘 IO 瓶颈、内存溢出（-9992）及物理断流（兹拉声）问题。
    """
    rows_to_play = [r for r in range(start_row, end_row + 1) if r in available_rows]
    n_total      = len(rows_to_play)

    # 查询设备原生采样率（一次即可，整个播放序列共用）
    device_sr = get_device_sample_rate(device_idx)
    
    # ================= 阶段 1：内存预加载 =================
    print(f"\n[准备阶段] 正在将 {n_total} 段音频预加载至内存 (消除硬盘读写延迟)...")
    playlist_data = []
    
    # 提前生成用于补齐的静音数据
    silence_samples = int(device_sr * SILENCE_DURATION)
    silence_data = np.zeros((silence_samples, NUM_CHANNELS), dtype=np.float32)

    for idx, row_num in enumerate(rows_to_play):
        fname = f"combined_row_{row_num:04d}.wav"
        fpath = os.path.join(audio_subdir, fname)

        if not os.path.exists(fpath):
            continue

        # 从硬盘读取音频
        audio_data, file_sr = sf.read(fpath, dtype='float32')

        # 严格校验采样率
        if file_sr != device_sr:
            print(f"\n[严重错误] 音频文件采样率 ({file_sr}Hz) 与设备 ({device_sr}Hz) 不匹配！")
            print("为保证物理实验的高保真度，严禁在播放时进行软件重采样。")
            print("请修改 audio_generator.py 中的 SAMPLE_RATE 并重新生成音频。")
            sys.exit(1)

        # 确保是 4 声道对齐
        if audio_data.ndim == 1:
            audio_data = np.tile(audio_data[:, np.newaxis], (1, NUM_CHANNELS))
        elif audio_data.shape[1] < NUM_CHANNELS:
            pad = np.zeros((audio_data.shape[0], NUM_CHANNELS - audio_data.shape[1]), dtype=np.float32)
            audio_data = np.concatenate([audio_data, pad], axis=1)

        # 在内存中拼接：发声区 + 静音区
        audio_with_silence = np.concatenate((audio_data, silence_data), axis=0)
        playlist_data.append((row_num, fname, audio_with_silence))

        # 打印预加载进度
        if (idx + 1) % 10 == 0 or (idx + 1) == n_total:
            print(f"  已预加载: {idx + 1}/{n_total}...")

    # ================= 阶段 2：连续流直通播放 =================
    print(f"\n[播放开始] 物理直通通道已建立 | 设备采样率 {device_sr} Hz")
    print(f"  按 Enter 随时终止播放\n")
    print(f"  {'进度':<8} {'行号':<8} {'状态'}")
    print(f"  {'-'*40}")

    _stop_flag.clear()
    listener = threading.Thread(target=_listen_for_stop, daemon=True)
    listener.start()

    try:
        # 使用 OutputStream 建立持续连接。
        # blocksize=4096 和 latency='high' 专为解决断流/欠载设计。
        with sd.OutputStream(samplerate=device_sr, 
                             device=device_idx, 
                             channels=NUM_CHANNELS, 
                             dtype='float32',
                             blocksize=4096,
                             latency='high') as stream:
            
            for idx, (row_num, fname, audio_chunk) in enumerate(playlist_data):
                if _stop_flag.is_set():
                    print(f"\n  [用户终止] 已在第 {idx}/{n_total} 段停止。")
                    break

                progress = f"{idx+1}/{n_total}"
                print(f"  {progress:<8} {row_num:<8} 播放中...", end='', flush=True)

                # 将内存中的拼接数据块写入底层缓冲区
                # .write() 会阻塞直到这段数据完全流入缓冲区，代替了旧的 time.sleep()
                stream.write(audio_chunk)
                
                print(f"\r  {progress:<8} {row_num:<8} ✓       ", flush=True)

    except Exception as e:
        print(f"\n[音频流错误] 播放过程中断: {e}")

    if not _stop_flag.is_set():
        print(f"\n  [播放完毕] 全部 {n_total} 段已物理直通输出。")

# ==========================================
# 主流程
# ==========================================
def main():
    print("=" * 58)
    print("        音频播放器  -  Audio Player")
    print("=" * 58)

    # 1. 设备选择
    device_idx = select_device()

    # 2. 可选：设备测试
    ans = input("\n是否进行扬声器声道测试？(y/n，默认 n): ").strip().lower()
    if ans == 'y':
        ok = test_device(device_idx)
        if not ok:
            print("声道对应有误，请重新检查接线后运行程序。")
            # 清除缓存，下次强制重新选择
            if os.path.exists(CONFIG_FILE):
                os.remove(CONFIG_FILE)
            sys.exit(0)

    # 3. 选择播放范围
    target_name, audio_subdir, start_row, end_row, available_rows = get_play_range()

    # 4. 播放摘要确认
    n_play = len([r for r in range(start_row, end_row + 1) if r in available_rows])
    total_time = n_play * (0.5 + SILENCE_DURATION)   # 音频 0.5s + 静音 0.5s
    print(f"\n[播放计划]")
    print(f"  数据文件 : {target_name}")
    print(f"  行范围   : {start_row} ~ {end_row}")
    print(f"  有效段数 : {n_play}")
    print(f"  预计耗时 : {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
    print(f"  段间静音 : {SILENCE_DURATION} 秒")

    confirm = input("\n确认开始播放？(y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消。")
        sys.exit(0)

    # 5. 播放
    play_sequence(audio_subdir, start_row, end_row, available_rows, device_idx)


if __name__ == "__main__":
    main()
