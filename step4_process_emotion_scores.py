import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端以支持无显示
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
from pathlib import Path


# 配置路径
BASE_DIR = str(Path(__file__).resolve().parent)
DATA_DIR = os.path.join(BASE_DIR, "results", "prompt_v2")
RAW_SCORES_DIR = os.path.join(BASE_DIR, "results", "raw_scores")
SMOOTH_SCORES_DIR = os.path.join(BASE_DIR, "results", "smooth_scores")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")

# 确保输出目录存在
os.makedirs(RAW_SCORES_DIR, exist_ok=True)
os.makedirs(SMOOTH_SCORES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def parse_response_content(content, mode=1):
    """
    解析API返回的response字符串，提取每秒的6维情绪评分。
    mode=1/2: 逐行解析，每行包含"第"字样 (1秒一次)
    mode=3: 只有一次评分 (10秒一次)，格式如 "高兴: 0.0; 惊讶: 2.0; ..."
    返回: list of [高兴, 惊讶, 悲伤, 愤怒, 厌恶, 恐惧]
    """
    scores_list = []
    emotions = ["高兴", "惊讶", "悲伤", "愤怒", "厌恶", "恐惧"]

    if mode == 3:
        # FIXME. debug一下mode3是什么情况
        # Mode 3: 只有一次评分
        current_scores = []
        for emotion in emotions:
            try:
                pattern = fr"{emotion}:\s*(\[?[\d\.]+\]?)"
                match = re.search(pattern, content)
                if match:
                    val_str = match.group(1)
                    val_str = val_str.replace('[', '').replace(']', '')
                    score = float(val_str)
                else:
                    score = 0.0
                    print(f"Warning: Could not find score for {emotion} in mode 3 content")  # FIXME
                current_scores.append(score)
            except Exception as e:
                print(f"Error parsing {emotion} in mode 3. Error: {e}")
                current_scores.append(0.0)

        if len(current_scores) == 6:
            scores_list.append(current_scores)

    else:
        # Mode 1/2: 逐行解析
        lines = content.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line or "第" not in line:
                continue

            current_scores = []
            for emotion in emotions:
                # 查找形如 "高兴: 0.5" 的模式
                # 使用简单的字符串查找或正则
                try:
                    # 寻找 emotion 后面的冒号和数字
                    # 格式可能是 "高兴: 0.5;" 或 "高兴: 0.5"
                    pattern = fr"{emotion}:\s*(\[?[\d\.]+\]?)"
                    match = re.search(pattern, line)
                    if match:
                        val_str = match.group(1)
                        # 处理可能存在的方括号
                        val_str = val_str.replace('[', '').replace(']', '')
                        score = float(val_str)
                    else:
                        score = 0.0  # 默认值，或者抛出警告
                        print(f"Warning: Could not find score for {emotion} in line: {line}")
                    current_scores.append(score)
                except Exception as e:
                    print(f"Error parsing {emotion} in line: {line}. Error: {e}")
                    current_scores.append(0.0)

            if len(current_scores) == 6:
                scores_list.append(current_scores)

    return np.array(scores_list)


def read_and_merge_scores(data_dir, video_id):
    if video_id == "rzdf_final_segment_2":
        pass
    """
    读取并合并指定视频ID的所有评分文件
    """

    # files = [f for f in os.listdir(data_dir) if f.startswith(video_id) and f.endswith('.json')]
    # files = sorted(from_episode_to_clip(video_id))
    def get_start_second(path_obj):
        path_str = str(path_obj)
        match = re.search(r'segment_\d_(\d+)s', path_str)
        if match:
            return int(match.group(1))
        return 0  # 如果没匹配到，排在最前面

    files = sorted(list(Path(data_dir).glob(f"{video_id}_*.json")), key=get_start_second)

    all_scores = []

    print(f"Found {len(files)} files for video episode {video_id}")

    for filename in files:
        print(f"Processing {filename}...")
        _mode = int(str(filename.stem).split('_')[-1])
        file_path = str(filename)

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        response_text = data.get('response', '')
        if not response_text:
            raise ValueError(f"No response text found in file: {file_path}")

        scores = parse_response_content(response_text, mode=_mode)
        assert scores.size > 0
        if scores.size > 0:
            if _mode == 3:
                # Mode 3: 10秒一次评分，需要扩展到每一秒
                duration = 10
                # 复制分数 duration 次
                # scores 是 (1, 6) 的数组
                expanded_scores = np.tile(scores, (duration, 1))
                all_scores.append(expanded_scores)
            else:

                all_scores.append(scores)

    if not all_scores:
        return np.array([]), np.array([])

    # 垂直堆叠
    merged_scores = np.vstack(all_scores)
    # 生成时间轴 (假设每行代表1秒)
    raw_times = np.arange(merged_scores.shape[0])

    return raw_times, merged_scores


def smooth_scores_window(raw_times, raw_scores, window_size=10, step_size=2):
    """
    使用中心滑动窗口平滑数据
    window_size: 窗口总大小 (前后各 window_size/2)
    step_size: 步长
    """
    if raw_scores.size == 0:
        return np.array([]), np.array([])

    half_win = window_size / 2
    max_time = raw_times[-1]

    target_times = np.arange(0, max_time + 1, step_size)
    smoothed_scores = []

    for t in target_times:
        # 定义窗口范围 [t - half_win, t + half_win]
        # 注意: 这里包含边界
        mask = (raw_times >= (t - half_win)) & (raw_times <= (t + half_win))

        # 提取窗口内的分数
        scores_in_window = raw_scores[mask]

        if scores_in_window.size > 0:
            # 计算均值
            mean_scores = np.mean(scores_in_window, axis=0)
        else:
            # 如果窗口内没有数据（理论上不应该发生，除非数据缺失），沿用上一个或全0
            mean_scores = np.zeros(6)

        smoothed_scores.append(mean_scores)

    return target_times, np.array(smoothed_scores)


def save_scores(output_dir, video_id, times, scores, suffix):
    """
    保存分数为CSV文件
    """
    if scores.size == 0:
        print("No data to save.")
        return

    df = pd.DataFrame(scores, columns=["Happy", "Surprise", "Sad", "Anger", "Disgust", "Fear"])
    df.insert(0, "Time", times)

    output_path = os.path.join(output_dir, f"{video_id}_{suffix}.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved {suffix} scores to: {output_path}")


def plot_scores(times, scores, video_id):
    """
    绘图并保存
    """
    if scores.size == 0:
        return

    colors = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']
    emos = ["Happiness", "Surprise", "Sadness", "Anger", "Disgust", "Fear"]

    plt.figure(figsize=(15, 12))

    for j in range(6):
        plt.subplot(6, 1, j + 1)
        plt.plot(times, scores[:, j], color=colors[j], linewidth=1.5)
        plt.xlim([0, max(times) if len(times) > 0 else 10])
        plt.ylim([0, 7])
        plt.ylabel('Scores')
        plt.xlabel('Time/s')
        plt.title(emos[j])
        plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, f"{video_id}_smoothed_plot.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved plot to: {save_path}")
    # plt.show() # 在非交互式环境中不需要


def main():
    # 获取电影episode列表
    ___ = sorted(list(Path('/home/liaoyizhi/codes/ncclab_ddh/results/prompt_v2').glob('*.json')))
    # ___ = sorted(list(Path('/home/liaoyizhi/codes/ncclab_ddh/results_mode2/prompt_v2').glob('*_mode_2.json')))
    # ___ = sorted(list(Path('/home/liaoyizhi/codes/ncclab_ddh/results_mode3/prompt_v2').glob('*_mode_3.json')))
    episodes_li = sorted(
        list(
            set(
                f'{__.stem.split('_')[0]}_{__.stem.split('_')[1]}_{__.stem.split('_')[2]}_{__.stem.split('_')[3]}'
                for __ in ___
            )
        )
    )

    for episode in episodes_li:
        print(f"Start processing video: {episode}")

        # 1. 读取并合并
        # DATA_DIR: '/home/liaoyizhi/codes/ncclab_ddh/results/prompt_v2'
        # episode: 'agzz_final_segment_0'
        raw_times, raw_scores = read_and_merge_scores(DATA_DIR, episode)
        print(f"Merged raw data shape: {raw_scores.shape}")

        if raw_scores.size == 0:
            print("Error: No data found.")
            return

        # 2. 保存原始数据
        save_scores(RAW_SCORES_DIR, episode, raw_times, raw_scores, "raw")

        # 3. 平滑处理
        # 时间窗10秒，平移2秒
        smooth_times, smooth_data = smooth_scores_window(raw_times, raw_scores, window_size=10, step_size=2)
        print(f"Smoothed data shape: {smooth_data.shape}")

        # 4. 保存平滑数据
        save_scores(SMOOTH_SCORES_DIR, episode, smooth_times, smooth_data, "smooth")

        # 5. 绘图
        plot_scores(smooth_times, smooth_data, episode)

        print("Done.")


if __name__ == "__main__":
    main()
