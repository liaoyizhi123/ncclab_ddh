"""
读取弹幕（如果有）和视频描述，调用LLM获取情绪评分）
"""

import json
import os
import re
import requests
import pandas as pd
from utils.danmu_cache_handler import get_video_danmu_cache, format_danmu_by_second, convert_to_seconds

# ================= 配置区域 =================
# 1. 目标视频ID列表 (硬编码)
# TARGET_VIDEO_IDS = ['neg_s_2'] # 示例，用户可在此修改
video_offset_dict = {
    'agzz': [
        0 * 60 + 7,
        15 * 60 + 31,
        30 * 60 + 37,
        46 * 60 + 14,
        62 * 60 + 5,
        76 * 60 + 14,
        91 * 60 + 47,
        106 * 60 + 43,
        122 * 60 + 19,
    ],
    'lldq2': [
        0 * 60 + 0,
        14 * 60 + 50,
        29 * 60 + 35,
        44 * 60 + 57,
        59 * 60 + 41,
        73 * 60 + 46,
        87 * 60 + 59,
        103 * 60 + 39,
        118 * 60 + 12,
        132 * 60 + 51,
        146 * 60 + 58,
    ],
    'rzdf': [
        0 * 60 + 0,
        14 * 60 + 27,
        30 * 60 + 22,
        46 * 60 + 2,
        60 * 60 + 14,
        74 * 60 + 43,
        89 * 60 + 24,
        104 * 60 + 16,
        118 * 60 + 52,
    ],
    'sdd': [
        0 * 60 + 0,
        14 * 60 + 12,
        29 * 60 + 14,
        44 * 60 + 54,
        59 * 60 + 55,
        75 * 60 + 3,
        89 * 60 + 51,
        103 * 60 + 40,
        119 * 60 + 4,
    ],
    'xltfn': [1 * 60 + 18, 19 * 60 + 29, 35 * 60 + 6, 51 * 60 + 12, 64 * 60 + 25, 78 * 60 + 35],
    'ymlw': [
        0 * 60 + 47,
        16 * 60 + 37,
        31 * 60 + 29,
        46 * 60 + 25,
        61 * 60 + 6,
        75 * 60 + 15,
        89 * 60 + 11,
        104 * 60 + 17,
    ],
    'zccsh': [
        0 * 60 + 7,
        15 * 60 + 9,
        30 * 60 + 13,
        44 * 60 + 49,
        60 * 60 + 14,
        74 * 60 + 50,
        89 * 60 + 45,
        104 * 60 + 28,
        120 * 60 + 11,
    ],
    'zgjqxs': [None, None, None, None, None],
}

# 2. 路径配置
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = BASE_DIR / "clips"
PATH_VIDEO_START_TIME = os.path.join(str(BASE_DIR), "video_start_time.json")  # 用不到
# DIR_DESCRIPTIONS = str(BASE_DIR/ 'forrest_output' / 'ViLAMP-llava-qwen-1_1.jsonl')
DIR_DESCRIPTIONS = str(BASE_DIR)
DIR_RAW_DANMU = os.path.join(BASE_DIR, "Raw_danmu")
DIR_DANMU_CACHE = os.path.join(BASE_DIR, "danmu_per_second")
DIR_OUTPUT = os.path.join(BASE_DIR, "results", "prompt_v2")

# 3. 控制开关
CALL_LLM = True  # 是否实际调用LLM API
FORCE_REBUILD_PROMPT = False  # 是否强制重新构建Prompt (即使结果文件已存在)
MODE = 1  # 1: 视频+弹幕, 2: 仅弹幕, 3: 仅视频

# 4. LLM API 配置
DEEPSEEK_API_KEY = "sk-e97c561fb4ab42ed9b5e07453f181c84"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"


# 5. LLM API 调用函数
def call_llm_api(prompt):
    if not CALL_LLM:
        # 如果不调用LLM，返回模拟数据或空字符串
        print("Skipping LLM call (CALL_LLM=False)")
        return "Simulated Response"

    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    data = {"model": DEEPSEEK_MODEL, "messages": [{"role": "user", "content": prompt}], "stream": False}

    try:
        print("Calling DeepSeek API...")
        response = requests.post(f"{DEEPSEEK_BASE_URL}/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content']
        print("API Call Success")
        return content
    except Exception as e:
        print(f"LLM API Call Error: {e}")
        return f"Error: {e}"


# ===========================================


def parse_time_str(time_obj):
    """解析 video_start_time.json 中的时间格式"""
    if isinstance(time_obj, dict):
        # 处理 {"start_time": "0:24:36", "duration": "45"} 这种情况
        t_str = time_obj.get("start_time", "0:00:00")
        return convert_to_seconds(t_str)
    else:
        # 处理直接是字符串的情况 "0:24:36"
        return convert_to_seconds(time_obj)


def get_duration(time_obj):
    """获取视频总时长"""
    if isinstance(time_obj, dict):
        if "duration" not in time_obj:
            raise ValueError(f"Metadata missing 'duration' field: {time_obj}")
        return int(time_obj["duration"])
    else:
        # 如果json里只有开始时间没有duration，直接报错
        raise ValueError(f"Metadata format error (expected dict with 'duration'): {time_obj}")


def load_video_metadata():
    with open(PATH_VIDEO_START_TIME, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_prompt(mode, actual_len, video_desc, danmu_text):
    prompt = ""
    if mode == 1:  # 视频 + 弹幕
        prompt = f"""任务：请综合以下对视频所传达情感的描述和视频弹幕，推测观众在观看对应视频时的情绪感受。对高兴、惊讶、悲伤、愤怒、厌恶、恐惧六种情绪类别进行评价，评分为0到7之间的连续取值（可精确到小数点后一位），0表示完全没有，7表示非常强烈。

视频情绪描述（{actual_len}秒整体）：{video_desc}

弹幕（按秒列出）：
{danmu_text}

请严格按以下格式给出{actual_len}行评分结果，每一行代表一秒的情绪（无需返回额外的评论）：
第1秒: 高兴: [评分]; 惊讶: [评分]; 悲伤: [评分]; 愤怒: [评分]; 厌恶: [评分]; 恐惧: [评分]
... (以此类推到第{actual_len}秒)
"""
    elif mode == 2:  # 仅弹幕
        prompt = f"""任务：请仅根据以下视频弹幕，推测观众在观看对应视频时的情绪感受。若某秒没有弹幕，请参考最近的弹幕氛围。对高兴、惊讶、悲伤、愤怒、厌恶、恐惧六种情绪类别进行评价，评分为0到7之间的连续取值（可精确到小数点后一位），0表示完全没有，7表示非常强烈。

弹幕（按秒列出）：
{danmu_text}

请严格按以下格式给出{actual_len}行评分结果，每一行代表一秒的情绪（无需返回额外的评论）：
第1秒: 高兴: [评分]; 惊讶: [评分]; 悲伤: [评分]; 愤怒: [评分]; 厌恶: [评分]; 恐惧: [评分]
... (以此类推到第{actual_len}秒)
"""
    elif mode == 3:  # 仅视频
        prompt = f"""任务：请综合以下对视频所传达情感的描述，推测观众在观看对应视频时的情绪感受。对高兴、惊讶、悲伤、愤怒、厌恶、恐惧六种情绪类别进行评价，评分为0到7之间的连续取值（可精确到小数点后一位），0表示完全没有，7表示非常强烈。

视频情绪描述：{video_desc}

请严格按以下格式给出评分结果（无需返回额外的评论）：高兴: [评分]; 惊讶: [评分]; 悲伤: [评分]; 愤怒: [评分]; 厌恶: [评分]; 恐惧: [评分]
"""
    return prompt


def main():
    # 0. 准备目录
    if not os.path.exists(DIR_OUTPUT):
        os.makedirs(DIR_OUTPUT)
    if not os.path.exists(DIR_DANMU_CACHE):
        os.makedirs(DIR_DANMU_CACHE)

    # 1. 加载元数据
    # print("Loading metadata...")
    # metadata = load_video_metadata()

    # iterate data/CineBrain_8s
    TARGET_VIDEO_IDS = []
    for _ in VIDEOS_DIR.rglob("*.mp4"):
        vid = _.stem
        TARGET_VIDEO_IDS.append(vid)
    TARGET_VIDEO_IDS = sorted(TARGET_VIDEO_IDS)

    # 2. 遍历视频, 加载弹幕缓存和描述文件
    # 因为缓存好了，所以注释掉
    # for video_id in TARGET_VIDEO_IDS:
    #     print(f"\nProcessing Video: {video_id}")

    #     # time_info = metadata[video_id]
    #     # start_time_abs = parse_time_str(time_info)
    #     # total_duration = get_duration(time_info)

    #     # print(f"  Start Time (Abs): {start_time_abs}s")
    #     # print(f"  Total Duration: {total_duration}s")

    #     # 加载弹幕
    #     start_time_abs = 0
    #     total_duration = 10
    #     danmu_cache = get_video_danmu_cache(
    #         video_id, DIR_RAW_DANMU, DIR_DANMU_CACHE, start_time_abs, duration=total_duration
    #     )
    #     pass
    # del danmu_cache

    # 加载描述文件 (JSONL)
    desc_file_path = os.path.join(DIR_DESCRIPTIONS, "forrest_output/ViLAMP-llava-qwen-1_1.jsonl")

    descriptions = []
    with open(desc_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    descriptions.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"    Error parsing JSONL line: {e}")

    # 3. iterate over clip descriptions
    # add tqdm for progress bar
    from tqdm import tqdm

    for idx, item in enumerate(tqdm(descriptions)):
        video_path = item.get("video", "")
        try:
            # 提取文件名 -> 去后缀 -> 取最后一段
            slice_file_name = os.path.basename(video_path)
            slice_name = os.path.splitext(slice_file_name)[0]
        except Exception as e:
            continue
        video_desc = item.get("response", "")

        # 通过clip_name_stem找到弹幕pkl文件
        for __ in sorted(list(Path(DIR_DANMU_CACHE).glob("*.pkl"))):
            # _min_idx = __.stem.split('_')[0]
            # _max_idx = __.stem.split('_')[2]
            if slice_name.split('_')[0] in __.stem:
                danmu_cache_path = __
                break
            # if int(_min_idx) <= int(clip_name_stem.split('_')[-1]) <= int(_max_idx):
            # danmu_cache_path = __
            # break
        # 读取对应的弹幕缓存
        import pickle

        try:
            with open(danmu_cache_path, 'rb') as f:
                danmu_cache = pickle.load(f)
        except Exception as e:
            print(f"  Failed to load danmu cache from {danmu_cache_path}: {e}")
            danmu_cache = {}

        duration = 10
        video_name = slice_name.split('_')[0]
        if 'agzz' not in video_name:  # FIXME. 临时处理
            continue
        video_idx = int(slice_name.split('_')[3])
        slice_start_time = int(slice_name.split('_')[-3][:-1])
        slice_end_time = int(slice_name.split('_')[-1][:-1])

        video_offset = video_offset_dict.get(video_name)[video_idx]

        if video_offset == None:
            danmu_text = ""
        else:
            # from 0(duration=10) 取1-10秒的弹幕
            # from 936(duration=10) 取937-946秒的弹幕
            # from 936+100(duration=10) 取1037-1046秒的弹幕
            danmu_text = format_danmu_by_second(danmu_cache, slice_start_time + video_offset, duration)
        # FIXME. 弹幕太多，可以删掉一些

        # FIXME. 下面代码是筛选无弹幕的，已注释掉
        # pattern = r"第\d+秒: \(无弹幕\)"
        # matches = re.findall(pattern, text)
        # if len(matches) == 10 or len(matches) == 8:
        # do something

        MODE = 1
        if not danmu_text:
            MODE = 3  # 如果没有弹幕，切换到仅视频模式

        if video_name == 'agzz':
            for MODE in [2, 3]:
                existing_data = {}
                current_prompt = build_prompt(MODE, duration, video_desc, danmu_text)
                existing_data["prompt"] = current_prompt
                existing_data["duration"] = duration  # 记录时长方便解析

                output_path = Path(DIR_OUTPUT) / f"{slice_name}_mode_{MODE}.json"

                # 检查是否已存在结果文件
                if output_path.exists() and not FORCE_REBUILD_PROMPT:
                    print(f"  Output already exists at {output_path}, skipping...")
                    continue

                # === LLM 调用逻辑 ===
                if CALL_LLM:
                    # 检查是否已有 response (断点续传)
                    if not existing_data.get("response"):
                        try:
                            response = call_llm_api(current_prompt)
                            existing_data["response"] = response
                        except Exception as e:
                            print(f"    LLM Call Failed: {e}")
                    else:
                        pass  # 已有 response，跳过

                # === 保存结果 ===
                # 即使不调用 LLM，也保存 prompt
                with open(str(output_path), 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, ensure_ascii=False, indent=2)
        else:
            existing_data = {}
            current_prompt = build_prompt(MODE, duration, video_desc, danmu_text)
            existing_data["prompt"] = current_prompt
            existing_data["duration"] = duration  # 记录时长方便解析

            output_path = Path(DIR_OUTPUT) / f"{slice_name}_mode_{MODE}.json"

            # 检查是否已存在结果文件
            if output_path.exists() and not FORCE_REBUILD_PROMPT:
                print(f"  Output already exists at {output_path}, skipping...")
                continue

            # === LLM 调用逻辑 ===
            if CALL_LLM:
                # 检查是否已有 response (断点续传)
                if not existing_data.get("response"):
                    try:
                        response = call_llm_api(current_prompt)
                        existing_data["response"] = response
                    except Exception as e:
                        print(f"    LLM Call Failed: {e}")
                else:
                    pass  # 已有 response，跳过

            # === 保存结果 ===
            # 即使不调用 LLM，也保存 prompt
            with open(str(output_path), 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)

    print(f"Finished processing {video_idx}")


if __name__ == "__main__":
    main()
