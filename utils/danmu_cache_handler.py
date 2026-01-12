import os
import pandas as pd
import pickle
import numpy as np
import chardet

def convert_to_seconds(time_str):
    """转换时间字符串为秒数"""
    if pd.isna(time_str):
        return np.nan
    time_str = str(time_str)
    # 如果是分:秒格式（没有两个冒号）
    if time_str.count(':') == 1:
        time_str = '00:' + time_str  # 补充小时部分
    try:
        return pd.to_timedelta(time_str).total_seconds()  # 转换为秒数
    except Exception:
        return np.nan  # 遇到无法解析的值返回NaN

def get_video_danmu_cache(video_id, raw_danmu_dir, cache_dir, video_start_time_abs, duration=None):
    """
    获取视频的秒级弹幕缓存。
    如果缓存存在，直接读取；否则从CSV读取并生成缓存。
    
    Args:
        video_id: 视频ID (e.g., 'neg_s_1')
        raw_danmu_dir: 原始弹幕CSV所在目录
        cache_dir: 缓存保存目录
        video_start_time_abs: 视频在电影中的绝对起始时间(秒)
        duration: 视频总时长(秒)，用于过滤超出范围的弹幕
        
    Returns:
        dict: {second_idx: [danmu_text1, danmu_text2, ...]}
        second_idx 是相对于视频开始的秒数 (0, 1, 2...)
    """

    # 根据video_id判断应该读取哪个弹幕文件
    from pathlib import Path
    danmu_csv_name = None
    for _ in sorted(list(Path(raw_danmu_dir).glob("*.csv"))):
        danmu_file_name = _.stem
        # start = danmu_file_name.split('_')[0]
        # end = danmu_file_name.split('_')[-1]
        
        # 判断video_id是否在范围内
        # if int(start) <= int(video_id.split('_')[0]) <= int(end):
            # danmu_csv_name = danmu_file_name
            # break
        # 根据名字判断要找的danmu_csv_name
        if danmu_file_name in video_id:
            danmu_csv_name = danmu_file_name
            break
    
    cache_path = os.path.join(cache_dir, f"{danmu_csv_name}_second_level_cache.pkl")
    
    # 1. 尝试读取缓存
    if os.path.exists(cache_path):
        print(f"Loading danmu cache for {video_id}...")
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                return data
        except Exception as e:
            print(f"Error loading cache, rebuilding: {e}")
    
    # 2. 缓存不存在或读取失败，重新构建
    csv_path = os.path.join(raw_danmu_dir, f"{danmu_csv_name}.csv")
    if not os.path.exists(csv_path):
        return {}
        
    print(f"Building danmu cache for {danmu_csv_name}...")
    
    # 读取CSV (复用原有的读取逻辑，处理编码)
    try:
        with open(csv_path, 'rb') as f:
            encoding_type = chardet.detect(f.read())['encoding']
            if not encoding_type:
                encoding_type = 'UTF-8'
        try:
            df = pd.read_csv(csv_path, encoding='gbk')
        except:
            df = pd.read_csv(csv_path, encoding=encoding_type)
    except Exception as e:
        print(f"Failed to read CSV {csv_path}: {e}")
        return {}

    movie_time_name = 'movie_time'
    danmu_name = 'danmu'
    
    # 清洗和转换时间
    try:
        df[movie_time_name] = pd.to_numeric(df[movie_time_name], errors='coerce')
    except:
        pass
        
    # 如果转换后大部分是NaN，可能原本是时分秒格式
    if df[movie_time_name].isna().sum() > len(df) * 0.5:
         # 重新读取原始数据进行apply转换
        try: #重新读取防止上面操作影响
             with open(csv_path, 'rb') as f:
                encoding_type = chardet.detect(f.read())['encoding']
             try:
                df = pd.read_csv(csv_path, encoding='gbk')
             except:
                df = pd.read_csv(csv_path, encoding=encoding_type)
             df[movie_time_name] = df[movie_time_name].apply(convert_to_seconds)
        except:
             pass

    df = df.dropna(subset=[movie_time_name])
    df[movie_time_name] = df[movie_time_name].astype(float)
    # 读取danmu.csv中的Duration
    with open('danmu.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if line.split(',')[0] == danmu_csv_name:
                original_duration = float(line.split(',')[-1].strip())
                break

    # rescale from 0-original_duration to 0-18*60
    # if original_duration > 0:
    #     df[movie_time_name] = df[movie_time_name] * (18*60) / original_duration

    # 将movie_time_name列转为int，向下取整秒数
    df[movie_time_name] = np.floor(df[movie_time_name]).astype(int) + 1
    # Group by movie_time_name
    cache_dict = df.groupby(movie_time_name)[danmu_name].apply(list).to_dict()

    # 保存缓存
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_dict, f)
        print(f"Cache saved to {cache_path}")
    except Exception as e:
        print(f"Failed to save cache: {e}")
        
    return cache_dict

def format_danmu_by_second(danmu_cache, start_sec, duration):
    """
    将弹幕格式化为字符串，按秒罗列。
    
    Args:
        danmu_cache: 弹幕缓存字典 {sec: [texts]}
        start_sec: 片段相对于视频开始的秒数 (e.g. 0, 10, 20)
        duration: 片段时长 (通常10，最后一段可能更短)
        
    Returns:
        str: 格式化后的弹幕文本
    """
    lines = []
    for i in range(1, duration+1):
        current_sec = start_sec + i
        texts = danmu_cache.get(current_sec, [])
        
        # 清洗文本：去除换行符，转为字符串
        clean_texts = [str(t).replace('\n', ' ').strip() for t in texts]
        clean_texts = [t for t in clean_texts if t] # 去空
        
        line_prefix = f"第{i}秒:"
        if clean_texts:
            content = "; ".join(clean_texts)
            lines.append(f"{line_prefix} {content}")
        else:
            lines.append(f"{line_prefix} (无弹幕)")
            
    return "\n".join(lines)
