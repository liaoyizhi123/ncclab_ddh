# !/usr/bin/env python3
# -*- coding: utf-8 -*-


import cv2
from pathlib import Path

MOVIE_PATH = f'/mnt/dataset0/sitian/code/movie/周处除三害'
OUTPUT_PATH = f'/home/liaoyizhi/codes/ncclab_ddh/clips/周处除三害'


def main():
    for movie_segment_file in sorted(list(Path(MOVIE_PATH).glob('*.mp4'))):
        if movie_segment_file.name.startswith('.') or 'final_' not in movie_segment_file.name:
            continue

        segment_name = movie_segment_file.stem  # 等价于 split('.')[0]

        # 每个源视频建立一个独立文件夹
        clip_dir = Path(OUTPUT_PATH)
        clip_dir.mkdir(parents=True, exist_ok=True)

        # read mp4
        cap = cv2.VideoCapture(str(movie_segment_file))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {movie_segment_file}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            cap.release()
            raise RuntimeError(f"Invalid FPS ({fps}) for: {movie_segment_file}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames_per_clip = int(round(10 * fps))  # 10秒
        if frames_per_clip <= 0:
            cap.release()
            raise RuntimeError("frames_per_clip <= 0, check fps or segment length setting.")

        # 最后一段不足10秒就舍弃 -> 只输出完整段数
        full_clips = total_frames // frames_per_clip
        if full_clips <= 0:
            print(f"[SKIP] {movie_segment_file.name}: total_frames={total_frames} < frames_per_clip={frames_per_clip}")
            cap.release()
            continue

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        print(f"[INFO] Processing: {movie_segment_file.name}")
        print(
            f"       fps={fps:.3f}, size=({width},{height}), total_frames={total_frames}, "
            f"frames_per_clip={frames_per_clip}, full_clips={full_clips}"
        )

        # 逐段输出
        for clip_idx in range(full_clips):
            start_sec = clip_idx * 10
            end_sec = start_sec + 10
            out_path = clip_dir / f"{segment_name}_{start_sec}s_to_{end_sec}s.mp4"

            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                cap.release()
                raise RuntimeError(f"Cannot create VideoWriter: {out_path}")

            # 写入固定 frames_per_clip 帧
            for _ in range(frames_per_clip):
                ret, frame = cap.read()
                if not ret:
                    # 理论上不会发生（因为 full_clips 是按 total_frames 算的），保险起见处理
                    writer.release()
                    cap.release()
                    print(f"[WARN] Early EOF when writing {out_path}, discard this clip.")
                    # 删除不完整文件（可选）
                    try:
                        out_path.unlink(missing_ok=True)
                    except TypeError:
                        if out_path.exists():
                            out_path.unlink()
                    return
                writer.write(frame)

            writer.release()

        cap.release()
        print(f"[DONE] {movie_segment_file.name} -> {full_clips} clips saved in: {clip_dir}\n")


if __name__ == '__main__':
    main()
