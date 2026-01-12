import os
from operator import attrgetter
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IGNORE_INDEX,
)
from llava.conversation import conv_templates, SeparatorStyle

import torch
import cv2
import numpy as np
from PIL import Image
import requests
import copy
import warnings
from decord import VideoReader, cpu

from tqdm import tqdm
import pandas as pd
import json
import argparse
import time

from transformers import CLIPModel, CLIPProcessor, CLIPTextConfig
from tqdm import tqdm
from peft import PeftModel, PeftConfig

warnings.filterwarnings("ignore")
# Load the base model
pretrained = "models/{version}"
model_name = "llava_qwen"
device = "cuda:2"
device_map = None  # "auto"
tokenizer, model, image_processor, max_length = None, None, None, None

prompt = "{question}. Please choose the right answer from the following options: {options}."

clip_processor = CLIPProcessor.from_pretrained("clip-ViT-B-32/0_CLIPModel")


def access_video_by_sec(video_path, max_frame_num=3600):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps)
    frames = []
    frame_count = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            ret, frame = cap.retrieve()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame_rgb)
            frames.append(frame)
        frame_count += 1
    cap.release()
    if len(frames) > max_frame_num:
        sampled_indices = np.linspace(0, len(frames) - 1, max_frame_num, dtype=int)
        frames = [frames[i] for i in sampled_indices]
    return frames


def load_dataset(path):
    with open(path, 'r') as f:
        dataset = [json.loads(_) for _ in f]
    return dataset


def inference(PIL_images, query, question, device):
    video_frames = [np.array(img) for img in PIL_images]
    video_frames = np.array(video_frames)

    image_tensors = []
    frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().to(device)
    image_tensors.append(frames)

    image_inputs = clip_processor(images=PIL_images, return_tensors='pt').to(device)
    query_inputs = clip_processor(
        text=[question], return_tensors='pt', truncation=True, max_length=CLIPTextConfig().max_position_embeddings
    ).to(device)
    kwargs = {'image_inputs': [image_inputs], 'query_inputs': [query_inputs]}
    kwargs['offload_params'] = list(kwargs.keys())

    # Prepare conversation input
    conv_template = "qwen_1_5"
    query = "{DEFAULT_IMAGE_TOKEN}\n{query}".format(DEFAULT_IMAGE_TOKEN=DEFAULT_IMAGE_TOKEN, query=query)
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(device)
    )

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id  # 兜底：很多 Qwen/LLaMA 没有 pad，就用 eos

    attention_mask = (input_ids != pad_id).long()

    image_sizes = [frame.size for frame in video_frames]

    # Generate response
    cont = model.generate(
        input_ids,
        attention_mask=attention_mask,
        images=image_tensors,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=256,
        modalities=["video"],
        **kwargs,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    return text_outputs[0]


if __name__ == '__main__':

    # prompt = '''
    # You are an expert in affective computing and emotion understanding.

    # Watch the video carefully and describe the emotions expressed in the scene.

    # Focus on observable emotional cues, including but not limited to:
    # - facial expressions
    # - body language and posture
    # - movement and pacing
    # - scene atmosphere and tone
    # - lighting, color, and visual mood
    # - interactions between characters or with the environment

    # Describe the emotional states in a natural, detailed, and open-ended manner.
    # Emotions may be mixed, subtle, evolving, or ambiguous.

    # Do NOT assign numerical scores.
    # Do NOT restrict yourself to predefined emotion categories.

    # Output a single coherent paragraph describing the emotions conveyed by the video.
    # '''
    # import glob
    # videos = sorted(glob.glob("clips/**/*.mp4", recursive=True))
    # out_path = "icml/data.jsonl"
    # with open(out_path, "w", encoding="utf-8") as f:
    #     for v in videos:
    #         f.write(json.dumps({"video": os.path.abspath(v), "query": prompt},
    #         ensure_ascii=False) + "\n")
    # pass
    
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--dataset_path', default='icml/data.jsonl')
    arg_parser.add_argument('--output_dir', default='forrest_output')
    arg_parser.add_argument('--version', default='ViLAMP-llava-qwen')
    arg_parser.add_argument('--split', default='1_1')
    arg_parser.add_argument('--max_frame_num', type=int, default=600)

    args = arg_parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    dataset = load_dataset(args.dataset_path)

    pretrained = pretrained.format(version=args.version)
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained, None, model_name, device_map=device_map, attn_implementation=None
    )
    model.eval()
    device_ = torch.device(device)
    model.to(device_)

    args.output_file = os.path.join(args.output_dir, '{}-{}.jsonl'.format(args.version, args.split))
    print('LOADING DATASET...')

    cur_id, chunk_nums = args.split.split('_')
    cur_id = int(cur_id)
    chunk_nums = int(chunk_nums)
    dataset = [dataset[i] for i in range(0, len(dataset)) if i % chunk_nums == cur_id - 1]

    with open(args.output_file, 'w') as f:
        for data in tqdm(dataset):
            video_path = data['video']
            question = data['query']
            options = data['options'] if 'options' in data else None
            if options is not None:
                query = prompt.format(question=question, options=options)
            else:
                query = question
            # try:
            frames = access_video_by_sec(video_path, args.max_frame_num)
            response = inference(frames, query, question, device_)
            # except Exception as e:
            #     print(e)
            #     response = '[E]' + str(e)

            data['response'] = response

            f.write(json.dumps(data, ensure_ascii=False))
            f.write('\n')
