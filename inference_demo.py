"""
Inference code for SAMWISE, on Ref-Youtube-VOS
Modified from DETR (https://github.com/facebookresearch/detr)
"""

import argparse
import os
import random
import sys
import time
from os.path import join
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import opts
import util.misc as utils
from datasets.transform_utils import VideoEvalDataset, vis_add_mask
from models.samwise import build_samwise
from tools.colormap import colormap
from util.misc import on_load_checkpoint


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


color_list = colormap()
color_list = color_list.astype("uint8").tolist()


def ensure_dir(path: str | None) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def main(args):
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    save_path_prefix = args.output_root
    os.makedirs(save_path_prefix, exist_ok=True)

    start_time = time.time()
    model = build_samwise(args)
    device = torch.device(args.device)
    model.to(device)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if list(checkpoint["model"].keys())[0].startswith("module"):
            checkpoint["model"] = {key.replace("module.", ""): value for key, value in checkpoint["model"].items()}
        checkpoint = on_load_checkpoint(model, checkpoint)
        model.load_state_dict(checkpoint["model"], strict=False)

    print("Start inference")
    inference(args, model, save_path_prefix, args.input_path, args.text_prompts)

    total_time = time.time() - start_time
    print("Total inference time: %.4f s" % total_time)


def extract_frames_from_mp4(video_path):
    extract_folder = "frames_" + os.path.basename(video_path).split(".")[0]
    print(f"Extracting frames from .mp4 in {extract_folder} with ffmpeg...")
    if os.path.isdir(extract_folder):
        print(f"{extract_folder} already exists, using frames in that folder")
    else:
        os.makedirs(extract_folder)
        extract_cmd = "ffmpeg -i {in_path} -loglevel error -vf fps=10 {folder}/frame_%05d.png"
        ret = os.system(extract_cmd.format(in_path=video_path, folder=extract_folder))
        if ret != 0:
            print("Something went wrong extracting frames with ffmpeg")
            sys.exit(ret)
    frames_list = os.listdir(extract_folder)
    frames_list = sorted([os.path.splitext(frame)[0] for frame in frames_list])
    return extract_folder, frames_list, ".png"


def collect_directory_frames(frames_folder: str) -> tuple[list[str], str]:
    frame_paths = sorted(
        path
        for path in Path(frames_folder).iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not frame_paths:
        raise FileNotFoundError(f"No image frames found in directory: {frames_folder}")
    ext = frame_paths[0].suffix
    frames_list = [path.stem for path in frame_paths if path.suffix == ext]
    if not frames_list:
        raise FileNotFoundError(f"No frames with extension {ext} found in directory: {frames_folder}")
    return frames_list, ext


def compute_masks(args, model, text_prompt, frames_folder, frames_list, ext):
    all_pred_masks = []
    dataset = VideoEvalDataset(frames_folder, frames_list, ext=ext)
    dataloader = DataLoader(dataset, batch_size=args.eval_clip_window, num_workers=args.num_workers, shuffle=False)
    origin_w, origin_h = dataset.origin_w, dataset.origin_h

    for imgs, clip_frames_ids in tqdm(dataloader):
        clip_frames_ids = clip_frames_ids.tolist()
        imgs = imgs.to(args.device)
        img_h, img_w = imgs.shape[-2:]
        size = torch.as_tensor([int(img_h), int(img_w)]).to(args.device)
        target = {"size": size, "frame_ids": clip_frames_ids}

        with torch.no_grad():
            outputs = model([imgs], [text_prompt], [target])

        pred_masks = outputs["pred_masks"]
        pred_masks = pred_masks.unsqueeze(0)
        pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode="bilinear", align_corners=False)
        pred_masks = (pred_masks.sigmoid() > args.threshold)[0].cpu()
        all_pred_masks.append(pred_masks)

    return torch.cat(all_pred_masks, dim=0).numpy()


def save_mask(mask: np.ndarray, path: str) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255).convert("L").save(path)


def inference(args, model, save_path_prefix, in_path, text_prompts):
    if os.path.isfile(in_path) and not args.image_level:
        frames_folder, frames_list, ext = extract_frames_from_mp4(in_path)
    elif os.path.isfile(in_path) and args.image_level:
        fname, ext = os.path.splitext(in_path)
        frames_list = [os.path.basename(fname)]
        frames_folder = os.path.dirname(in_path)
    else:
        frames_folder = in_path
        frames_list, ext = collect_directory_frames(frames_folder)

    model.eval()
    print(f"Begin inference on {len(frames_list)} frames")

    for index, text_prompt in enumerate(text_prompts):
        all_pred_masks = compute_masks(args, model, text_prompt, frames_folder, frames_list, ext)

        text_slug = text_prompt.replace(" ", "_")
        save_visualize_path_dir = args.overlay_frames_dir or join(save_path_prefix, text_slug)
        pred_masks_dir = args.pred_masks_dir or join(save_path_prefix, text_slug, "pred_masks")
        ensure_dir(save_visualize_path_dir)
        ensure_dir(pred_masks_dir)
        print(f"Saving output to disk in {save_visualize_path_dir}")

        out_files_w_mask = []
        for frame_index, frame_name in enumerate(frames_list):
            img_path = join(frames_folder, frame_name + ext)
            source_img = Image.open(img_path).convert("RGBA")
            source_img = vis_add_mask(source_img, all_pred_masks[frame_index], color_list[index % len(color_list)])

            overlay_path = join(save_visualize_path_dir, frame_name + ".png")
            source_img.save(overlay_path)
            out_files_w_mask.append(overlay_path)

            mask_path = join(pred_masks_dir, frame_name + ".png")
            save_mask(all_pred_masks[frame_index], mask_path)

        if not args.image_level and not args.skip_video_write:
            from moviepy import ImageSequenceClip

            output_video_path = args.output_video_path or join(save_path_prefix, text_slug + ".mp4")
            clip = ImageSequenceClip(out_files_w_mask, fps=args.video_fps)
            clip.write_videofile(output_video_path, codec="libx264")

    print(f"Output masks and videos can be found in {save_path_prefix}")


def check_args(args):
    assert os.path.isfile(args.input_path) or os.path.isdir(
        args.input_path
    ), f"The provided path {args.input_path} does not exist"
    args.image_level = False
    if os.path.isfile(args.input_path):
        ext = os.path.splitext(args.input_path)[1]
        assert ext in [".jpg", ".png", ".mp4", ".jpeg"], "Provided file extension should be one of ['.jpg', '.png', '.mp4']"
        if ext in [".jpg", ".png", ".jpeg"]:
            args.image_level = True
            pretrained_model = "pretrain/pretrained_model.pth"
            pretrained_model_link = "https://drive.google.com/file/d/1gRGzARDjIisZ3PnCW77Y9TMM_SbV8aaa/view?usp=drive_link"
            print("Specified path is an image, using image-level configuration")

    if not args.image_level:
        args.HSA = True
        args.use_cme_head = False
        pretrained_model = "pretrain/final_model_mevis.pth"
        pretrained_model_link = "https://drive.google.com/file/d/1Molt2up2bP41ekeczXWQU-LWTskKJOV2/view?usp=sharing"
        print("Specified path is a video or folder with frames, using video-level configuration")

    if args.resume == "":
        args.resume = pretrained_model

    assert os.path.isfile(
        args.resume
    ), f"You should download the model checkpoint first. Run 'cd pretrain &&  gdown --fuzzy {pretrained_model_link}"


if __name__ == "__main__":
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser("SAMWISE evaluation script", parents=[opts.get_args_parser()])
    parser.add_argument("--input_path", default=None, type=str, required=True, help="path to mp4 video or frames folder")
    parser.add_argument(
        "--text_prompts",
        default=[""],
        type=str,
        required=True,
        nargs="+",
        help="List of referring expressions, separated by whitespace",
    )
    parser.add_argument("--output_root", default="demo_output", type=str, help="Root directory for demo outputs")
    parser.add_argument("--overlay_frames_dir", default=None, type=str, help="Directory to save overlay frames")
    parser.add_argument("--pred_masks_dir", default=None, type=str, help="Directory to save raw predicted masks")
    parser.add_argument("--output_video_path", default=None, type=str, help="Override output video path")
    parser.add_argument("--skip_video_write", action="store_true", help="Skip writing the demo mp4")
    parser.add_argument("--video_fps", default=10.0, type=float, help="FPS used when writing the demo mp4")

    args = parser.parse_args()
    check_args(args)
    main(args)
