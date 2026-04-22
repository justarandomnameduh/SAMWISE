#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from tools.colormap import colormap


COLOR_LIST = colormap(rgb=True).astype("uint8")


def blend_mask(image_rgb: np.ndarray, mask: np.ndarray, color: np.ndarray) -> np.ndarray:
    output = image_rgb.copy()
    mask = mask.astype(bool)
    if mask.any():
        output[mask] = output[mask] * 0.25 + color * 0.75
    return output.astype(np.uint8)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_video(frames_rgb: list[np.ndarray], output_path: Path, fps: int) -> None:
    import cv2

    if not frames_rgb:
        raise ValueError(f"No frames to write for {output_path}")

    height, width = frames_rgb[0].shape[:2]
    ensure_parent(output_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    try:
        for frame_rgb in frames_rgb:
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def load_rgb(image_path: Path) -> np.ndarray:
    return np.asarray(Image.open(image_path).convert("RGB"))


def load_binary_mask(mask_path: Path) -> np.ndarray:
    return np.asarray(Image.open(mask_path).convert("L")) > 127


def export_mevis(args: argparse.Namespace) -> None:
    image_root = Path(args.mevis_path) / args.mevis_split / "JPEGImages"
    meta_path = Path(args.mevis_path) / args.mevis_split / "meta_expressions.json"
    pred_root = Path(args.output_root) / "mevis" / args.version / "Annotations"
    overlay_root = Path(args.output_root) / "mevis" / args.version / "overlay_videos"

    with open(meta_path, "r", encoding="utf-8") as handle:
        videos = json.load(handle)["videos"]

    total_exports = 0
    for video_name, video_dict in videos.items():
        frames = video_dict["frames"]
        expression_ids = list(video_dict["expressions"].keys())

        for expr_index, expr_id in enumerate(expression_ids):
            mask_dir = pred_root / video_name / expr_id
            if not mask_dir.is_dir():
                print(f"[overlay] Skipping MeViS {video_name}/{expr_id}: missing {mask_dir}")
                continue

            color = COLOR_LIST[expr_index % len(COLOR_LIST)]
            rendered_frames = []
            missing_frame = False
            for frame_name in frames:
                image_path = image_root / video_name / f"{frame_name}.jpg"
                mask_path = mask_dir / f"{frame_name}.png"
                if not image_path.is_file() or not mask_path.is_file():
                    print(f"[overlay] Skipping MeViS {video_name}/{expr_id}: missing frame or mask for {frame_name}")
                    missing_frame = True
                    break

                image_rgb = load_rgb(image_path)
                mask = load_binary_mask(mask_path)
                rendered_frames.append(blend_mask(image_rgb, mask, color))

            if missing_frame:
                continue

            write_video(rendered_frames, overlay_root / video_name / f"{expr_id}.mp4", args.fps)
            total_exports += 1

    print(f"[overlay] Exported {total_exports} MeViS videos to {overlay_root}")


def export_davis(args: argparse.Namespace) -> None:
    image_root = Path(args.davis_path) / args.davis_split / "JPEGImages"
    meta_path = Path(args.davis_path) / "meta_expressions" / args.davis_split / "meta_expressions.json"
    pred_root = Path(args.output_root) / "davis" / args.version / "eval_davis" / args.davis_split
    overlay_root = Path(args.output_root) / "davis" / args.version / "overlay_videos"

    with open(meta_path, "r", encoding="utf-8") as handle:
        videos = json.load(handle)["videos"]

    total_exports = 0
    for annotator in range(4):
        anno_dir = pred_root / f"anno_{annotator}"
        for video_name, video_dict in videos.items():
            frames = video_dict["frames"]
            mask_video_dir = anno_dir / video_name
            if not mask_video_dir.is_dir():
                print(f"[overlay] Skipping DAVIS anno_{annotator}/{video_name}: missing {mask_video_dir}")
                continue

            rendered_frames = []
            missing_frame = False
            for frame_index, frame_name in enumerate(frames):
                image_path = image_root / video_name / f"{frame_name}.jpg"
                mask_path = mask_video_dir / f"{frame_index:05d}.png"
                if not image_path.is_file() or not mask_path.is_file():
                    print(
                        f"[overlay] Skipping DAVIS anno_{annotator}/{video_name}: "
                        f"missing frame or mask for {frame_name}"
                    )
                    missing_frame = True
                    break

                image_rgb = load_rgb(image_path)
                mask = np.asarray(Image.open(mask_path))
                frame_rgb = image_rgb.copy()
                object_ids = [object_id for object_id in np.unique(mask) if object_id != 0]
                for object_id in object_ids:
                    color = COLOR_LIST[(int(object_id) - 1) % len(COLOR_LIST)]
                    frame_rgb = blend_mask(frame_rgb, mask == object_id, color)
                rendered_frames.append(frame_rgb)

            if missing_frame:
                continue

            write_video(rendered_frames, overlay_root / f"anno_{annotator}" / f"{video_name}.mp4", args.fps)
            total_exports += 1

    print(f"[overlay] Exported {total_exports} DAVIS videos to {overlay_root}")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Export overlay MP4 videos from completed SAMWISE DAVIS and MeViS inference outputs."
    )
    parser.add_argument("--version", required=True, help="Experiment version under output/{davis,mevis}/")
    parser.add_argument(
        "--output-root",
        default=str(repo_root / "output"),
        help="Root output directory containing davis/ and mevis/ subdirectories",
    )
    parser.add_argument(
        "--datasets",
        default="all",
        choices=["all", "davis", "mevis"],
        help="Which dataset overlays to export",
    )
    parser.add_argument("--fps", type=int, default=10, help="Output video FPS")
    parser.add_argument(
        "--davis-path",
        default=str(repo_root / "data" / "ref-davis"),
        help="Prepared Ref-DAVIS dataset root",
    )
    parser.add_argument(
        "--mevis-path",
        default=str(repo_root / "data" / "MeViS_release"),
        help="Prepared MeViS dataset root",
    )
    parser.add_argument("--davis-split", default="valid", choices=["valid"], help="DAVIS split to render")
    parser.add_argument(
        "--mevis-split",
        default="valid_u",
        choices=["valid", "valid_u", "test"],
        help="MeViS split to render",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.datasets in ("all", "davis"):
        export_davis(args)
    if args.datasets in ("all", "mevis"):
        export_mevis(args)


if __name__ == "__main__":
    main()
