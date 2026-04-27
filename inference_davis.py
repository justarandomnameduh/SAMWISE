'''
Inference code for SAMWISE, on Ref-Youtube-VOS
Modified from DETR (https://github.com/facebookresearch/detr)
Ref-Davis17 does not support visualize
'''


import pandas as pd
from davis2017.evaluation import DAVISEvaluation
import sys
from models.samwise import build_samwise
from util.misc import on_load_checkpoint
import random
import time
from pathlib import Path
import numpy as np
import torch
from datasets.transform_utils import VideoEvalDataset
from torch.utils.data import DataLoader
from os.path import join
import util.misc as utils
import os
from PIL import Image
import torch.nn.functional as F
import json
import argparse
from tqdm import tqdm
from tools.colormap import colormap
import opts
np.bool = np.bool_

# colormap
color_list = colormap()
color_list = color_list.astype('uint8').tolist()
OVERLAY_FPS = 10


def blend_mask(image_rgb, mask, color):
    output = image_rgb.copy()
    mask = mask.astype(bool)
    if mask.any():
        output[mask] = output[mask] * 0.25 + np.array(color) * 0.75
    return output.astype(np.uint8)


def write_video(frames_rgb, output_path, fps=OVERLAY_FPS):
    import cv2

    if not frames_rgb:
        raise ValueError(f"No frames to write for {output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    height, width = frames_rgb[0].shape[:2]
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    try:
        for frame_rgb in frames_rgb:
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def write_manifest(path, entries):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2)


def load_palette(davis_path):
    davis_root = Path(davis_path) / "DAVIS"
    for pattern in ("Annotations_unsupervised/480p/*/*.png", "Annotations/480p/*/*.png"):
        for candidate in davis_root.glob(pattern):
            return Image.open(candidate).getpalette()
    return None


def read_expression_mask(results_path, video, exp_id, frame_name):
    mask_path = Path(results_path) / video / str(exp_id) / f"{frame_name}.png"
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing DAVIS expression mask: {mask_path}")
    return np.array(Image.open(mask_path).convert("L")) > 0


def reconstruct_davis_annotator_results(results_path, data, output_path, davis_path, num_anno=4):
    palette = load_palette(davis_path)
    output_path = Path(output_path)
    for anno_id in range(num_anno):
        anno_root = output_path / f"anno_{anno_id}"
        anno_root.mkdir(parents=True, exist_ok=True)
        for video, video_meta in data.items():
            expressions = video_meta["expressions"]
            expression_ids = list(expressions.keys())
            frames = video_meta["frames"]
            num_obj = len(expression_ids) // num_anno
            video_root = anno_root / video
            video_root.mkdir(parents=True, exist_ok=True)
            for frame_name in frames:
                composite = None
                for obj_id in range(num_obj):
                    exp_id = expression_ids[obj_id * num_anno + anno_id]
                    mask = read_expression_mask(results_path, video, exp_id, frame_name)
                    if composite is None:
                        composite = np.zeros(mask.shape, dtype=np.uint8)
                    composite[mask] = obj_id + 1
                if composite is None:
                    raise RuntimeError(f"No expressions found for DAVIS video: {video}")
                image = Image.fromarray(composite)
                if palette:
                    image.putpalette(palette)
                image.save(video_root / f"{frame_name}.png")
    return [output_path / f"anno_{idx}" for idx in range(num_anno)]


def main(args):
    print("Inference only supports for batch size = 1")
    print(args)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    utils.init_distributed_mode(args)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # save path
    output_dir = args.output_dir
    args.log_file = join(args.output_dir, 'log.txt')
    with open(args.log_file, 'w') as fp:
        fp.writelines(" ".join(sys.argv) + '\n')
        fp.writelines(str(args.__dict__) + '\n\n')

    start_time = time.time()
    # model
    model = build_samwise(args)
    device = torch.device(args.device)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('number of params:', n_parameters)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if list(checkpoint['model'].keys())[0].startswith('module'):
            checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        checkpoint = on_load_checkpoint(model_without_ddp, checkpoint)
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))

    print('Start inference')
    eval_davis(args, model, output_dir)

    end_time = time.time()
    total_time = end_time - start_time

    print("Total inference time: %.4f s" % (total_time))



def eval_davis(args, model, save_path_prefix):
    print("Inference only supports for batch size = 1")

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # save path
    eval_output_path = os.path.join(save_path_prefix, "eval_davis", args.split)
    save_path_prefix = os.path.join(save_path_prefix, "Annotations")
    os.makedirs(save_path_prefix, exist_ok=True)
    overlay_root = os.path.join(args.output_dir, "overlay_videos")

    # load data
    root = Path(args.davis_path)  # data/ref-davis
    img_folder = os.path.join(root, args.split, "JPEGImages")
    meta_file = os.path.join(root, "meta_expressions", args.split, "meta_expressions.json")
    with open(meta_file, "r") as f:
        data = json.load(f)["videos"]
    video_list = list(data.keys())

    start_time = time.time()
    print('Start inference')
    sub_video_list = video_list

    sub_processor(args, model, data, save_path_prefix, overlay_root, img_folder, sub_video_list)

    args.results_path = save_path_prefix
    args.eval_output_path = eval_output_path
    eval_davis_compute_metrics(args, data)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"Total inference time: {total_time:.2f} s")


def sub_processor(args, model, data, save_path_prefix, overlay_root, img_folder, video_list):
    progress = tqdm(
            total=len(video_list),
            ncols=0
        )

    # start inference
    model.eval()

    # 1. for each video
    for video in video_list:
        metas = []

        expressions = data[video]["expressions"]
        expression_list = list(expressions.keys())
        num_expressions = len(expression_list)

        # read all the anno meta
        for i in range(num_expressions):
            meta = {}
            meta["video"] = video
            meta["exp"] = expressions[expression_list[i]]["exp"]
            meta["exp_id"] = expression_list[i]  # start from 0
            meta["frames"] = data[video]["frames"]
            metas.append(meta)
        meta = metas
        annotation_manifest = [[os.path.join(item["exp_id"]), item["exp"]] for item in meta]
        overlay_manifest = [[f"exp_{item['exp_id']}.mp4", item["exp"]] for item in meta]
        write_manifest(os.path.join(save_path_prefix, video, "manifest.json"), annotation_manifest)
        write_manifest(os.path.join(overlay_root, video, "manifest.json"), overlay_manifest)

        # since there are 4 annotations
        num_obj = num_expressions // 4

        # 2. for each annotator
        for anno_id in range(4):  # 4 annotators
            for obj_id in range(num_obj):
                i = obj_id * 4 + anno_id
                video_name = meta[i]["video"]
                exp = meta[i]["exp"]
                exp_id = meta[i]["exp_id"]
                frames = meta[i]["frames"]

                all_pred_masks = []

                vd = VideoEvalDataset(join(img_folder, video_name), frames, max_size=args.max_size)
                dl = DataLoader(vd, batch_size=args.eval_clip_window,
                                num_workers=args.num_workers, shuffle=False)
                origin_w, origin_h = vd.origin_w, vd.origin_h
                # 3. for each clip
                for imgs, clip_frames_ids in dl:
                    clip_frames_ids = clip_frames_ids.tolist()
                    img_h, img_w = imgs.shape[-2:]
                    imgs = imgs.to(args.device)
                    size = torch.as_tensor([int(img_h), int(img_w)]).to(args.device)
                    target = {"size": size, 'frame_ids': clip_frames_ids}
                    with torch.no_grad():
                        outputs = model([imgs], [exp], [target])

                    pred_masks = outputs["pred_masks"]  # [t, q, h, w]
                    pred_masks = pred_masks.unsqueeze(0)

                    pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear',
                                               align_corners=False)
                    pred_masks = pred_masks.sigmoid()[0]  # [t, h, w], NOTE: here mask is score
                    all_pred_masks.append(pred_masks)

                all_pred_masks = torch.cat(all_pred_masks, dim=0)  # (video_len, h, w)
                binary_pred_masks = (all_pred_masks > args.threshold).detach().cpu().numpy().astype(np.uint8)
                exp_save_path = os.path.join(save_path_prefix, video_name, exp_id)
                os.makedirs(exp_save_path, exist_ok=True)
                if utils.is_main_process():
                    for frame_index, frame_name in enumerate(frames):
                        mask = Image.fromarray(binary_pred_masks[frame_index] * 255).convert("L")
                        mask.save(os.path.join(exp_save_path, frame_name + ".png"))

                rendered_frames = []
                color = color_list[i % len(color_list)]
                for frame_index, frame_name in enumerate(frames):
                    image_path = os.path.join(img_folder, video_name, frame_name + ".jpg")
                    image_rgb = np.asarray(Image.open(image_path).convert("RGB"))
                    rendered_frames.append(blend_mask(image_rgb, binary_pred_masks[frame_index], color))
                if utils.is_main_process():
                    write_video(
                        rendered_frames,
                        os.path.join(overlay_root, video_name, f"exp_{exp_id}.mp4"),
                    )

            torch.cuda.empty_cache()
            import gc
            gc.collect()
        progress.update(1)

def evaluate_davis_annotator(args, anno_results_path, csv_name_global, csv_name_per_sequence):
    time_start = time.time()
    csv_name_global = f'global_results-{args.set}.csv'
    csv_name_per_sequence = f'per-sequence_results-{args.set}.csv'
    args.log_file = join(args.output_dir, 'log.txt')
    print(f'using {anno_results_path}')
    # Check if the method has been evaluated before, if so read the results, otherwise compute the results
    csv_name_global_path = os.path.join(anno_results_path, csv_name_global)
    csv_name_per_sequence_path = os.path.join(anno_results_path, csv_name_per_sequence)
    if os.path.exists(csv_name_global_path) and os.path.exists(csv_name_per_sequence_path):
        print('Using precomputed results...')
        table_g = pd.read_csv(csv_name_global_path)
        table_seq = pd.read_csv(csv_name_per_sequence_path)
    else:
        print(f'Evaluating sequences for the {args.task} task...')
        # Create dataset and evaluate
        dataset_eval = DAVISEvaluation(davis_root=args.davis_path + "/DAVIS", task=args.task, gt_set=args.set)
        metrics_res = dataset_eval.evaluate(str(anno_results_path))
        J, F = metrics_res['J'], metrics_res['F']

        # Generate dataframe for the general results
        g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
        final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
        g_res = np.array(
            [final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]),
             np.mean(F["D"])])
        g_res = np.reshape(g_res, [1, len(g_res)])
        table_g = pd.DataFrame(data=g_res, columns=g_measures)
        with open(csv_name_global_path, 'w') as f:
            table_g.to_csv(f, index=False, float_format="%.5f")
        print(f'Global results saved in {csv_name_global_path}')

        # Generate a dataframe for the per sequence results
        seq_names = list(J['M_per_object'].keys())
        seq_measures = ['Sequence', 'J-Mean', 'F-Mean']
        J_per_object = [J['M_per_object'][x] for x in seq_names]
        F_per_object = [F['M_per_object'][x] for x in seq_names]
        table_seq = pd.DataFrame(data=list(zip(seq_names, J_per_object, F_per_object)), columns=seq_measures)
        if utils.is_main_process():
            with open(csv_name_per_sequence_path, 'w') as f:
                table_seq.to_csv(f, index=False, float_format="%.5f")
        print(f'Per-sequence results saved in {csv_name_per_sequence_path}')

    # Print the results
    sys.stdout.write(f"--------------------------- Global results for {args.set} ---------------------------\n")
    print(table_g.to_string(index=False))
    sys.stdout.write(f"\n---------- Per sequence results for {args.set} ----------\n")
    print(table_seq.to_string(index=False))
    total_time = time.time() - time_start
    sys.stdout.write('\nTotal time:' + str(total_time))
    if utils.get_rank() == 0:
        with open(args.log_file, 'a') as fp:
            fp.write(table_seq.to_string(index=False) + '\n')
            fp.write(table_g.to_string(index=False) + '\n\n')
    return table_g


def eval_davis_compute_metrics(args, data):
    eval_output_path = getattr(args, "eval_output_path", os.path.join(args.output_dir, "eval_davis", args.split))
    print(f"Reconstructing DAVIS annotator composites from {args.results_path}")
    print(f"Saving DAVIS evaluation artifacts to: {eval_output_path}")
    anno_result_paths = reconstruct_davis_annotator_results(
        args.results_path,
        data,
        eval_output_path,
        args.davis_path,
        num_anno=4,
    )

    csv_name_global = f'global_results-{args.set}.csv'
    csv_name_per_sequence = f'per-sequence_results-{args.set}.csv'
    all_results = []
    for anno_results_path in anno_result_paths:
        table_g = evaluate_davis_annotator(args, anno_results_path, csv_name_global, csv_name_per_sequence)
        all_results.append(table_g)
        print("\n")

    all_results = pd.concat(all_results, axis=0, ignore_index=True)
    all_results.index = [f'an{i}' for i in range(4)]
    avg_results = pd.DataFrame([all_results.mean()], index=['avg'])
    all_results = pd.concat([all_results, avg_results], axis=0, ignore_index=False)
    print(all_results.to_string())


if __name__ == '__main__':
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser('SAMWISE evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    name_exp = args.name_exp
    args.output_dir = os.path.join(args.output_dir, name_exp)


    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
