from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
from PIL import Image

try:
    from pycocotools import mask as cocomask
except ImportError:  # pragma: no cover - runtime dependent
    cocomask = None


GROUP_ORDER = [
    "DAVIS anno_0",
    "DAVIS anno_1",
    "DAVIS anno_2",
    "DAVIS anno_3",
    "MeViS",
]

PALETTE = {
    "DAVIS anno_0": "#1f77b4",
    "DAVIS anno_1": "#ff7f0e",
    "DAVIS anno_2": "#2ca02c",
    "DAVIS anno_3": "#d62728",
    "MeViS": "#9467bd",
}

MEVIS_VIDEO_LOG_PATTERN = re.compile(
    r"^(?P<index>\d+)/(?P<total>\d+)\s+CURRENT J&F:\s+(?P<jf>\S+)\s+J:\s+(?P<j>\S+)\s+F:\s+(?P<f>\S+)$"
)


@dataclass
class AnalysisContext:
    repo_root: Path
    output_root: Path
    version: str
    mevis_split: str
    save_root: Path
    davis_output_root: Path
    mevis_output_root: Path
    mevis_pred_root: Path
    sibling_dataset_root: Path
    davis_rgb_root: Optional[Path]
    davis_gt_root: Optional[Path]
    mevis_rgb_root: Optional[Path]
    mevis_meta_path: Optional[Path]
    mevis_mask_path: Optional[Path]


def db_eval_iou(annotation: np.ndarray, segmentation: np.ndarray, void_pixels: Optional[np.ndarray] = None) -> np.ndarray:
    assert annotation.shape == segmentation.shape, (
        f"Annotation({annotation.shape}) and segmentation:{segmentation.shape} dimensions do not match."
    )
    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)

    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape, (
            f"Annotation({annotation.shape}) and void pixels:{void_pixels.shape} dimensions do not match."
        )
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(segmentation, dtype=bool)

    inters = np.sum((segmentation & annotation) & np.logical_not(void_pixels), axis=(-2, -1))
    union = np.sum((segmentation | annotation) & np.logical_not(void_pixels), axis=(-2, -1))

    j = inters / union
    if np.ndim(j) == 0:
        j = 1 if np.isclose(union, 0) else j
    else:
        j[np.isclose(union, 0)] = 1
    return j


def _seg2bmap(seg: np.ndarray, width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
    seg = seg.astype(bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]
    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)
    assert not (width > w or height > h or abs(ar1 - ar2) > 0.01), (
        f"Cannot convert {w}x{h} seg to {width}x{height} bmap."
    )

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        return b

    bmap = np.zeros((height, width))
    for x in range(w):
        for y in range(h):
            if b[y, x]:
                j = 1 + np.floor((y - 1) + height / h)
                i = 1 + np.floor((x - 1) + width / h)
                bmap[int(j), int(i)] = 1
    return bmap


def _f_measure(
    foreground_mask: np.ndarray,
    gt_mask: np.ndarray,
    void_pixels: Optional[np.ndarray] = None,
    bound_th: float = 0.008,
) -> float:
    assert np.atleast_3d(foreground_mask).shape[2] == 1
    if void_pixels is not None:
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(foreground_mask).astype(bool)

    bound_pix = bound_th if bound_th >= 1 else np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    fg_boundary = _seg2bmap(foreground_mask * np.logical_not(void_pixels))
    gt_boundary = _seg2bmap(gt_mask * np.logical_not(void_pixels))

    from skimage.morphology import disk

    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))

    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def db_eval_boundary(
    annotation: np.ndarray,
    segmentation: np.ndarray,
    void_pixels: Optional[np.ndarray] = None,
    bound_th: float = 0.008,
) -> np.ndarray:
    assert annotation.shape == segmentation.shape
    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape
    if annotation.ndim == 3:
        n_frames = annotation.shape[0]
        f_res = np.zeros(n_frames)
        for frame_id in range(n_frames):
            void_pixels_frame = None if void_pixels is None else void_pixels[frame_id, :, :]
            f_res[frame_id] = _f_measure(
                segmentation[frame_id, :, :],
                annotation[frame_id, :, :],
                void_pixels_frame,
                bound_th=bound_th,
            )
        return f_res
    if annotation.ndim == 2:
        return np.array(_f_measure(segmentation, annotation, void_pixels, bound_th=bound_th))
    raise ValueError(f"db_eval_boundary does not support tensors with {annotation.ndim} dimensions")


def apply_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.figsize"] = (10, 8)
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_repo_root(start: Optional[Path] = None) -> Path:
    cwd = (start or Path.cwd()).resolve()
    candidates = [
        cwd,
        cwd / "segmentation" / "SAMWISE",
        cwd / "SAMWISE",
        cwd.parent,
        cwd.parent / "SAMWISE",
    ]
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "output").exists() and (candidate / "inference_davis.py").exists():
            return candidate
    return cwd


def first_existing(paths: Iterable[Optional[Path]]) -> Optional[Path]:
    for path in paths:
        if path is None:
            continue
        path = Path(path)
        if path.exists():
            return path
    return None


def build_context(
    version: str = "roberta_repro",
    mevis_split: str = "valid_u",
    repo_root: Optional[Path] = None,
    output_root: Optional[Path] = None,
    save_root: Optional[Path] = None,
) -> AnalysisContext:
    repo = resolve_repo_root(repo_root)
    output = Path(output_root).resolve() if output_root else repo / "output"
    save = Path(save_root).resolve() if save_root else output / "visualize" / version
    sibling_dataset_root = repo.parent.parent / "dataset"

    return AnalysisContext(
        repo_root=repo,
        output_root=output,
        version=version,
        mevis_split=mevis_split,
        save_root=save,
        davis_output_root=output / "davis" / version / "eval_davis" / "valid",
        mevis_output_root=output / "mevis" / version,
        mevis_pred_root=output / "mevis" / version / "Annotations",
        sibling_dataset_root=sibling_dataset_root,
        davis_rgb_root=first_existing(
            [
                repo / "data" / "ref-davis" / "valid" / "JPEGImages",
                sibling_dataset_root / "davis17" / "valid" / "JPEGImages",
            ]
        ),
        davis_gt_root=first_existing(
            [
                repo / "data" / "ref-davis" / "valid" / "Annotations",
                sibling_dataset_root / "davis17" / "valid" / "Annotations",
            ]
        ),
        mevis_rgb_root=first_existing(
            [
                repo / "data" / "MeViS_release" / mevis_split / "JPEGImages",
                sibling_dataset_root / "mevis" / mevis_split / "JPEGImages",
            ]
        ),
        mevis_meta_path=first_existing(
            [
                repo / "data" / "MeViS_release" / mevis_split / "meta_expressions.json",
                repo / "data" / "MeViS_release" / mevis_split / "meta_expressions_v2.json",
                sibling_dataset_root / "mevis" / mevis_split / "meta_expressions.json",
                sibling_dataset_root / "mevis" / mevis_split / "meta_expressions_v2.json",
                sibling_dataset_root / "mevis" / mevis_split / "meta_expressions_v2_release.json",
            ]
        ),
        mevis_mask_path=first_existing(
            [
                repo / "data" / "MeViS_release" / mevis_split / "mask_dict.json",
                sibling_dataset_root / "mevis" / mevis_split / "mask_dict.json",
            ]
        ),
    )


def print_context_summary(ctx: AnalysisContext) -> None:
    print(f"REPO_ROOT: {ctx.repo_root}")
    print(f"OUTPUT_ROOT: {ctx.output_root}")
    print(f"VERSION: {ctx.version}")
    print(f"DAVIS_OUTPUT_ROOT: {ctx.davis_output_root}")
    print(f"MEVIS_OUTPUT_ROOT: {ctx.mevis_output_root}")
    print(f"MEVIS_PRED_ROOT: {ctx.mevis_pred_root}")
    print(f"DAVIS_RGB_ROOT: {ctx.davis_rgb_root}")
    print(f"DAVIS_GT_ROOT: {ctx.davis_gt_root}")
    print(f"MEVIS_RGB_ROOT: {ctx.mevis_rgb_root}")
    print(f"MEVIS_META_PATH: {ctx.mevis_meta_path}")
    print(f"MEVIS_MASK_PATH: {ctx.mevis_mask_path}")
    print(f"SAVE_ROOT: {ctx.save_root}")


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_davis_results(ctx: AnalysisContext) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ann_root = ctx.davis_output_root
    if not ann_root.exists():
        raise FileNotFoundError(f"Missing DAVIS annotation root: {ann_root}")

    rows = []
    for anno_dir in sorted(ann_root.glob("anno_*")):
        csv_path = anno_dir / "per-sequence_results-val.csv"
        if not csv_path.exists():
            warnings.warn(f"Skipping missing DAVIS CSV: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        split_df = df["Sequence"].str.rsplit("_", n=1, expand=True)
        df["source_video"] = split_df[0]
        df["sequence_id"] = split_df[1]
        df["annotator"] = anno_dir.name
        df["group"] = f"DAVIS {anno_dir.name}"
        df["J"] = df["J-Mean"] * 100.0
        df["F"] = df["F-Mean"] * 100.0
        df["J&F"] = (df["J"] + df["F"]) / 2.0
        rows.append(df[["annotator", "group", "Sequence", "source_video", "sequence_id", "J", "F", "J&F"]])

    if not rows:
        raise RuntimeError("No DAVIS per-sequence CSVs were loaded.")

    raw = pd.concat(rows, ignore_index=True)
    video = (
        raw.groupby(["annotator", "group", "source_video"], as_index=False)[["J", "F", "J&F"]]
        .mean()
        .sort_values(["annotator", "J&F", "source_video"])
        .reset_index(drop=True)
    )

    disagreement = (
        video.groupby("source_video", as_index=False)
        .agg(
            annotator_count=("annotator", "nunique"),
            J_mean=("J", "mean"),
            F_mean=("F", "mean"),
            JF_mean=("J&F", "mean"),
            J_std=("J", "std"),
            F_std=("F", "std"),
            JF_std=("J&F", "std"),
            J_min=("J", "min"),
            J_max=("J", "max"),
            F_min=("F", "min"),
            F_max=("F", "max"),
            JF_min=("J&F", "min"),
            JF_max=("J&F", "max"),
        )
        .fillna(0.0)
    )
    disagreement["J_range"] = disagreement["J_max"] - disagreement["J_min"]
    disagreement["F_range"] = disagreement["F_max"] - disagreement["F_min"]
    disagreement["JF_range"] = disagreement["JF_max"] - disagreement["JF_min"]
    disagreement = disagreement.sort_values("JF_range", ascending=False).reset_index(drop=True)
    return raw, video, disagreement


def load_mevis_metadata(ctx: AnalysisContext) -> Optional[dict]:
    if ctx.mevis_meta_path is None or not ctx.mevis_meta_path.exists():
        warnings.warn("MeViS metadata file was not found; detailed MeViS analysis will fall back where possible.")
        return None
    return load_json(ctx.mevis_meta_path)


def load_mevis_mask_dict(ctx: AnalysisContext) -> Optional[dict]:
    if ctx.mevis_mask_path is None or not ctx.mevis_mask_path.exists():
        warnings.warn("MeViS mask_dict.json was not found; GT-backed MeViS analysis is unavailable.")
        return None
    return load_json(ctx.mevis_mask_path)


def combine_video_results(davis_video_df: pd.DataFrame, mevis_video_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if not davis_video_df.empty:
        frames.append(davis_video_df[["group", "source_video", "J", "F", "J&F"]])
    if not mevis_video_df.empty:
        frames.append(mevis_video_df[["group", "source_video", "J", "F", "J&F"]])
    if not frames:
        return pd.DataFrame(columns=["group", "source_video", "J", "F", "J&F"])
    return pd.concat(frames, ignore_index=True)


def _load_mevis_video_log(ctx: AnalysisContext) -> pd.DataFrame:
    log_path = ctx.mevis_output_root / "log_metrics_byvid.txt"
    if not log_path.exists():
        warnings.warn(f"Missing MeViS video-metric log: {log_path}")
        return pd.DataFrame(columns=["source_video", "J", "F", "J&F", "group", "metrics_source"])

    rows = []
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            match = MEVIS_VIDEO_LOG_PATTERN.match(line.strip())
            if not match:
                continue
            rows.append(
                {
                    "index": int(match.group("index")),
                    "J": float(match.group("j")) * 100.0,
                    "F": float(match.group("f")) * 100.0,
                    "J&F": float(match.group("jf")) * 100.0,
                }
            )

    if not rows:
        warnings.warn(f"No per-video MeViS metrics could be parsed from {log_path}")
        return pd.DataFrame(columns=["source_video", "J", "F", "J&F", "group", "metrics_source"])

    pred_dirs = sorted(path.name for path in ctx.mevis_pred_root.iterdir() if path.is_dir()) if ctx.mevis_pred_root.exists() else []
    if len(pred_dirs) == len(rows):
        for row, source_video in zip(rows, pred_dirs):
            row["source_video"] = source_video
        metrics_source = "log_metrics_byvid_sorted_prediction_dirs"
    else:
        warnings.warn(
            "MeViS metadata is unavailable and prediction-dir count did not match the per-video log count; "
            "using synthetic video ids for plots."
        )
        for row in rows:
            row["source_video"] = f"video_{row['index']:03d}"
        metrics_source = "log_metrics_byvid_synthetic_ids"

    video_df = pd.DataFrame(rows).sort_values(["J&F", "source_video"]).reset_index(drop=True)
    video_df["group"] = "MeViS"
    video_df["metrics_source"] = metrics_source
    return video_df


def build_mevis_frame_index(meta: Optional[dict]) -> Dict[str, Dict[str, int]]:
    frame_index: Dict[str, Dict[str, int]] = {}
    if meta is None:
        return frame_index
    for source_video, video_meta in meta.get("videos", {}).items():
        frames = video_meta.get("frames", [])
        frame_index[source_video] = {frame_name: idx for idx, frame_name in enumerate(frames)}
    return frame_index


def decode_mevis_gt_mask(
    source_video: str,
    expression_id: str,
    frame_name: str,
    mevis_meta: Optional[dict],
    mevis_mask_dict: Optional[dict],
    mevis_frame_index: Dict[str, Dict[str, int]],
) -> Optional[np.ndarray]:
    if cocomask is None:
        warnings.warn("pycocotools is unavailable; MeViS GT reconstruction is disabled.")
        return None
    if mevis_meta is None or mevis_mask_dict is None:
        return None

    video_meta = mevis_meta.get("videos", {}).get(source_video)
    if video_meta is None:
        return None
    expr_meta = video_meta.get("expressions", {}).get(str(expression_id), {})
    anno_ids = expr_meta.get("anno_id", [])
    if not anno_ids:
        return None

    frame_idx = mevis_frame_index.get(source_video, {}).get(frame_name)
    if frame_idx is None:
        return None

    mask = None
    for anno_id in anno_ids:
        anno_masks = mevis_mask_dict.get(str(anno_id), [])
        if frame_idx >= len(anno_masks):
            continue
        rle = anno_masks[frame_idx]
        if not rle:
            continue
        decoded = cocomask.decode(rle)
        if decoded.ndim == 3:
            decoded = decoded[..., 0]
        decoded = decoded.astype(np.uint8)
        if mask is None:
            mask = np.zeros_like(decoded, dtype=np.uint8)
        mask = np.maximum(mask, decoded)
    return mask


def _compute_mevis_results_from_gt(
    ctx: AnalysisContext,
    meta: dict,
    mask_dict: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frame_index = build_mevis_frame_index(meta)
    rows = []

    for source_video, video_meta in meta.get("videos", {}).items():
        pred_video_dir = ctx.mevis_pred_root / source_video
        if not pred_video_dir.exists():
            continue
        frames = video_meta.get("frames", [])
        for expression_id, expr_meta in video_meta.get("expressions", {}).items():
            pred_expr_dir = pred_video_dir / str(expression_id)
            if not pred_expr_dir.exists():
                continue

            available_frames = [frame for frame in frames if (pred_expr_dir / f"{frame}.png").exists()]
            if not available_frames:
                continue

            pred_masks = []
            gt_masks = []
            for frame_name in available_frames:
                pred_mask = load_mask_image(pred_expr_dir / f"{frame_name}.png")
                gt_mask = decode_mevis_gt_mask(
                    source_video,
                    str(expression_id),
                    frame_name,
                    meta,
                    mask_dict,
                    frame_index,
                )
                if pred_mask is None or gt_mask is None:
                    continue
                pred_masks.append((pred_mask > 0).astype(np.uint8))
                gt_masks.append((gt_mask > 0).astype(np.uint8))

            if not pred_masks:
                continue

            pred_stack = np.stack(pred_masks, axis=0)
            gt_stack = np.stack(gt_masks, axis=0)
            j = float(db_eval_iou(gt_stack, pred_stack).mean()) * 100.0
            f = float(db_eval_boundary(gt_stack, pred_stack).mean()) * 100.0
            rows.append(
                {
                    "source_video": source_video,
                    "expression_id": str(expression_id),
                    "J": j,
                    "F": f,
                    "J&F": (j + f) / 2.0,
                    "expression_text": expr_meta.get("exp"),
                    "anno_ids": expr_meta.get("anno_id", []),
                    "obj_ids": expr_meta.get("obj_id", []),
                    "num_frames": len(available_frames),
                    "metrics_source": "recomputed_from_predictions_and_gt",
                }
            )

    expr_df = pd.DataFrame(rows)
    if expr_df.empty:
        warnings.warn("No GT-backed MeViS expression metrics were reconstructed; falling back to the per-video log.")
        return expr_df, _load_mevis_video_log(ctx)

    expr_df = expr_df.sort_values(["J&F", "source_video", "expression_id"]).reset_index(drop=True)
    video_df = (
        expr_df.groupby("source_video", as_index=False)[["J", "F", "J&F"]]
        .mean()
        .assign(group="MeViS", metrics_source="recomputed_from_predictions_and_gt")
        .sort_values(["J&F", "source_video"])
        .reset_index(drop=True)
    )
    return expr_df, video_df


def load_mevis_results(
    ctx: AnalysisContext,
    meta: Optional[dict] = None,
    mask_dict: Optional[dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if meta is not None and mask_dict is not None:
        return _compute_mevis_results_from_gt(ctx, meta, mask_dict)

    video_df = _load_mevis_video_log(ctx)
    expr_df = pd.DataFrame(
        columns=[
            "source_video",
            "expression_id",
            "J",
            "F",
            "J&F",
            "expression_text",
            "anno_ids",
            "obj_ids",
            "num_frames",
            "metrics_source",
        ]
    )
    return expr_df, video_df


def annotate_worst_points(ax: plt.Axes, df: pd.DataFrame, label_col: str, n_per_group: int = 3) -> None:
    if df.empty:
        return
    for group_name in df["group"].dropna().unique():
        subset = df[df["group"] == group_name].nsmallest(n_per_group, "J&F")
        for _, row in subset.iterrows():
            ax.annotate(
                str(row[label_col]),
                (row["J"], row["F"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=9,
                alpha=0.85,
            )


def plot_main_video_scatter(
    davis_video_df: pd.DataFrame,
    mevis_video_df: pd.DataFrame,
    label_top_n: int = 3,
) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    combined = combine_video_results(davis_video_df, mevis_video_df)
    fig, ax = plt.subplots(figsize=(12, 10))
    for group_name in GROUP_ORDER:
        subset = combined[combined["group"] == group_name]
        if subset.empty:
            continue
        ax.scatter(
            subset["J"],
            subset["F"],
            s=70,
            alpha=0.8,
            c=PALETTE[group_name],
            edgecolor="white",
            linewidth=0.8,
            label=group_name,
        )
    annotate_worst_points(ax, combined, label_col="source_video", n_per_group=label_top_n)
    ax.set_title("Video-level J vs F")
    ax.set_xlabel("J (%)")
    ax.set_ylabel("F (%)")
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.legend(title="Group", loc="lower right")
    fig.tight_layout()
    return fig, ax, combined


def plot_metric_distributions(combined_video_df: pd.DataFrame) -> Tuple[plt.Figure, np.ndarray]:
    melted = combined_video_df.melt(
        id_vars=["group", "source_video"],
        value_vars=["J", "F", "J&F"],
        var_name="metric",
        value_name="score",
    )
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)
    for ax, metric_name in zip(axes, ["J", "F", "J&F"]):
        subset = melted[melted["metric"] == metric_name]
        present_groups = [group for group in GROUP_ORDER if group in subset["group"].unique()]
        palette = [PALETTE[group] for group in present_groups]
        sns.boxplot(
            data=subset,
            x="group",
            y="score",
            hue="group",
            order=present_groups,
            palette=palette,
            ax=ax,
            showfliers=False,
            dodge=False,
            legend=False,
        )
        sns.stripplot(
            data=subset,
            x="group",
            y="score",
            order=present_groups,
            color="black",
            alpha=0.5,
            size=4,
            ax=ax,
        )
        ax.set_title(f"{metric_name} distribution")
        ax.set_xlabel("")
        ax.set_ylabel(f"{metric_name} (%)")
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig, axes


def plot_davis_raw_sequence_scatter(
    davis_raw_df: pd.DataFrame,
    label_top_n: int = 2,
) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(12, 10))
    for group_name in GROUP_ORDER[:-1]:
        subset = davis_raw_df[davis_raw_df["group"] == group_name]
        if subset.empty:
            continue
        ax.scatter(
            subset["J"],
            subset["F"],
            s=40,
            alpha=0.6,
            c=PALETTE[group_name],
            edgecolor="white",
            linewidth=0.4,
            label=group_name,
        )
    annotate_worst_points(ax, davis_raw_df, label_col="Sequence", n_per_group=label_top_n)
    ax.set_title("DAVIS raw-sequence J vs F")
    ax.set_xlabel("J (%)")
    ax.set_ylabel("F (%)")
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.legend(title="Annotator", loc="lower right")
    fig.tight_layout()
    return fig, ax


def build_ranked_tables(
    davis_raw: pd.DataFrame,
    davis_video: pd.DataFrame,
    davis_disagreement: pd.DataFrame,
    mevis_expr: pd.DataFrame,
    mevis_video: pd.DataFrame,
    top_n: int = 15,
) -> Dict[str, pd.DataFrame]:
    tables = {
        "worst_davis_videos_per_annotator": (
            davis_video.sort_values(["annotator", "J&F", "J", "F", "source_video"])
            .groupby("annotator", group_keys=False)
            .head(top_n)
            .reset_index(drop=True)
        ),
        "worst_davis_sequences": davis_raw.nsmallest(top_n, "J&F").reset_index(drop=True),
        "largest_davis_annotator_disagreement": davis_disagreement.head(top_n).reset_index(drop=True),
    }
    if not mevis_video.empty:
        tables["worst_mevis_videos"] = mevis_video.nsmallest(top_n, "J&F").reset_index(drop=True)
    else:
        tables["worst_mevis_videos"] = pd.DataFrame()
    if not mevis_expr.empty:
        tables["worst_mevis_expressions"] = mevis_expr.nsmallest(top_n, "J&F").reset_index(drop=True)
    else:
        tables["worst_mevis_expressions"] = pd.DataFrame()
    return tables


def load_image_rgb(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None or not Path(path).exists():
        return None
    return np.array(Image.open(path).convert("RGB"))


def load_mask_image(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None or not Path(path).exists():
        return None
    return np.array(Image.open(path))


def choose_frame_names(frame_names: Iterable[str], n: int = 6) -> list[str]:
    frame_names = list(frame_names)
    if not frame_names:
        return []
    if len(frame_names) <= n:
        return frame_names
    idx = np.linspace(0, len(frame_names) - 1, num=n, dtype=int)
    return [frame_names[i] for i in idx]


def to_binary_mask(mask_array: Optional[np.ndarray], object_id: Optional[int] = None) -> Optional[np.ndarray]:
    if mask_array is None:
        return None
    mask_array = np.asarray(mask_array)
    if object_id is None:
        return (mask_array > 0).astype(np.uint8)
    return (mask_array == int(object_id)).astype(np.uint8)


def fp_fn_overlay(
    gt_mask: Optional[np.ndarray],
    pred_mask: Optional[np.ndarray],
    rgb: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    if gt_mask is None or pred_mask is None:
        return None
    if rgb is None:
        base = np.full((*gt_mask.shape, 3), 240, dtype=np.uint8)
    else:
        base = np.asarray(rgb).copy()
        if base.ndim == 2:
            base = np.stack([base] * 3, axis=-1)
        base = base.astype(np.uint8)

    overlay = base.astype(np.float32)
    gt_mask = gt_mask.astype(bool)
    pred_mask = pred_mask.astype(bool)
    tp = gt_mask & pred_mask
    fp = pred_mask & (~gt_mask)
    fn = gt_mask & (~pred_mask)

    overlay[tp] = 0.65 * overlay[tp] + 0.35 * np.array([80, 200, 120], dtype=np.float32)
    overlay[fp] = np.array([230, 76, 60], dtype=np.float32)
    overlay[fn] = np.array([65, 105, 225], dtype=np.float32)
    return overlay.astype(np.uint8)


def positive_overlay(
    pred_mask: Optional[np.ndarray],
    rgb: Optional[np.ndarray] = None,
    color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.45,
) -> Optional[np.ndarray]:
    if pred_mask is None:
        return None
    pred_mask = pred_mask.astype(bool)
    if rgb is None:
        base = np.full((*pred_mask.shape, 3), 240, dtype=np.uint8)
    else:
        base = np.asarray(rgb).copy()
        if base.ndim == 2:
            base = np.stack([base] * 3, axis=-1)
        base = base.astype(np.uint8)
    overlay = base.astype(np.float32)
    overlay[pred_mask] = (1.0 - alpha) * overlay[pred_mask] + alpha * np.array(color, dtype=np.float32)
    return overlay.astype(np.uint8)


def show_panel(ax: plt.Axes, image: Optional[np.ndarray], title: str, cmap: Optional[str] = None) -> None:
    if image is None:
        ax.text(0.5, 0.5, "Missing", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        ax.set_title(title)
        return
    if image.ndim == 2:
        ax.imshow(image, cmap=cmap or "viridis")
    else:
        ax.imshow(image)
    ax.set_title(title)
    ax.set_axis_off()


def create_davis_case_figure(
    ctx: AnalysisContext,
    davis_video_df: pd.DataFrame,
    source_video: str,
    annotator: str = "anno_0",
    num_frames: int = 6,
    object_id: Optional[int] = None,
) -> plt.Figure:
    pred_dir = ctx.davis_output_root / annotator / source_video
    if not pred_dir.exists():
        raise FileNotFoundError(f"Missing DAVIS prediction directory: {pred_dir}")

    frame_names = sorted(path.stem for path in pred_dir.glob("*.png"))
    selected_frames = choose_frame_names(frame_names, n=num_frames)
    if not selected_frames:
        raise RuntimeError(f"No DAVIS prediction frames found for {annotator}/{source_video}")

    metric_row = None
    metric_subset = davis_video_df[
        (davis_video_df["annotator"] == annotator) & (davis_video_df["source_video"] == source_video)
    ]
    if not metric_subset.empty:
        metric_row = metric_subset.iloc[0]

    fig, axes = plt.subplots(len(selected_frames), 4, figsize=(18, 4 * len(selected_frames)))
    if len(selected_frames) == 1:
        axes = np.array([axes])

    for row_idx, frame_name in enumerate(selected_frames):
        rgb = load_image_rgb(ctx.davis_rgb_root / source_video / f"{frame_name}.jpg" if ctx.davis_rgb_root else None)
        gt_raw = load_mask_image(ctx.davis_gt_root / source_video / f"{frame_name}.png" if ctx.davis_gt_root else None)
        pred_raw = load_mask_image(pred_dir / f"{frame_name}.png")

        gt_bin = to_binary_mask(gt_raw, object_id=object_id)
        pred_bin = to_binary_mask(pred_raw, object_id=object_id)
        overlay = fp_fn_overlay(gt_bin, pred_bin, rgb=rgb)

        display_gt = gt_raw if object_id is None else gt_bin
        display_pred = pred_raw if object_id is None else pred_bin

        show_panel(axes[row_idx, 0], rgb, f"RGB {frame_name}")
        show_panel(axes[row_idx, 1], display_gt, f"GT {frame_name}", cmap="tab20")
        show_panel(axes[row_idx, 2], display_pred, f"Pred {frame_name}", cmap="tab20")
        show_panel(axes[row_idx, 3], overlay, f"FP/FN {frame_name}")

    summary = f"DAVIS {annotator} / {source_video}"
    if object_id is not None:
        summary += f" / object_id={object_id}"
    if metric_row is not None:
        summary += f" | J={metric_row['J']:.2f}, F={metric_row['F']:.2f}, J&F={metric_row['J&F']:.2f}"
    fig.suptitle(summary + "\nOverlay colors: red=FP, blue=FN, green=TP", y=1.02, fontsize=16)
    fig.tight_layout()
    return fig


def create_mevis_case_figure(
    ctx: AnalysisContext,
    mevis_expr_df: pd.DataFrame,
    source_video: str,
    expression_id: Optional[str] = None,
    num_frames: int = 6,
    mevis_meta: Optional[dict] = None,
    mevis_mask_dict: Optional[dict] = None,
) -> plt.Figure:
    expr_subset = mevis_expr_df[mevis_expr_df["source_video"] == source_video].copy()
    pred_video_dir = ctx.mevis_pred_root / source_video
    if not pred_video_dir.exists():
        raise FileNotFoundError(f"Missing MeViS prediction directory: {pred_video_dir}")

    if expression_id is None:
        if not expr_subset.empty:
            expression_id = str(expr_subset.nsmallest(1, "J&F").iloc[0]["expression_id"])
        else:
            expression_ids = sorted(path.name for path in pred_video_dir.iterdir() if path.is_dir())
            if not expression_ids:
                raise RuntimeError(f"No MeViS expressions found for video {source_video}")
            expression_id = expression_ids[0]
    else:
        expression_id = str(expression_id)

    row = expr_subset[expr_subset["expression_id"] == expression_id]
    metric_row = row.iloc[0] if not row.empty else None

    pred_dir = pred_video_dir / expression_id
    if not pred_dir.exists():
        raise FileNotFoundError(f"Missing MeViS prediction directory: {pred_dir}")

    pred_frame_names = {path.stem for path in pred_dir.glob("*.png")}
    meta_frames = []
    if mevis_meta is not None:
        meta_frames = mevis_meta.get("videos", {}).get(source_video, {}).get("frames", [])
    frame_names = [frame for frame in meta_frames if frame in pred_frame_names] if meta_frames else sorted(pred_frame_names)
    selected_frames = choose_frame_names(frame_names, n=num_frames)
    if not selected_frames:
        raise RuntimeError(f"No MeViS prediction frames found for {source_video}/{expression_id}")

    mevis_frame_index = build_mevis_frame_index(mevis_meta)
    fig, axes = plt.subplots(len(selected_frames), 4, figsize=(18, 4 * len(selected_frames)))
    if len(selected_frames) == 1:
        axes = np.array([axes])

    for row_idx, frame_name in enumerate(selected_frames):
        rgb = load_image_rgb(ctx.mevis_rgb_root / source_video / f"{frame_name}.jpg" if ctx.mevis_rgb_root else None)
        pred_raw = load_mask_image(pred_dir / f"{frame_name}.png")
        pred_bin = to_binary_mask(pred_raw)
        gt_bin = decode_mevis_gt_mask(
            source_video,
            expression_id,
            frame_name,
            mevis_meta,
            mevis_mask_dict,
            mevis_frame_index,
        )
        overlay = fp_fn_overlay(gt_bin, pred_bin, rgb=rgb)
        if overlay is None:
            overlay = positive_overlay(pred_bin, rgb=rgb)

        show_panel(axes[row_idx, 0], rgb, f"RGB {frame_name}")
        show_panel(axes[row_idx, 1], gt_bin, f"GT {frame_name}", cmap="gray")
        show_panel(axes[row_idx, 2], pred_bin, f"Pred {frame_name}", cmap="gray")
        show_panel(axes[row_idx, 3], overlay, "FP/FN" if gt_bin is not None else "Pred overlay")

    summary = f"MeViS {source_video} / expression {expression_id}"
    if metric_row is not None:
        summary += f" | J={metric_row['J']:.2f}, F={metric_row['F']:.2f}, J&F={metric_row['J&F']:.2f}"
        expr_text = metric_row.get("expression_text") or "<missing expression text>"
        summary += f"\nExpression: {expr_text}"
    elif mevis_meta is not None:
        expr_meta = mevis_meta.get("videos", {}).get(source_video, {}).get("expressions", {}).get(str(expression_id), {})
        expr_text = expr_meta.get("exp")
        if expr_text:
            summary += f"\nExpression: {expr_text}"
    if mevis_mask_dict is None or mevis_meta is None:
        summary += "\nGT unavailable locally; showing prediction-only overlay in the last column."
    else:
        summary += "\nOverlay colors: red=FP, blue=FN, green=TP"
    fig.suptitle(summary, y=1.02, fontsize=15)
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path: Path, dpi: int = 200, close: bool = True) -> Path:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return path
