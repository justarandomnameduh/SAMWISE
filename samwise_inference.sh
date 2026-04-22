#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

DATASET_ROOT="${DATASET_ROOT:-$(cd "$REPO_ROOT/../.." && pwd)/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output}"
VERSION="${VERSION:-roberta_repro}"
SAM2_VERSION="${SAM2_VERSION:-base}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_CLIP_WINDOW="${EVAL_CLIP_WINDOW:-8}"

DAVIS_CKPT="${DAVIS_CKPT:-$REPO_ROOT/ckpt/refdavis_refytvos_roberta_hiera_b.pth}"
MEVIS_CKPT="${MEVIS_CKPT:-$REPO_ROOT/ckpt/mevis_roberta_hiera_b.pth}"

RUN_CKPT=1
RUN_PREPARE=1

usage() {
  cat <<EOF
Usage: bash samwise_inference.sh [options]

Runs the official SAMWISE Ref-DAVIS17 and MeViS inference flows with local
dataset preparation and writes a compact report under output/reports/.

Options:
  --dataset-root PATH       Dataset root. Default: $DATASET_ROOT
  --output-root PATH        Output root. Default: $OUTPUT_ROOT
  --version NAME            Version label. Default: $VERSION
  --sam2-version NAME       SAM2 backbone: tiny|base|large. Default: $SAM2_VERSION
  --num-workers N           DataLoader workers. Default: $NUM_WORKERS
  --eval-clip-window N      Eval clip window. Default: $EVAL_CLIP_WINDOW
  --davis-ckpt PATH         Override DAVIS checkpoint path
  --mevis-ckpt PATH         Override MeViS checkpoint path
  --skip-ckpt               Skip checkpoint bootstrap
  --skip-prepare            Skip dataset layout preparation
  -h, --help                Show this help
EOF
}

note() {
  echo "[samwise] $*"
}

die() {
  echo "[samwise] ERROR: $*" >&2
  exit 1
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || die "Missing required file: $path"
}

require_dir() {
  local path="$1"
  [[ -d "$path" ]] || die "Missing required directory: $path"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --version)
      VERSION="$2"
      shift 2
      ;;
    --sam2-version)
      SAM2_VERSION="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --eval-clip-window)
      EVAL_CLIP_WINDOW="$2"
      shift 2
      ;;
    --davis-ckpt)
      DAVIS_CKPT="$2"
      shift 2
      ;;
    --mevis-ckpt)
      MEVIS_CKPT="$2"
      shift 2
      ;;
    --skip-ckpt)
      RUN_CKPT=0
      shift
      ;;
    --skip-prepare)
      RUN_PREPARE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

LOG_DIR="$OUTPUT_ROOT/logs/$VERSION"
REPORT_DIR="$OUTPUT_ROOT/reports"
DAVIS_OUTPUT_ROOT="$OUTPUT_ROOT/davis"
MEVIS_OUTPUT_ROOT="$OUTPUT_ROOT/mevis"
mkdir -p "$LOG_DIR" "$REPORT_DIR" "$DAVIS_OUTPUT_ROOT" "$MEVIS_OUTPUT_ROOT"

note "Repo root: $REPO_ROOT"
note "Dataset root: $DATASET_ROOT"
note "Output root: $OUTPUT_ROOT"
note "Version: $VERSION"
note "SAM2 version: $SAM2_VERSION"

require_dir "$DATASET_ROOT"

if ! command -v python >/dev/null 2>&1; then
  die "python is not available in PATH"
fi

python - <<'PY'
import sys

try:
    import torch
except Exception as exc:
    raise SystemExit(f"PyTorch import failed: {exc}")

print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"device_count={torch.cuda.device_count()}")

if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is not active for this shell. Activate the SAMWISE environment on a GPU node "
        "and rerun the job."
    )
PY

if [[ "$RUN_CKPT" -eq 1 ]]; then
  note "Ensuring released checkpoints are present"
  bash "$REPO_ROOT/ckpt.sh" | tee "$LOG_DIR/ckpt.log"
fi

require_file "$DAVIS_CKPT"
require_file "$MEVIS_CKPT"

if [[ "$RUN_PREPARE" -eq 1 ]]; then
  note "Preparing local DAVIS and MeViS data layout"
  bash "$REPO_ROOT/tools/prepare_local_rvos_data.sh" --dataset-root "$DATASET_ROOT" \
    | tee "$LOG_DIR/prepare_data.log"
fi

require_file "$REPO_ROOT/data/ref-davis/meta_expressions/valid/meta_expressions.json"
require_dir "$REPO_ROOT/data/ref-davis/valid/JPEGImages"
require_dir "$REPO_ROOT/data/ref-davis/valid/Annotations"
require_dir "$REPO_ROOT/data/ref-davis/DAVIS"
require_file "$REPO_ROOT/data/MeViS_release/valid_u/meta_expressions.json"
require_file "$REPO_ROOT/data/MeViS_release/valid_u/mask_dict.json"
require_dir "$REPO_ROOT/data/MeViS_release/valid_u/JPEGImages"

note "Running Ref-DAVIS17 inference"
python "$REPO_ROOT/inference_davis.py" \
  --resume "$DAVIS_CKPT" \
  --name_exp "$VERSION" \
  --output_dir "$DAVIS_OUTPUT_ROOT" \
  --davis_path "$REPO_ROOT/data/ref-davis" \
  --sam2_version "$SAM2_VERSION" \
  --num_workers "$NUM_WORKERS" \
  --eval_clip_window "$EVAL_CLIP_WINDOW" \
  --HSA \
  --use_cme_head \
  --no_distributed \
  | tee "$LOG_DIR/davis_inference.log"

note "Running MeViS valid_u inference"
python "$REPO_ROOT/inference_mevis.py" \
  --split valid_u \
  --resume "$MEVIS_CKPT" \
  --name_exp "$VERSION" \
  --output_dir "$MEVIS_OUTPUT_ROOT" \
  --mevis_path "$REPO_ROOT/data/MeViS_release" \
  --sam2_version "$SAM2_VERSION" \
  --num_workers "$NUM_WORKERS" \
  --eval_clip_window "$EVAL_CLIP_WINDOW" \
  --HSA \
  --use_cme_head \
  --no_distributed \
  | tee "$LOG_DIR/mevis_inference.log"

REPORT_PATH="$REPORT_DIR/${VERSION}_report.md"
SUMMARY_JSON_PATH="$REPORT_DIR/${VERSION}_summary.json"

note "Generating report at $REPORT_PATH"
python - "$OUTPUT_ROOT" "$VERSION" "$REPORT_PATH" "$SUMMARY_JSON_PATH" <<'PY'
import csv
import datetime as dt
import json
import os
import re
import sys

output_root, version, report_path, summary_json_path = sys.argv[1:]

davis_target = {"J": 67.4, "F": 74.5, "J&F": 70.6}
mevis_paper_target = {"J": 46.6, "F": 52.4, "J&F": 49.5}

davis_root = os.path.join(output_root, "davis", version, "eval_davis", "valid")
mevis_root = os.path.join(output_root, "mevis", version)
mevis_meta = os.path.join("data", "MeViS_release", "valid_u", "meta_expressions.json")

davis_rows = []
for annotator in range(4):
    csv_path = os.path.join(davis_root, f"anno_{annotator}", "global_results-val.csv")
    if not os.path.isfile(csv_path):
        raise SystemExit(f"Missing DAVIS summary CSV: {csv_path}")
    with open(csv_path, newline="") as handle:
        row = next(csv.DictReader(handle))
    davis_rows.append(
        {
            "annotator": annotator,
            "J&F": float(row["J&F-Mean"]),
            "J": float(row["J-Mean"]),
            "F": float(row["F-Mean"]),
        }
    )

davis_metrics = {
    "J": sum(row["J"] for row in davis_rows) / len(davis_rows),
    "F": sum(row["F"] for row in davis_rows) / len(davis_rows),
    "J&F": sum(row["J&F"] for row in davis_rows) / len(davis_rows),
}

mevis_log = os.path.join(mevis_root, "log.txt")
if not os.path.isfile(mevis_log):
    raise SystemExit(f"Missing MeViS log: {mevis_log}")

with open(mevis_log, "r", encoding="utf-8") as handle:
    log_text = handle.read()

match = None
for candidate in re.finditer(r"^J&F:\s*([0-9.]+)\s*J:\s*([0-9.]+)\s*F:\s*([0-9.]+)\s*$", log_text, re.MULTILINE):
    match = candidate
if match is None:
    raise SystemExit(f"Could not parse MeViS metrics from {mevis_log}")

mevis_metrics = {
    "J&F": float(match.group(1)),
    "J": float(match.group(2)),
    "F": float(match.group(3)),
}

with open(mevis_meta, "r", encoding="utf-8") as handle:
    mevis_videos = json.load(handle)["videos"]
num_expressions = sum(len(video["expressions"]) for video in mevis_videos.values())

summary = {
    "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
    "version": version,
    "output_root": output_root,
    "davis": {
        "metrics": davis_metrics,
        "paper_targets": davis_target,
        "per_annotator": davis_rows,
        "delta": {key: davis_metrics[key] - davis_target[key] for key in davis_metrics},
        "predictions_root": davis_root,
    },
    "mevis": {
        "split": "valid_u",
        "metrics": mevis_metrics,
        "paper_targets": mevis_paper_target,
        "reference_note": (
            "Paper targets come from the official MeViS benchmark in CVPR 2025 Table 1; "
            "this local run evaluates the public valid_u split."
        ),
        "num_expressions": num_expressions,
        "log_path": mevis_log,
    },
}

with open(summary_json_path, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)

report = f"""# SAMWISE Reproduction Report

- Generated: {summary["generated_at"]}
- Version: `{version}`
- Output root: `{output_root}`
- Logs: `{os.path.join(output_root, "logs", version)}`

## Ref-DAVIS17

| Metric | Observed | Paper Target | Delta |
| --- | ---: | ---: | ---: |
| J | {davis_metrics["J"]:.2f} | {davis_target["J"]:.2f} | {davis_metrics["J"] - davis_target["J"]:.2f} |
| F | {davis_metrics["F"]:.2f} | {davis_target["F"]:.2f} | {davis_metrics["F"] - davis_target["F"]:.2f} |
| J&F | {davis_metrics["J&F"]:.2f} | {davis_target["J&F"]:.2f} | {davis_metrics["J&F"] - davis_target["J&F"]:.2f} |

Per-annotator summary:

| Annotator | J | F | J&F |
| --- | ---: | ---: | ---: |
{os.linesep.join(f'| anno_{row["annotator"]} | {row["J"]:.2f} | {row["F"]:.2f} | {row["J&F"]:.2f} |' for row in davis_rows)}

## MeViS

- Split: `valid_u`
- Expressions evaluated: {num_expressions}
- Note: Paper targets below come from the official MeViS benchmark in CVPR 2025 Table 1; this local run is on public `valid_u`.

| Metric | Observed valid_u | Paper Target |
| --- | ---: | ---: |
| J | {mevis_metrics["J"]:.2f} | {mevis_paper_target["J"]:.2f} |
| F | {mevis_metrics["F"]:.2f} | {mevis_paper_target["F"]:.2f} |
| J&F | {mevis_metrics["J&F"]:.2f} | {mevis_paper_target["J&F"]:.2f} |

## Artifacts

- DAVIS predictions: `{davis_root}`
- DAVIS overlays: `{os.path.join(output_root, "davis", version, "overlay_videos")}`
- MeViS logs: `{mevis_log}`
- MeViS overlays: `{os.path.join(output_root, "mevis", version, "overlay_videos")}`
- JSON summary: `{summary_json_path}`
"""

with open(report_path, "w", encoding="utf-8") as handle:
    handle.write(report)
PY

note "Finished. Summary JSON: $SUMMARY_JSON_PATH"
note "Finished. Report: $REPORT_PATH"
