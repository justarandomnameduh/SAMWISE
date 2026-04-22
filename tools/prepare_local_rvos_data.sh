#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_DATASET_ROOT="$(cd "$REPO_ROOT/../.." && pwd)/dataset"
DATASET_ROOT="$DEFAULT_DATASET_ROOT"

usage() {
  cat <<'EOF'
Usage: bash tools/prepare_local_rvos_data.sh [--dataset-root /absolute/path/to/dataset]

Prepare the local SAMWISE runtime dataset layout from the sibling dataset tree.

Expected canonical source layout:
  dataset/
    davis17/
    davis17_raw/DAVIS/
    mevis/

This script is designed for inference-oriented local reproduction and prepares:
  data/ref-davis/
  data/MeViS_release/{valid,valid_u}/
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root)
      [[ $# -ge 2 ]] || { echo "Missing value for --dataset-root" >&2; exit 1; }
      DATASET_ROOT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

require_path() {
  local path="$1"
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
}

replace_with_symlink() {
  local src="$1"
  local dst="$2"
  rm -rf "$dst"
  mkdir -p "$(dirname "$dst")"
  ln -s "$src" "$dst"
}

extract_split_images() {
  local src_dir="$1"
  local dst_dir="$2"
  local tar_path="$src_dir/JPEGImages.tar"

  if [[ -d "$src_dir/JPEGImages" ]] && find "$src_dir/JPEGImages" -mindepth 1 -print -quit >/dev/null; then
    replace_with_symlink "$src_dir/JPEGImages" "$dst_dir/JPEGImages"
    return 0
  fi

  require_path "$tar_path"
  if [[ -d "$dst_dir/JPEGImages" ]] && find "$dst_dir/JPEGImages" -mindepth 1 -print -quit >/dev/null; then
    return 0
  fi

  rm -rf "$dst_dir/JPEGImages"
  mkdir -p "$dst_dir"
  tar -xf "$tar_path" -C "$dst_dir"
}

prepare_mevis_split() {
  local split="$1"
  local src_dir="$DATASET_ROOT/mevis/$split"
  local dst_dir="$REPO_ROOT/data/MeViS_release/$split"
  local meta_name

  require_path "$src_dir"
  require_path "$src_dir/mask_dict.json"
  if [[ "$split" == "valid" ]]; then
    meta_name="meta_expressions_v2_release.json"
  else
    meta_name="meta_expressions_v2.json"
  fi
  require_path "$src_dir/$meta_name"

  mkdir -p "$dst_dir"
  extract_split_images "$src_dir" "$dst_dir"
  replace_with_symlink "$src_dir/mask_dict.json" "$dst_dir/mask_dict.json"
  replace_with_symlink "$src_dir/$meta_name" "$dst_dir/meta_expressions.json"
}

require_path "$DATASET_ROOT/davis17/train"
require_path "$DATASET_ROOT/davis17/valid"
require_path "$DATASET_ROOT/davis17/meta_expressions"
require_path "$DATASET_ROOT/davis17_raw/DAVIS"

mkdir -p "$REPO_ROOT/data"
replace_with_symlink "$DATASET_ROOT/davis17/train" "$REPO_ROOT/data/ref-davis/train"
replace_with_symlink "$DATASET_ROOT/davis17/valid" "$REPO_ROOT/data/ref-davis/valid"
replace_with_symlink "$DATASET_ROOT/davis17/meta_expressions" "$REPO_ROOT/data/ref-davis/meta_expressions"
replace_with_symlink "$DATASET_ROOT/davis17_raw/DAVIS" "$REPO_ROOT/data/ref-davis/DAVIS"

prepare_mevis_split "valid"
prepare_mevis_split "valid_u"

echo "SAMWISE local data layout is ready under $REPO_ROOT/data"
