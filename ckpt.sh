#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
CKPT_DIR="$REPO_ROOT/ckpt"
PRETRAIN_DIR="$REPO_ROOT/pretrain"

DAVIS_ID="17Ei9XU678tCiiV14c-9EB9ZqXVrj4qEw"
MEVIS_ID="1Molt2up2bP41ekeczXWQU-LWTskKJOV2"

DAVIS_CKPT="$CKPT_DIR/refdavis_refytvos_roberta_hiera_b.pth"
MEVIS_CKPT="$CKPT_DIR/mevis_roberta_hiera_b.pth"

note() {
  echo "[ckpt] $*"
}

download_gdrive_file() {
  local file_id="$1"
  local output_path="$2"

  if gdown --help 2>&1 | grep -q -- "--fuzzy"; then
    gdown --fuzzy "https://drive.google.com/file/d/${file_id}/view" -O "$output_path"
  else
    gdown "$file_id" -O "$output_path"
  fi
}

if ! command -v gdown >/dev/null 2>&1; then
  echo "[ckpt] ERROR: gdown is not available. Install requirements.txt first." >&2
  exit 1
fi

mkdir -p "$CKPT_DIR" "$PRETRAIN_DIR"

if [[ ! -f "$DAVIS_CKPT" ]]; then
  note "Downloading Ref-DAVIS / Ref-Youtube-VOS checkpoint"
  download_gdrive_file "$DAVIS_ID" "$DAVIS_CKPT"
else
  note "Reusing existing $DAVIS_CKPT"
fi

if [[ ! -f "$MEVIS_CKPT" ]]; then
  note "Downloading MeViS checkpoint"
  download_gdrive_file "$MEVIS_ID" "$MEVIS_CKPT"
else
  note "Reusing existing $MEVIS_CKPT"
fi

cat <<EOF
Checkpoint bootstrap complete.

Expected files:
  $DAVIS_CKPT
  $MEVIS_CKPT

SAM2 and RoBERTa weights are downloaded automatically by the official code into:
  $PRETRAIN_DIR
EOF
