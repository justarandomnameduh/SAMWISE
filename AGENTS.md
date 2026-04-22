# Repository Guidelines

## Project Structure

This repo is the SAMWISE fork used for local editing and GitHub-backed delivery. Core model and dataset code live under `models/`, `datasets/`, `davis2017/`, `tools/`, and `util/`. Fork-local reproduction helpers live at the repo root and in `tools/`, for example:

- `tools/prepare_local_rvos_data.sh`
- `ckpt.sh`
- `samwise_inference.sh`
- `MANUAL.md`

Large runtime artifacts are intentionally untracked. The current `.gitignore` already excludes `data/`, `output/`, `pretrain/`, checkpoints, archives, and demo outputs.

## Working Model

Work is split across two machines:

- Local editing machine: use for code changes, documentation, Git operations, and lightweight script validation.
- GPU machine such as Bunya: use for environment setup, checkpoint downloads, and all real inference runs.

Do not assume the local absolute paths also exist on the GPU machine. Before suggesting run commands in a new session, confirm:

```bash
pwd
git remote -v
git branch --show-current
```

If the user wants code changes, make them in this fork and push to `origin`.

## Reproduction Workflow

When a future session needs to suggest or run reproduction steps, prefer this sequence:

1. Confirm the active checkout and branch, then `git pull`.
2. Create or activate the `samwise` environment.
3. Verify PyTorch and CUDA before any long run.
4. Prepare the local dataset layout with `tools/prepare_local_rvos_data.sh`.
5. Ensure released checkpoints exist with `ckpt.sh`.
6. Use `samwise_inference.sh` for the full Ref-DAVIS17 + MeViS local reproduction flow.
7. Read metrics from `output/reports/` instead of scraping stdout by hand.

If `torch.cuda.is_available()` is `False`, do not start inference yet.

## Environment Notes

The upstream README expects Python 3.10 and PyTorch 2.3.1 with CUDA 11.8. A typical environment setup is:

```bash
conda create -n samwise python=3.10 -y
conda activate samwise
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Quick health check:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
```

If the cluster requires module loading, handle that before activating the env.

## Dataset Expectations

This fork assumes a sibling dataset tree by default:

```text
phd/
├── dataset/
└── segmentation/
    └── SAMWISE/
```

The helper scripts are written around the canonical local source layout:

- `dataset/davis17/`
- `dataset/davis17_raw/DAVIS/`
- `dataset/mevis/`

The runtime layout generated inside this repo is:

- `data/ref-davis/`
- `data/MeViS_release/valid/`
- `data/MeViS_release/valid_u/`

## Git And Change Scope

Keep infra changes thin and localized. Prefer small wrapper scripts and docs over invasive edits to the official model code unless there is a concrete bug that blocks reproduction.

Do not commit:

- checkpoints
- downloaded weights
- extracted datasets
- output logs
- generated reports

Before finishing a coding task, check:

```bash
git status --short
git diff --stat
```

Push the completed changes to `origin/main` unless the user says otherwise.
