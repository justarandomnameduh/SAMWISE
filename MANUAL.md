# SAMWISE Reproduction Manual

This fork adds a thin local reproduction layer around the official SAMWISE code so the repo can be edited locally, pushed to GitHub, and then pulled on Bunya for the actual GPU runs.

Assumed local source layout:

- repo: `/home/uqqnguy9/phd/segmentation/SAMWISE`
- sibling dataset root: `/home/uqqnguy9/phd/dataset`

On Bunya or another GPU machine, adapt the absolute paths to the active checkout.

## 1. Pull the latest fork state

```bash
cd /path/to/SAMWISE
git pull origin main
```

## 2. Create the environment

```bash
cd /path/to/SAMWISE

conda create -n samwise python=3.10 -y
conda activate samwise

pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

If your cluster requires CUDA modules, load them before running inference.

Quick check:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
```

## 3. Prepare the local data layout

```bash
cd /path/to/SAMWISE

bash tools/prepare_local_rvos_data.sh \
  --dataset-root /path/to/dataset
```

This creates the runtime layout expected by the official scripts:

- `data/ref-davis`
- `data/MeViS_release/valid`
- `data/MeViS_release/valid_u`

## 4. Download the released checkpoints

```bash
cd /path/to/SAMWISE

bash ckpt.sh
```

This creates:

- `ckpt/refdavis_refytvos_roberta_hiera_b.pth`
- `ckpt/mevis_roberta_hiera_b.pth`

SAM2 and RoBERTa weights are downloaded automatically by the official code into `pretrain/` if missing.

## 5. Run the full local reproduction

```bash
cd /path/to/SAMWISE
conda activate samwise

bash samwise_inference.sh \
  --dataset-root /path/to/dataset \
  --output-root /path/to/SAMWISE/output \
  --version roberta_repro
```

## 6. Read the outputs

Main artifacts:

- report: `output/reports/roberta_repro_report.md`
- summary json: `output/reports/roberta_repro_summary.json`
- DAVIS logs: `output/logs/roberta_repro/davis_inference.log`
- MeViS logs: `output/logs/roberta_repro/mevis_inference.log`
- DAVIS predictions: `output/davis/roberta_repro/Annotations`
- DAVIS evaluation CSVs: `output/davis/roberta_repro/Annotations/anno_<id>/global_results-val.csv`
- MeViS predictions: `output/mevis/roberta_repro/Annotations`
- DAVIS overlay videos: `output/davis/roberta_repro/overlay_videos/<video>/exp_<exp_id>.mp4`
- MeViS overlay videos: `output/mevis/roberta_repro/overlay_videos/<video>/exp_<exp_id>.mp4`
- metric plots: `output/visualize/<dataset>/roberta_repro/<video>/<exp_id>.png`

Overlay video folders include per-video `manifest.json` files with `[relative_path, query]` entries. Overlay videos are generated automatically during inference for all DAVIS and MeViS outputs unless `--overlay-video-first-n 0` is passed.

## 7. Reference targets

Ref-DAVIS17, from the CVPR 2025 paper:

- `J = 67.4`
- `F = 74.5`
- `J&F = 70.6`

MeViS, from the CVPR 2025 paper Table 1:

- `J = 46.6`
- `F = 52.4`
- `J&F = 49.5`

Important note:

- The local helper here evaluates MeViS on the public `valid_u` split.
- The paper’s MeViS numbers come from the official benchmark protocol, so treat them as a reference point rather than a strict apples-to-apples local target.
