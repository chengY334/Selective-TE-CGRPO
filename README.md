# Selective-TE-CGRPO for Qwen2.5-VL-3B

This repository contains the 3B training and evaluation code for
Selective Temporal-Evidence Contrastive GRPO (Selective-TE-CGRPO), built on
Qwen2.5-VL-3B and the Video-R1/Open-R1 training stack.

Only source code and evaluation metadata are included. Model weights, training
datasets, benchmark videos, checkpoints, logs, and raw evaluation outputs are
not distributed in this repository.

## Repository layout

```text
src/
├── scripts/
│   ├── run_selective_te_cgrpo_qwen25vl_3b_a100_3gpu.sh
│   └── run_selective_te_cgrpo_qwen25vl_3b_a100_3gpu_smoke.sh
├── eval_bench.py
├── eval_bench.sh
├── qwen-vl-utils/
└── r1-v/
    ├── Evaluation/
    ├── local_scripts/zero3_cpu_offload.json
    └── src/open_r1/
        ├── grpo.py
        └── trainer/
```

## Environment

- Python 3.10+
- CUDA-capable GPUs
- PyTorch 2.5+
- DeepSpeed 0.15.4
- TRL 0.16.0
- vLLM 0.7.2 for evaluation or optional vLLM training

Install from the repository root:

```bash
bash setup.sh
```

## Data and model preparation

Prepare the following resources outside the Git repository:

1. A Qwen2.5-VL-3B SFT checkpoint.
2. The Video-R1 training data JSON.
3. The corresponding image/video files under one data root.

The launcher requires these environment variables:

- `MODEL_PATH`: local path to the Qwen2.5-VL-3B SFT checkpoint.
- `DATA_ROOT`: root containing the referenced media files.
- `DATASET_PATH`: path to the training JSON file.

## Three-GPU training

```bash
MODEL_PATH=/path/to/qwen25vl-3b-sft \
DATA_ROOT=/path/to/video-r1-data \
DATASET_PATH=/path/to/train.json \
CUDA_VISIBLE_DEVICES=0,1,2 \
bash src/scripts/run_selective_te_cgrpo_qwen25vl_3b_a100_3gpu.sh
```

The default run uses three DeepSpeed ranks, 1,000 optimization steps, two
generations per prompt, selective temporal-evidence token weighting, shuffled
temporal corruption, and an output directory under `outputs/`.

Run a one-step smoke test with the same required paths:

```bash
MODEL_PATH=/path/to/qwen25vl-3b-sft \
DATA_ROOT=/path/to/video-r1-data \
DATASET_PATH=/path/to/train.json \
CUDA_VISIBLE_DEVICES=0,1,2 \
bash src/scripts/run_selective_te_cgrpo_qwen25vl_3b_a100_3gpu_smoke.sh
```

All principal hyperparameters can be overridden through environment variables;
see the launcher for the complete list.

## Evaluation

Place benchmark media below `src/r1-v/Evaluation/` following the paths in the
provided evaluation JSON files, then run for example:

```bash
python src/eval_bench.py \
  --model_path /path/to/checkpoint \
  --file_name selective_te_cgrpo_3b \
  --datasets tempcompass,vsibench \
  --max_frames 16
```

Raw outputs are written to `src/r1-v/eval_results/`, which is excluded from
version control.

## License and attribution

The training package under `src/r1-v/` retains its Apache-2.0 license and
upstream attribution. This project builds on the Video-R1, Open-R1, TRL, and
Qwen-VL ecosystems.
