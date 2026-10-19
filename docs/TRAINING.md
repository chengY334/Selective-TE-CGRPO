# 3B Training Guide

This guide describes the source release's training entry path. The method overview is in the root [README](../README.md). Package versions and validation boundaries are in [Environment](ENVIRONMENT.md).

## Model and data

Use a shared Qwen2.5-VL-3B **SFT checkpoint**, not a completed baseline RL checkpoint. All compared methods should start independently from that checkpoint. Obtain the Video-R1 training mixture and media separately through the [upstream project](https://github.com/tulerfeng/Video-R1) under their applicable access and usage terms.

Dataset input may be a JSON array or JSONL. Each record must provide `problem`, `solution`, `data_type`, `problem_type`, and `path`. `data_type` is `image` or `video`. Supported task types are `multiple choice`, `numerical`, `regression`, `OCR`, and `free-form`; multiple-choice records also require an `options` list. Media paths must be absolute or resolve relative to `data_root`.

Training uses the full configured dataset. Only smoke tests construct a small mixed-modality subset, without modifying the original records.

## Private configuration

```bash
cp server_3b_config.json server_3b_config.local.json
```

Set `model_path`, `data_root`, `dataset_path`, and `output_root` to **absolute paths**. Distributed workers run from the training subdirectory, so relative paths can resolve differently from launcher paths. Set `cuda_visible_devices` to three available physical GPU IDs; the example uses `0,1,3`.

The tracked configuration contains placeholders. Git ignores the local configuration, but manual GitHub uploads do not honor ignore rules automatically. Do not upload private paths or credentials.

The launcher prepends local training and Qwen utility sources to `PYTHONPATH`. Replacing the local video sampler with an unrelated installed version can change the frame budget. Read the [environment notes](ENVIRONMENT.md) before changing dependencies.

## Default recipe

| Setting | Value |
| --- | --- |
| Backbone | Qwen2.5-VL-3B, SFT initialized |
| Training GPUs | 3 |
| Distributed setup | ZeRO-3 CPU offload, BF16 |
| Maximum updates | 1,000 |
| Video frame budget | At most 16 per video; shuffling reorders the same frames |
| Ordered / shuffled responses | 2 / 1 |
| Prompt / completion limits | 4,096 / 192 tokens |
| Maximum visual pixels | 401,408 |
| Learning rate / KL coefficient | 1e-6 / 0.04 |
| Weight decay / generation temperature | 0.01 / 1.0 |
| TE bonus cap / saturation margin | 0.50 / 0.25 |
| Positive-gap gate / answer eligibility | `g > 0` / `answer score > 0` |
| Token maximum raw weight / tau | 1.30 / 1.0 |
| Default checkpoint interval | 1,000 updates |

These are training settings. The manuscript's 8-, 16-, and 32-frame evaluations reuse trained checkpoints; they are not different training frame counts.

## Preflight and smoke tests

```bash
bash run_3b.sh preflight --config server_3b_config.local.json
bash run_3b.sh smoke --variant all_video --config server_3b_config.local.json
bash run_3b.sh smoke --variant selective --config server_3b_config.local.json
```

Preflight checks model/data paths, expected model structure, dependency APIs, video backend, visible GPUs, and available storage. It does not replace a forward/backward smoke test.

Each smoke run uses six mixed image/video records and two optimizer steps, without exporting a full checkpoint. `all_video` exercises video token weighting regardless of gap sign. A short `selective` run may contain no positive-gap group; do not fabricate a gap to force activation.

## Training and controls

```bash
CUDA_VISIBLE_DEVICES=0,1,3 bash run_3b.sh train --variant selective --config server_3b_config.local.json
```

Check GPU availability before launching. The code does not stop other GPU jobs. Every launch creates a fresh timestamped directory and refuses to overwrite an existing experiment.

| Variant | Answer scoring | Response-reward rule | Token weights |
| --- | --- | --- | --- |
| `selective` | Soft | TE Rule | Probability contrast, positive-gap video groups |
| `all_video` | Soft | TE Rule | Probability contrast, every video group |
| `no_rw` | Soft | TE Rule | Unit weights |
| `baseline` | Baseline task scorer | Inherited T-GRPO | Unit weights |
| `lexicon` | Soft | TE Rule | Legacy lexical control, not the main method |

Run controls separately with matched initialization and settings. Images retain unit weights in every variant. Disabling Token RW does not disable the proposed reward rule in `no_rw`. Conversely, the `baseline` variant explicitly disables both proposed switches.

The launcher sets `use_vllm=false`. The retained optional vLLM backend is not a validated probability-based Token RW implementation and is not selected by these commands. All probability-RW scoring uses the local non-vLLM trainer.

## Saving and resuming

Default training saves a full resumable checkpoint at step 1,000 and exports the final model. An earlier interruption has no checkpoint unless intermediate saving was explicitly enabled:

```bash
SAVE_STEPS=200 bash run_3b.sh train --variant selective --config server_3b_config.local.json
```

Full ZeRO checkpoints and final model export require substantial disk space and host RAM. Check both before training. To resume a compatible intermediate checkpoint:

```bash
bash run_3b.sh train --variant selective --config server_3b_config.local.json \
    --resume /path/to/run/checkpoint-200
```

The variant, configuration, dataset hash, and project Python source hashes must match the original manifest. Removing unused Python files or adding source comments changes those hashes and can cause an older run to be rejected. A completed step-1,000 run does not need to be resumed.

Paths and GPU IDs can be overridden using `MODEL_PATH`, `DATA_ROOT`, `DATASET_PATH`, `OUTPUT_ROOT`, and `CUDA_VISIBLE_DEVICES`. Parameter overrides include `SEED`, `MAX_STEPS`, `SAVE_STEPS`, `TE_ALPHA`, `TE_MARGIN`, `RW_MAX_WEIGHT`, `RW_TAU`, `MAX_PROMPT_LENGTH`, `MAX_COMPLETION_LENGTH`, `MAX_PIXELS`, and `ATTN_IMPLEMENTATION`. Record any deviation from the default recipe when reporting results.

## Diagnostics and tests

Each run records its arguments, dataset/source hashes, environment checks, console output, and exit status. Per-rank `rw_diagnostics` files record gap signs, activation, candidate and applied weights, response advantages, and relative coefficient variation. Optional token traces contain generated token IDs and cross-view log probabilities. Keep these run outputs private unless separately reviewed for release.

An average raw weight above one does not establish variation within a response. Inspect relative weight variation; inactive groups must have unit applied weights. Sensitivity weights redistribute both policy-loss and KL contributions, not response-advantage signs.

```bash
python -B -m unittest discover -s project/tests -p 'test_contrast_rw.py' -v
python -B -m unittest discover -s tests -v
python -B launch_3b.py train --dry-run
```

Synthetic tests require PyTorch but not pretrained weights. `--dry-run` prints a command without loading a model or launching training. CPU tests do not validate distributed GPU execution or benchmark performance. Validation evidence for this snapshot is documented separately in [Environment](ENVIRONMENT.md).
