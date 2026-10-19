# TE-CGRPO: 3B Training Code

Training code for temporal-contrast-guided reward shaping and probability-based token reweighting with Qwen2.5-VL-3B. This release contains the 3B training path, its local video sampler, and synthetic unit tests. It does not contain model weights, datasets, benchmark predictions, or a complete five-benchmark evaluation pipeline.

## Method

A shared rollout policy generates responses to ordered and shuffled views of the same sampled video frames. Soft answer scores define the ordered-minus-shuffled group gap. TE Rule adds a bounded, gap-dependent bonus to eligible ordered responses before advantage normalization.

Token RW teacher-forces the **same ordered response with identical text prefixes** under both views. For each token it computes:

```text
delta = abs(log_p_ordered - log_p_shuffled)
w = 1 + 0.30 * delta / (delta + 1.0)
```

The weights are detached. The main variant applies them only to video groups with a positive answer-score gap; images and other groups retain unit weights. Each response loss is divided by its sum of valid-token weights. Weights apply to both the policy-loss and KL terms. Probability sensitivity is not a token-level correctness label.

## Configuration

The default recipe uses 3 training GPUs, ZeRO-3 CPU offload, BF16, 1,000 optimizer steps, at most 16 sampled frames per video, 2 ordered responses and 1 shuffled response, prompt/completion limits of 4,096/192 tokens, and maximum visual pixels of 401,408. TE coefficients are `alpha=0.50` and `m=0.25`; Token RW uses a maximum raw weight of `1.30` and `tau=1.0`.

Use a **shared SFT checkpoint**, not a completed baseline RL checkpoint. Obtain the Video-R1 training mixture and its media separately from the [upstream project](https://github.com/tulerfeng/Video-R1), following the applicable access and usage terms.

On a Linux CUDA server, activate your compatible environment, then create a private configuration:

```bash
cp server_3b_config.json server_3b_config.local.json
```

Edit `model_path`, `data_root`, `dataset_path`, `output_root`, and `cuda_visible_devices` in the local copy. Use absolute paths for all four path fields: distributed workers run from the training subdirectory, so relative paths can resolve differently from launcher paths. The tracked configuration contains placeholders, not a runnable server-specific configuration. The local copy is ignored by Git; do not upload it through the GitHub web interface either.

Dataset input may be a JSON array or JSONL. Each record must provide `problem`, `solution`, `data_type`, `problem_type`, and `path`. `data_type` is `image` or `video`; multiple-choice records also require `options`. Media paths must be absolute or resolve relative to `data_root`.

See [the environment notes](docs/ENVIRONMENT.md) before installing or changing dependencies. The default non-vLLM path does not require a dedicated generation GPU.

## Preflight and smoke tests

```bash
bash run_3b.sh preflight --config server_3b_config.local.json
bash run_3b.sh smoke --variant all_video --config server_3b_config.local.json
bash run_3b.sh smoke --variant selective --config server_3b_config.local.json
```

Smoke runs use six mixed image/video records and two optimizer steps without exporting a full checkpoint. `all_video` exercises video token weighting regardless of gap sign. A short `selective` smoke run may contain no positive-gap group; this does not by itself indicate a gate failure.

## Training and controls

```bash
CUDA_VISIBLE_DEVICES=0,1,3 bash run_3b.sh train --variant selective --config server_3b_config.local.json
```

GPU IDs are physical IDs before CUDA remapping. Check that they are available before launching; this code does not stop other GPU jobs. Each launch creates a new timestamped output directory rather than overwriting an old experiment.

| Variant | Answer scoring | Response-reward rule | Token weights |
| --- | --- | --- | --- |
| `selective` | Soft | TE Rule | Probability contrast, positive-gap video groups |
| `all_video` | Soft | TE Rule | Probability contrast, every video group |
| `no_rw` | Soft | TE Rule | Unit weights |
| `baseline` | Baseline task scorer | Inherited T-GRPO | Unit weights |
| `lexicon` | Soft | TE Rule | Legacy lexical control, not the main method |

Run controls separately using the same SFT initialization and matched settings. The launcher explicitly sets `use_vllm=false`. The retained optional vLLM backend is not a validated implementation of probability-based Token RW.

By default, checkpoints are saved at step 1,000 only. An earlier interruption has no resumable checkpoint unless intermediate saving was enabled, for example:

```bash
SAVE_STEPS=200 bash run_3b.sh train --variant selective --config server_3b_config.local.json
```

Full ZeRO checkpoints and the final model export require substantial disk space and host RAM. Check storage before training. To resume, pass `--resume /path/to/run/checkpoint-N` with the same configuration, source tree, variant, and dataset. Source changes, including removing unused Python files, change the recorded source hashes and can cause resume checks to reject an older run.

## Diagnostics and tests

Each run records configuration, dataset/source hashes, software checks, and exit status. Per-rank diagnostics record gap signs, actual versus candidate token weights, response advantages, and relative weight variation. Optional token traces contain generated token IDs and log probabilities. These are private run outputs, not files to commit to the public repository.

```bash
python -B -m unittest discover -s project/tests -p 'test_contrast_rw.py' -v
python -B -m unittest discover -s tests -v
python -B launch_3b.py train --dry-run
```

Synthetic tests need PyTorch but not model weights. `--dry-run` does not load a model or start a training process. CPU tests do not validate distributed training or benchmark performance.

## Release scope and attribution

The available server log records successful two-step smoke runs for both scopes and a main run interrupted after nine completed updates. It does **not** establish completion of a 1,000-step run or reproduce manuscript benchmark scores. No performance numbers are asserted for this release.

Original upstream copyright headers are retained. See [third-party notices](THIRD_PARTY_NOTICES.md) and [Apache-2.0 license text](LICENSES/Apache-2.0.txt). A separate release license for original TE-CGRPO additions has not yet been designated by the authors; the bundled third-party license is not a claim that all contributions share one license.
