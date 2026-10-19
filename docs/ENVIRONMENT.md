# Environment and validation record

## Observed Linux environment

These versions were printed by the preflight process in the supplied 2026-10-19 server log. This is an observation, **not** a complete dependency lockfile or a verified fresh-install recipe.

| Component | Logged version |
| --- | --- |
| Python | 3.10.20 |
| PyTorch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| Transformers | 4.49.0.dev0 |
| Accelerate | 1.13.0 |
| TRL | 0.16.0 |
| DeepSpeed | 0.15.4 |
| datasets | 4.8.3 |
| NLTK | 3.9.3 |
| Video reader | decord |
| Hardware | 3 x NVIDIA A100-SXM4-40GB |

The log reports that FlashAttention, ROUGE scoring, the local Qwen sampler, model loading, and CUDA were usable. It does not record all package versions. In particular, the exact Transformers development commit, FlashAttention version, decord version, and complete dependency provenance remain unspecified. Do not present this table as an exact reproducible environment lock.

## Preparing an environment

Prefer a known-working Linux environment with the observed API versions. The launcher adds the local `r1-v` and `qwen-vl-utils` source directories to `PYTHONPATH`; an editable install is not needed to select those local sources. Runtime imports also require dependencies such as Pillow, requests, packaging, ROUGE scoring, and a working video reader. The `sft_video.py` entry point has additional dependencies and is not covered by the RL smoke result.

Do not blindly upgrade PyTorch, Transformers, TRL, DeepSpeed, or CUDA to the latest versions: this trainer uses the older GRPO API. `server_preflight.py` checks required APIs and visible hardware before training. A successful preflight does not replace a real forward/backward smoke test.

The retained `project/src/r1-v/setup.py` is upstream packaging metadata, not a complete environment specification. The old Qwen Rye lockfiles and Python 3.8 selector were removed because they do not describe the tested Python 3.10 training environment.

To improve environment reproducibility, export the actual working server environment and record any source-installed dependency commits. Review an export for private URLs, access tokens, and local paths before publishing it. A raw `requirements-server.txt` is ignored by this repository for that reason.

## Validation boundaries

- Synthetic CPU tests cover contrast weights, gating, diagnostics, and launcher arguments; they do not load the pretrained model.
- The supplied server log shows both two-step smoke variants completing and a main run reaching nine completed updates before a termination signal.
- A completed 1,000-step training run and five-benchmark evaluation are not demonstrated by that log.
- Original machine-specific logs and backup metadata are not part of the public tree.
