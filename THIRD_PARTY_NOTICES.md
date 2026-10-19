# Third-party source notices

This source tree is derived from the author's local Video-R1 checkout. Exact upstream commit identities were not recorded in that checkout, so this document does not claim a byte-identical release or a verified commit pin.

## Video-R1 and the R1-V / Open-R1 training stack

- Source project: https://github.com/tulerfeng/Video-R1
- Bundled training subtree: `project/src/r1-v/`
- Upstream Apache-2.0 license: https://github.com/tulerfeng/Video-R1/blob/main/src/r1-v/LICENSE
- `setup.py` identifies its original R1-V and Hugging Face authors. Existing copyright and attribution headers remain in the distributed files.
- A copy of the Apache-2.0 text from the supplied local checkout is included in `LICENSES/Apache-2.0.txt`.

The TE-CGRPO version of `grpo.py` adds graded answer scoring and explicit method switches. `trainer/grpo_trainer.py` adds bounded temporal rewards, selective fixed-prefix contrast weighting, weighted loss aggregation, and diagnostics. `trainer/__init__.py` defers the optional vLLM import. The legacy optional backend and SFT entry point are retained for source compatibility, not advertised as validated probability-RW training paths.

## Qwen vision utilities

- Source project: https://github.com/QwenLM/Qwen2.5-VL
- Bundled subtree: `project/src/qwen-vl-utils/`
- The bundled package metadata declares Apache-2.0; the upstream license is available at https://github.com/QwenLM/Qwen2.5-VL/blob/main/LICENSE.
- Its Video-R1 sampling changes are retained, including the 16-frame cap and video pixel limits. Installing an unrelated version of `qwen-vl-utils` must not silently replace this sampler during a run.

## Local modifications and excluded assets

The root launcher, environment checks, and tests form the standalone 3B entry path. Public-release preparation removes old example jobs, obsolete packaging records, local logs, and private machine paths; it does not change the training objective.

No model parameters, training media, dataset records, benchmark inputs, or predictions are distributed. Their licenses and access terms are separate from the source licenses above. Upstream licensing does not determine a license for every original addition; the authors must designate any additional release license separately.
