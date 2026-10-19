# TE-CGRPO

## Temporal-Contrast-Guided Credit Assignment for Video MLLMs

TE-CGRPO is a reinforcement-learning framework for video reasoning that coordinates **group selection, response-reward shaping, and token-loss allocation** through temporal contrast. This repository provides the **3B training implementation** with Qwen2.5-VL, including probability-based token reweighting and matched control variants.

[Method](#method) | [Manuscript results](#manuscript-results) | [Quick start](#quick-start) | [Training guide](docs/TRAINING.md)

## Overview

Video reasoning requires multimodal large language models (MLLMs) to connect actions, states, and events over time. Correct answers alone do not establish whether a model used temporal evidence, since static visual cues and language priors can also support correct predictions. A response-level reward further provides no direct indication of which generated positions are sensitive to video order.

TE-CGRPO builds on Group Relative Policy Optimization (GRPO) and Video-R1's ordered-versus-shuffled video comparison. Rather than applying a fixed temporal bonus and averaging all token losses uniformly, it combines graded answer scoring, positive-gap group selection, bounded reward shaping, and position-specific token weights. Cross-view scoring is confined to training, leaving the original inference pathway unchanged.

## Method

### 1. Answer scoring and temporal rewards

A shared policy initialized through supervised fine-tuning (SFT) generates responses to ordered and shuffled views of the **same sampled frames**. **Soft** applies the same task-specific answer scorer to both views, and their mean answer-score difference defines the group gap:

$$
g = \bar r_o - \bar r_c.
$$

Format and length rewards do not enter this gap. A positive gap selects a video group for Token RW. Within that group, **TE Rule** adds a bounded, gap-dependent bonus only to ordered responses with positive answer scores:

$$
b_{\mathrm{TE}}(g)=\alpha\min\!\left(\frac{\max(g,0)}{m},1\right).
$$

This reward adjustment occurs **before advantage normalization**. Shuffled responses estimate the gap; only ordered responses enter advantage normalization and the policy loss. Image inputs and non-positive-gap video groups retain the base rewards.

### 2. Fixed-prefix cross-view Token RW

A group-level gap does not identify order-sensitive positions within a response. **Token RW scores the same generated ordered response under both video views, keeping the question, target token IDs, and text prefixes identical.** It does not align independently generated ordered and shuffled answers.

For token position $t$ in response $i$, the pre-update rollout policy supplies ordered- and shuffled-view probabilities $p^o_{i,t}$ and $p^c_{i,t}$. Their absolute log-probability difference determines a bounded, detached weight:

$$
\delta_{i,t}=\left|\log p^o_{i,t}-\log p^c_{i,t}\right|,
\qquad
w_{i,t}=1+\lambda\,\operatorname{sg}\!\left(\frac{\delta_{i,t}}{\delta_{i,t}+\tau}\right).
$$

Here, $\operatorname{sg}$ denotes stop-gradient. The main configuration applies these weights only to **positive-gap video groups**. All other groups use unit weights. With $\lambda=0.30$ and $\tau=1.0$, raw weights lie within $[1,1.30]$. Every valid token remains in the objective, without a predefined temporal lexicon or hard token selection.

After advantage computation, each ordered-response loss is aggregated as

$$
\mathcal L_i=
\frac{\sum_t M_{i,t}w_{i,t}\ell_{i,t}}
{\sum_t M_{i,t}w_{i,t}},
$$

where $M_{i,t}$ is the valid-completion mask and $\ell_{i,t}$ contains the policy-loss term and Kullback-Leibler (KL) regularization. Response advantages retain their signs; token weights redistribute relative loss contributions. Cross-view sensitivity is not a correctness label for intermediate reasoning steps.

### Design summary

| Component | Training operation | Applied to |
| --- | --- | --- |
| Soft | Assigns graded, task-specific answer scores | Ordered and shuffled responses |
| TE Rule | Adds a bounded temporal bonus before advantage normalization | Positive-score ordered responses in positive-gap video groups |
| Token RW | Redistributes token-loss contributions after advantage computation | All ordered responses in positive-gap video groups |

Non-selected groups are retained in training. No extra model parameters or cross-view scoring are introduced at inference time.

## Manuscript results

The manuscript evaluates TE-CGRPO on **VSI-Bench, MMVU, MVBench, TempCompass, and Video-MME** at 3B and 7B scales. The table below reproduces the manuscript's matched local **3B comparison at 16 evaluation frames**, using a shared SFT initialization and a common evaluation protocol.

VSI-Bench reports its official Overall score; the other columns report accuracy. MMVU uses multiple-choice questions, and Video-MME excludes subtitles. Mean is the unweighted average of the five benchmark scores.

| Model | VSI-Bench | MMVU | MVBench | TempCompass | Video-MME | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-VL-3B-SFT | 31.82 | 56.80 | 52.98 | 62.63 | 47.26 | 50.30 |
| Video-R1-3B, local baseline | 33.01 | 54.88 | 57.18 | 63.97 | 48.00 | 51.41 |
| **TE-CGRPO-3B** | **34.74** | **60.48** | **57.78** | **64.12** | **49.67** | **53.36** |

TE-CGRPO-3B improves the five-benchmark mean by **1.95 points** over the local baseline at 16 frames. Component ablations favor the complete configuration. Restricting Token RW to positive-gap groups also yields a higher mean than reweighting every video group at each tested 3B evaluation budget of 8, 16, and 32 frames.

At 7B, the manuscript reports mean gains of **0.43** and **0.88 points** over the original March 2025 Video-R1 release at 16 and 32 evaluation frames, respectively. That comparison uses reported baseline scores and is distinct from the matched local 3B study.

These are **manuscript-reported results**, not measurements from a new run of this source snapshot. This release contains the 3B training path, not a 7B launcher or the complete benchmark evaluation pipeline. Prediction records, evaluation outputs, and checkpoints are not included.

## Quick start

Use a compatible Linux CUDA environment and obtain the SFT checkpoint, training mixture, and media separately. The default 3B recipe uses **3 GPUs with ZeRO-3 CPU offload**, at most **16 training frames**, $G=2$ ordered responses, $G_c=1$ shuffled response, and **1,000 optimizer updates**. Environment details are in [Environment](docs/ENVIRONMENT.md).

```bash
git clone https://github.com/chengY334/Selective-TE-CGRPO.git
cd Selective-TE-CGRPO
cp server_3b_config.json server_3b_config.local.json
```

Edit the four path fields in the local configuration using **absolute paths**, and select available GPUs. Keep this private configuration out of Git. Then run the checks before training:

```bash
bash run_3b.sh preflight --config server_3b_config.local.json
bash run_3b.sh smoke --variant all_video --config server_3b_config.local.json
bash run_3b.sh smoke --variant selective --config server_3b_config.local.json
bash run_3b.sh train --variant selective --config server_3b_config.local.json
```

See the [Training Guide](docs/TRAINING.md) for dataset format, control variants, checkpoint saving, resume requirements, and tests. By default, a resumable checkpoint is saved only at step 1,000; enable intermediate saving if needed.

## Code map

| Implementation | Location |
| --- | --- |
| Answer scoring and method switches | [`grpo.py`](project/src/r1-v/src/open_r1/grpo.py) |
| Temporal rewards, cross-view weights, and loss aggregation | [`grpo_trainer.py`](project/src/r1-v/src/open_r1/trainer/grpo_trainer.py) |
| Video frame sampling | [`vision_process.py`](project/src/qwen-vl-utils/src/qwen_vl_utils/vision_process.py) |
| 3B launch and configuration | [`launch_3b.py`](launch_3b.py), [`server_3b_config.json`](server_3b_config.json) |

## Acknowledgments and licensing

This implementation builds on [Video-R1](https://github.com/tulerfeng/Video-R1), the R1-V / Open-R1 training stack, and [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL). Original upstream copyright notices are retained. See [Third-Party Notices](THIRD_PARTY_NOTICES.md) for source attribution and licensing scope.
