# TE-CGRPO modifications: Soft answer scoring and explicit reward/RW controls.
# Standalone 3B loading and smoke/export handling are included in this version.
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from trainer import Qwen2VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from datasets import Dataset, DatasetDict

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy',
            'task_aligned_accuracy', 'soft_accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'task_aligned_accuracy', 'soft_accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    temporal: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using temporal GRPO"},
    )
    len_control: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using length reward"},
    )
    fut_grpo: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable Focused and Unhackable Temporal GRPO reward shaping."},
    )
    fut_token_weighting: Optional[bool] = field(
        default=False,
        metadata={"help": "Independent Token RW switch; set False for a pure baseline, even when fut_grpo=False."},
    )
    fut_token_weighting_method: str = field(
        default="contrast",
        metadata={"help": "contrast: same-completion cross-view sensitivity; lexicon: legacy weights."},
    )
    fut_token_sensitivity_tau: float = field(
        default=1.0,
        metadata={"help": "Positive sensitivity scale in natural-log units (pilot default, not tuned)."},
    )
    fut_token_weighting_scope: Optional[str] = field(
        default="selective",
        metadata={
            "help": (
                "Token-reweighting scope: 'selective' applies it only to video groups "
                "whose answer-score gap is positive, independently of the bonus; 'all_video' "
                "applies it to every ordered video group. Image samples remain unweighted."
            )
        },
    )
    fut_temporal_alpha: Optional[float] = field(
        default=0.3,
        metadata={"help": "Maximum temporal-margin bonus added to correct ordered-video completions."},
    )
    fut_temporal_margin: Optional[float] = field(
        default=0.15,
        metadata={"help": "Positive-gap width over which the selective temporal bonus ramps to its maximum."},
    )
    fut_gap_threshold: Optional[float] = field(
        default=0.0,
        metadata={
            "help": (
                "Compatibility option fixed to 0.0 by the final objective; nonzero "
                "values are rejected by the trainer."
            )
        },
    )
    fut_min_correct_reward: Optional[float] = field(
        default=0.0,
        metadata={
            "help": (
                "Compatibility option fixed to 0.0; the temporal bonus is applied "
                "exactly when the ordered response has positive soft reward."
            )
        },
    )
    fut_corruptions: Optional[str] = field(
        default="shuffle",
        metadata={"help": "Comma-separated temporal corruptions: shuffle, reverse, drop_middle, edges."},
    )
    fut_num_corruptions: Optional[int] = field(
        default=1,
        metadata={"help": "Number of temporal corruptions sampled per video prompt."},
    )
    fut_negative_generations: Optional[int] = field(
        default=0,
        metadata={"help": "Generations per corrupted view. 0 means max(1, num_generations // (2*num_corruptions))."},
    )
    fut_token_max_weight: Optional[float] = field(
        default=1.30,
        metadata={"help": "Maximum temporal-evidence token loss weight."},
    )
    fut_context_weight: Optional[float] = field(
        default=1.15,
        metadata={"help": "Loss weight for tokens adjacent to a matched temporal-evidence token."},
    )
    fut_template_min_weight: Optional[float] = field(
        default=0.95,
        metadata={"help": "Minimum loss weight for boilerplate reasoning tokens."},
    )



def accuracy_reward(completions, solution, **kwargs):
    
    def extract_answer(text):
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def normalize_number(num_str):
        try:
            num_str = num_str.replace(',', '')
            return float(num_str)
        except Exception as e:
            print(f"Error converting '{num_str}' to float: {e}")
            return None

    def wer(reference, hypothesis):
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        m = len(ref_words)
        n = len(hyp_words)
        d = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1):
            d[i][0] = i
        for j in range(n+1):
            d[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
        return d[m][n] / max(1, m)


    def compute_rouge_score(reference, hypothesis, use_stemmer=True):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
        scores = scorer.score(reference, hypothesis)
        average_fmeasure = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
        return average_fmeasure
    

    question_type = kwargs['problem_type'][0]
    
    contents = [completion[0]["content"] for completion in completions]
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []

    for content, sol in zip(contents, solution):
    
        try:
            output_ans = extract_answer(content)
            gt_ans = extract_answer(sol)
            if question_type == "multiple choice":
                reward = 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
            elif question_type == "numerical":
                gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
                out_has_decimal = ("." in output_ans) or ("," in output_ans)
                if gt_has_decimal != out_has_decimal:
                    reward = 0.0
                else:
                    gt_number = normalize_number(gt_ans)
                    out_number = normalize_number(output_ans)
                    if gt_number is None or out_number is None:
                        reward = 0.0
                    else:
                        reward = 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
            elif question_type == "OCR":
                error_rate = wer(gt_ans, output_ans)
                reward = 1 - error_rate
                reward = max(0.0, min(1.0, reward))
            elif question_type == "free-form":
                score = compute_rouge_score(gt_ans, output_ans)
                reward = max(0.0, min(1.0, score))
            elif question_type == "regression":
                gt_number = normalize_number(gt_ans)
                out_number = normalize_number(output_ans)
                if gt_number is None or out_number is None:
                    reward = 0.0
                rel_diff = (abs(out_number - gt_number) + 1e-9) / (abs(gt_number) + 1e-9)
                rel_diff = min(1.0, max(0.0, rel_diff))
                reward = 1 - rel_diff
            else:
                reward = 0.0
        except Exception as e:
            print(f"Error in reward_fn for question_type '{question_type}': {e}")
            reward = 0.0
    
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
            
    return rewards


def task_aligned_accuracy_reward(completions, solution, **kwargs):
    """Task reward aligned with the metrics used by the evaluation pipeline.

    Multiple-choice outputs receive strict exact-match accuracy. Regression
    outputs receive the same mean-relative-accuracy (MRA) score used by
    ``eval_bench.py``. Other question types retain the existing task metric.

    This function is intended to be the first reward function in selective
    TE-CGRPO, so ordered-vs-corrupted gaps, temporal-bonus eligibility, and
    length-reward eligibility are all based on the benchmark-aligned score.
    """

    def extract_answer(text):
        match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def normalize_number(num_str):
        try:
            return float(num_str.replace(",", "").strip())
        except Exception:
            return None

    def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):
        relative_error = abs(pred - target) / (abs(target) + 1e-8)
        threshold_count = int(round((end - start) / interval)) + 1
        thresholds = [start + index * interval for index in range(threshold_count)]
        return sum(relative_error < (1.0 - threshold) for threshold in thresholds) / len(thresholds)

    question_type = kwargs["problem_type"][0]
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, solution):
        try:
            output_ans = extract_answer(content)
            gt_ans = extract_answer(sol)

            if question_type == "multiple choice":
                reward = 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
            elif question_type == "numerical":
                gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
                out_has_decimal = ("." in output_ans) or ("," in output_ans)
                gt_number = normalize_number(gt_ans)
                out_number = normalize_number(output_ans)
                reward = (
                    1.0
                    if gt_has_decimal == out_has_decimal
                    and gt_number is not None
                    and out_number is not None
                    and round(gt_number, 2) == round(out_number, 2)
                    else 0.0
                )
            elif question_type == "regression":
                gt_number = normalize_number(gt_ans)
                out_number = normalize_number(output_ans)
                reward = (
                    mean_relative_accuracy(out_number, gt_number)
                    if gt_number is not None and out_number is not None
                    else 0.0
                )
            else:
                reward = accuracy_reward(
                    [[{"content": content}]],
                    [sol],
                    problem_type=[question_type],
                )[0]
        except Exception as error:
            print(f"Error in task_aligned_accuracy_reward for question_type '{question_type}': {error}")
            reward = 0.0

        rewards.append(max(0.0, min(1.0, reward)))

    return rewards


def soft_accuracy_reward(completions, solution, **kwargs):
    """Accuracy reward with small partial credit for parseable but non-canonical answers.

    This is intentionally conservative for single-choice video QA: exact option letters still
    receive 1.0, malformed answers receive at most partial credit, and wrong choices remain 0.
    The purpose is to reduce sparse-reward variance for small 3B video reasoning models.
    """

    def extract_answer(text):
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def normalize_choice(text):
        text = text.strip().upper()
        text = re.sub(r"^[\(\[\{]\s*([A-Z])\s*[\)\]\}]\.?$", r"\1", text)
        text = re.sub(r"^OPTION\s+([A-Z])\.?$", r"\1", text)
        text = re.sub(r"^([A-Z])[\.\):\s]*$", r"\1", text)
        return text

    def choice_letters(text):
        text = text.upper()
        return re.findall(r"(?<![A-Z])([A-Z])(?![A-Z])", text)

    def normalize_number(num_str):
        try:
            return float(num_str.replace(',', '').strip())
        except Exception:
            return None

    def numeric_soft_score(gt_ans, output_ans):
        gt_number = normalize_number(gt_ans)
        out_number = normalize_number(output_ans)
        if gt_number is None or out_number is None:
            return 0.0
        if round(gt_number, 2) == round(out_number, 2):
            return 1.0
        rel_diff = abs(out_number - gt_number) / (abs(gt_number) + 1e-9)
        if rel_diff <= 0.01:
            return 0.8
        if rel_diff <= 0.05:
            return 0.5
        if rel_diff <= 0.10:
            return 0.25
        return 0.0

    question_type = kwargs['problem_type'][0]
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, solution):
        try:
            output_ans = extract_answer(content)
            gt_ans = extract_answer(sol)
            if question_type == "multiple choice":
                output_norm = normalize_choice(output_ans)
                gt_norm = normalize_choice(gt_ans)
                if output_norm == gt_norm:
                    reward = 1.0
                else:
                    letters = choice_letters(output_ans)
                    if letters and letters[0] == gt_norm:
                        reward = 0.6
                    elif gt_norm in letters:
                        reward = 0.2
                    else:
                        reward = 0.0
            elif question_type in ["numerical", "regression"]:
                reward = numeric_soft_score(gt_ans, output_ans)
            else:
                reward = accuracy_reward([[{"content": content}]], [sol], problem_type=[question_type])[0]
        except Exception as e:
            print(f"Error in soft_accuracy_reward for question_type '{question_type}': {e}")
            reward = 0.0
        rewards.append(max(0.0, min(1.0, reward)))

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "task_aligned_accuracy": task_aligned_accuracy_reward,
    "soft_accuracy": soft_accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Match the requested mixed precision at load time rather than loading FP32
    # parameters unnecessarily. Explicit model_init_kwargs remain authoritative.
    training_args.model_init_kwargs = dict(training_args.model_init_kwargs or {})
    if training_args.bf16:
        training_args.model_init_kwargs.setdefault("torch_dtype", "bfloat16")
    elif training_args.fp16:
        training_args.model_init_kwargs.setdefault("torch_dtype", "float16")
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        dataset =  DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)


    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    
    QUESTION_TEMPLATE = (
        "{Question}\n"
        "Please think about this question as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
        "It's encouraged to include self-reflection or verification in the reasoning process. "
        "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."
    )

    TYPE_TEMPLATE = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
        "free-form": " Please provide your text answer within the <answer> </answer> tags.",
        "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
    }

    def make_conversation_image(example):
        
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }
    
        
    def make_conversation_video(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
    }
        
    def make_conversation_image_and_video(example):
        if example["problem_type"] == 'multiple choice':
            question = example['problem'] + "Options:\n"
            for op in example["options"]:
                question += op + "\n"
        else:
            question = example['problem']

        
        msg ={
            "prompt": 
               [{
                    "role": "user",
                    "content": [
                        {
                            "type": example['data_type'],
                            # example['data_type']: os.getcwd() + "/Video-R1-data" + example['path'][1:]
                        },
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[example['problem_type']]
                        }
                        ]
                }]
            }
        
        return msg

    
    dataset = dataset.map(make_conversation_image_and_video)

    
    if training_args.use_vllm and script_args.fut_token_weighting and script_args.fut_token_weighting_method == "contrast":
        raise ValueError("Contrast Token RW currently requires --use_vllm False; vLLM does not implement cross-view scoring.")
    if training_args.use_vllm:
        from trainer import Qwen2VLGRPOVLLMTrainerModified
        trainer_cls = Qwen2VLGRPOVLLMTrainerModified
    else:
        trainer_cls = Qwen2VLGRPOTrainer
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        script_args=script_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )
    
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Save and push to hub
    if os.environ.get("TE_CGRPO_SMOKE") == "1":
        print("Smoke run finished; full model/optimizer checkpoint export is disabled.")
    else:
        trainer.save_model(training_args.output_dir)
        trainer.save_state()
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
