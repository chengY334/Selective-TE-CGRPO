#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_ROOT="${REPO_ROOT}/src/r1-v"

MODEL_PATH="${MODEL_PATH:-}"
DATA_ROOT="${DATA_ROOT:-}"
DATASET_PATH="${DATASET_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/qwen25vl-3b-selective-te-cgrpo}"
DS_CONFIG="${DS_CONFIG:-${TRAIN_ROOT}/local_scripts/zero3_cpu_offload.json}"
TRAINING_LOG_PATH="${TRAINING_LOG_PATH:-${OUTPUT_DIR}/training_log.txt}"
RESUME_ARGS=()

for required_var in MODEL_PATH DATA_ROOT DATASET_PATH; do
    if [[ -z "${!required_var}" ]]; then
        echo "${required_var} must be set before launching training." >&2
        exit 1
    fi
done

if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    if [[ ! -d "${RESUME_FROM_CHECKPOINT}" ]]; then
        echo "Resume checkpoint does not exist: ${RESUME_FROM_CHECKPOINT}" >&2
        exit 1
    fi
    RESUME_ARGS=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

for required_path in "${MODEL_PATH}" "${DATA_ROOT}" "${DATASET_PATH}" "${DS_CONFIG}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path does not exist: ${required_path}" >&2
        exit 1
    fi
done

cd "${TRAIN_ROOT}"
mkdir -p "${OUTPUT_DIR}"

export VIDEO_R1_DATA_ROOT="${DATA_ROOT}"
export DEBUG_MODE="${DEBUG_MODE:-false}"
export LOG_PATH="${LOG_PATH:-${OUTPUT_DIR}/debug.log}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# Three GPUs are all DeepSpeed training ranks. No vLLM worker is used.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}" \
python -m torch.distributed.run \
    --nproc_per_node="${NPROC_PER_NODE:-3}" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="${MASTER_PORT:-12365}" \
    src/open_r1/grpo.py \
    --output_dir "${OUTPUT_DIR}" \
    "${RESUME_ARGS[@]}" \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset_name "${DATASET_PATH}" \
    --reward_funcs soft_accuracy format \
    --deepspeed "${DS_CONFIG}" \
    --max_prompt_length "${MAX_PROMPT_LENGTH:-4096}" \
    --max_completion_length "${MAX_COMPLETION_LENGTH:-192}" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
    --learning_rate "${LEARNING_RATE:-1e-6}" \
    --lr_scheduler_type cosine \
    --weight_decay "${WEIGHT_DECAY:-0.01}" \
    --logging_steps "${LOGGING_STEPS:-1}" \
    --bf16 true \
    --gradient_checkpointing true \
    --attn_implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}" \
    --min_pixels "${MIN_PIXELS:-3136}" \
    --max_pixels "${MAX_PIXELS:-401408}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
    --max_steps "${MAX_STEPS:-1000}" \
    --seed "${SEED:-42}" \
    --data_seed "${DATA_SEED:-42}" \
    --run_name "${RUN_NAME:-Qwen2.5-VL-3B-Selective-TE-CGRPO-a030-d010-m015-G2-Gc1-3GPU}" \
    --save_steps "${SAVE_STEPS:-100}" \
    --save_only_model false \
    --temporal true \
    --len_control true \
    --fut_grpo true \
    --fut_token_weighting "${FUT_TOKEN_WEIGHTING:-true}" \
    --fut_token_weighting_scope "${FUT_TOKEN_WEIGHTING_SCOPE:-selective}" \
    --fut_temporal_alpha "${FUT_TEMPORAL_ALPHA:-0.30}" \
    --fut_gap_threshold "${FUT_GAP_THRESHOLD:-0.10}" \
    --fut_temporal_margin "${FUT_TEMPORAL_MARGIN:-0.15}" \
    --fut_min_correct_reward "${FUT_MIN_CORRECT_REWARD:-0.10}" \
    --fut_corruptions "${FUT_CORRUPTIONS:-shuffle}" \
    --fut_num_corruptions "${FUT_NUM_CORRUPTIONS:-1}" \
    --fut_negative_generations "${FUT_NEGATIVE_GENERATIONS:-1}" \
    --fut_token_max_weight "${FUT_TOKEN_MAX_WEIGHT:-1.30}" \
    --fut_context_weight "${FUT_CONTEXT_WEIGHT:-1.15}" \
    --fut_template_min_weight "${FUT_TEMPLATE_MIN_WEIGHT:-0.95}" \
    --report_to "${REPORT_TO:-none}" \
    --beta "${BETA:-0.04}" \
    --max_grad_norm "${MAX_GRAD_NORM:-5}" \
    --temperature "${TEMPERATURE:-1.0}" \
    --num_generations "${NUM_GENERATIONS:-2}" \
    2>&1 | tee "${TRAINING_LOG_PATH}"
