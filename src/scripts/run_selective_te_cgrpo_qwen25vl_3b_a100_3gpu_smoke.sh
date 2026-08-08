#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/qwen25vl-3b-selective-te-cgrpo-smoke}"
export TRAINING_LOG_PATH="${TRAINING_LOG_PATH:-${OUTPUT_DIR}/training_log.txt}"
export MAX_STEPS="${MAX_STEPS:-1}"
export SAVE_STEPS="${SAVE_STEPS:-1}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-192}"
export MAX_PIXELS="${MAX_PIXELS:-200704}"
export REPORT_TO="${REPORT_TO:-none}"
export DEBUG_MODE="${DEBUG_MODE:-false}"

exec bash "${SCRIPT_DIR}/run_selective_te_cgrpo_qwen25vl_3b_a100_3gpu.sh"
