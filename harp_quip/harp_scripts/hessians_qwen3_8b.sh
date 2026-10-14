#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
HESS_PATH="${HESS_PATH:-hess/qwen3_8b_6144}"
LOG_DIR="${LOG_DIR:-log/qwen}"
mkdir -p "$HESS_PATH" "$LOG_DIR"

"$PYTHON" -m quantize_llama.hessian_offline_qwen3 \
  --base_model "$BASE_MODEL" \
  --save_path "$HESS_PATH" \
  --batch_size "${HESS_BATCH_SIZE:-1}" \
  --devset_size "${HESS_SAMPLES:-6144}" \
  --ctx_size "${CONTEXT_LENGTH:-4096}" \
  --chunk_size "${HESS_CHUNK_SIZE:-256}" \
  --sample_proc "${SAMPLE_PROCESSES:-4}" \
  "$@" 2>&1 | tee "$LOG_DIR/hessians_qwen3_8b.log"
