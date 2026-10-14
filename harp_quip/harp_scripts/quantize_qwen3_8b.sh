#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
HESS_PATH="${HESS_PATH:-hess/qwen3_8b_6144}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-ckpt}"
EXPORT_DIR="${EXPORT_DIR:-hf}"
LOG_DIR="${LOG_DIR:-log/qwen}"
mkdir -p "$CHECKPOINT_DIR" "$EXPORT_DIR" "$LOG_DIR"

if [[ ! -f "$HESS_PATH/0_qkv.pt" ]]; then
  echo "Missing Hessians in $HESS_PATH; run hessians_qwen3_8b.sh first." >&2
  exit 1
fi

for mode in harp had; do
  name="qwen3_8b_2bit_${mode}"
  save_path="$CHECKPOINT_DIR/$name"
  export_path="$EXPORT_DIR/$name"
  if [[ -e "$export_path" ]]; then
    echo "Export already exists: $export_path. Choose a new EXPORT_DIR." >&2
    exit 1
  fi
  harp_args=()
  if [[ "$mode" == harp ]]; then
    harp_args=(
      --harp_b 8 --harp_max_b 8 --harp_passes 1
      --harp_ordering_mode stride --harp_fixed_mixer had_or_qr
      --harp_steps "${HARP_STEPS:-1600}" --harp_strategy proxy
      --harp_lr_u 0.03 --harp_lr_v 0.03
      --harp_hbd_mode ldl --harp_hbd_lambda "${HARP_HBD_LAMBDA:-3.0}"
      --harp_eval_every "${HARP_EVAL_EVERY:-50}"
      --harp_hbd_block 8 --harp_chunk_size "${HARP_CHUNK_SIZE:-1048576}"
      --harp_q_recompute_every 1
    )
  fi
  "$PYTHON" -m quantize_llama.quantize_finetune_qwen3 \
    --base_model "$BASE_MODEL" --hessian_path "$HESS_PATH" \
    --save_path "$save_path" --codebook E8P12 --scale_override 0.9 \
    --ft_train_mode --ft_epochs 0 --devset_size 384 --ft_valid_size 128 \
    --ctx_size "${CONTEXT_LENGTH:-4096}" --incoh_mode "$mode" \
    "${harp_args[@]}" "$@" 2>&1 | tee "$LOG_DIR/${name}_quantize.log"
  "$PYTHON" -m quantize_llama.hfize_qwen3 \
    --quantized_path "$save_path" --hf_output_path "$export_path" \
    2>&1 | tee "$LOG_DIR/${name}_export.log"
  if [[ "${RUN_EVAL:-1}" == 1 ]]; then
    bash harp_scripts/eval_qwen3_8b.sh "$export_path"
  fi
done
