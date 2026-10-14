#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHON="${PYTHON:-python}"
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
export HESS_PATH="${HESS_PATH:-hess/qwen3_8b_6144}"

hessians_complete() {
  "$PYTHON" - "$BASE_MODEL" "$HESS_PATH" <<'PY'
import sys
from pathlib import Path
from transformers import AutoConfig

config = AutoConfig.from_pretrained(sys.argv[1])
root = Path(sys.argv[2])
complete = all(
    (root / f"{layer}_{projection}.pt").is_file()
    for layer in range(config.num_hidden_layers)
    for projection in ("qkv", "o", "up", "down")
)
raise SystemExit(0 if complete else 1)
PY
}

if ! hessians_complete; then
  bash harp_scripts/hessians_qwen3_8b.sh
  hessians_complete
fi
bash harp_scripts/quantize_qwen3_8b.sh "$@"
