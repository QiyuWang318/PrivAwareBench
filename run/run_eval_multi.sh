#!/usr/bin/env bash
set -euo pipefail

TYPE="${TYPE:-Multi-Exposure}"

OUTPUTS_ROOT="${OUTPUTS_ROOT:-model_outputs}"
OUTPUTS_DIR="${OUTPUTS_DIR:-${OUTPUTS_ROOT}/${TYPE}}"

OUT_DIR="${OUT_DIR:-evaluation/${TYPE}}"

BASE_URL="${BASE_URL:-<OPENAI_COMPATIBLE_BASE_URL>}"
export OPENAI_BASE_URL="$BASE_URL"
export OPENAI_API_KEY="${OPENAI_API_KEY:-<YOUR_OPENAI_COMPATIBLE_API_KEY>}"

EVALUATOR_MODEL="${EVALUATOR_MODEL:-gpt-5.2}"

INFER_PROC="${INFER_PROC:-20}"
SAVE_FREQ="${SAVE_FREQ:-50}"
RESUME="${RESUME:-1}"
GLOB="${GLOB:-*.jsonl}"

RETRIES="${RETRIES:-4}"
API_KEY_ENV="${API_KEY_ENV:-OPENAI_API_KEY}"

python privacy_eval_multi.py \
  --outputs_dir "$OUTPUTS_DIR" \
  --out_dir "$OUT_DIR" \
  --base_url "$BASE_URL" \
  --evaluator_model "$EVALUATOR_MODEL" \
  --infer_proc "$INFER_PROC" \
  --save_frequency "$SAVE_FREQ" \
  --resume "$RESUME" \
  --glob "$GLOB" \
  --retries "$RETRIES" \
  --api_key_env "$API_KEY_ENV"

echo "[OK] eval outputs in: $OUT_DIR"