#!/usr/bin/env bash
set -euo pipefail

TYPE="${TYPE:-<token|password|personal_id|financial>}"
DIF="${DIF:-standard}"

case "$TYPE" in
  token)
    DATASET_JSONL="${DATASET_JSONL:-DATA/Standard/token.jsonl}"
    BASE_TYPE_DIR="token"
    ;;
  password)
    DATASET_JSONL="${DATASET_JSONL:-DATA/Standard/password.jsonl}"
    BASE_TYPE_DIR="password"
    ;;
  personal_id)
    DATASET_JSONL="${DATASET_JSONL:-DATA/Standard/personal_id.jsonl}"
    BASE_TYPE_DIR="personal_id"
    ;;
  financial)
    DATASET_JSONL="${DATASET_JSONL:-DATA/Standard/financial.jsonl}"
    BASE_TYPE_DIR="financial"
    ;;
  *)
    echo "[ERROR] TYPE must be one of: token, password, personal_id, financial"
    exit 1
    ;;
esac

case "$DIF" in
  standard)
    TYPE_DIR="${BASE_TYPE_DIR}"
    ;;
  ass)
    TYPE_DIR="${BASE_TYPE_DIR}_ass"
    ;;
  fake)
    TYPE_DIR="${BASE_TYPE_DIR}_fake"
    ;;
  *)
    echo "[ERROR] DIF must be one of: standard, ass, fake"
    exit 1
    ;;
esac

OUTPUTS_ROOT="${OUTPUTS_ROOT:-model_outputs/Standard}"
OUTPUTS_DIR="${OUTPUTS_ROOT}/${TYPE_DIR}"

OUT_ROOT="${OUT_ROOT:-evaluation/Standard}"
OUT_DIR="${OUT_ROOT}/${TYPE_DIR}"

BASE_URL="${BASE_URL:-<OPENAI_COMPATIBLE_BASE_URL>}"
export OPENAI_BASE_URL="$BASE_URL"
export OPENAI_API_KEY="${OPENAI_API_KEY:-<YOUR_OPENAI_COMPATIBLE_API_KEY>}"

EVALUATOR_MODEL="${EVALUATOR_MODEL:-gpt-4o-mini}"

INFER_PROC="${INFER_PROC:-20}"
SAVE_FREQ="${SAVE_FREQ:-50}"
RESUME="${RESUME:-1}"
GLOB="${GLOB:-*.jsonl}"

python eval.py \
  --dataset_jsonl "$DATASET_JSONL" \
  --outputs_dir "$OUTPUTS_DIR" \
  --out_dir "$OUT_DIR" \
  --base_url "$BASE_URL" \
  --evaluator_model "$EVALUATOR_MODEL" \
  --infer_proc "$INFER_PROC" \
  --save_frequency "$SAVE_FREQ" \
  --resume "$RESUME" \
  --glob "$GLOB"

echo "[OK] eval outputs in: $OUT_DIR"