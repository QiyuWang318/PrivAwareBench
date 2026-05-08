#!/usr/bin/env bash
set -euo pipefail

export LIMIT="${LIMIT:-300}"

TYPE="${TYPE:-<token|password|personal_id|financial>}"

case "$TYPE" in
  token)
    DATA_JSONL="${DATA_JSONL:-DATA/Standard/token.jsonl}"
    ;;
  password)
    DATA_JSONL="${DATA_JSONL:-DATA/Standard/password.jsonl}"
    ;;
  personal_id)
    DATA_JSONL="${DATA_JSONL:-DATA/Standard/personal_id.jsonl}"
    ;;
  financial)
    DATA_JSONL="${DATA_JSONL:-DATA/Standard/financial.jsonl}"
    ;;
  *)
    echo "[ERROR] TYPE must be one of: token, password, personal_id, financial"
    exit 1
    ;;
esac

OUT_ROOT_BASE="${OUT_ROOT_BASE:-model_outputs/Standard}"
OUT_ROOT="${OUT_ROOT_BASE}/${TYPE}"

BASE_URL="${BASE_URL:-<OPENAI_COMPATIBLE_BASE_URL>}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-<YOUR_OPENAI_COMPATIBLE_API_KEY>}"

TARGET_ARGS=()

TARGET_ARGS+=(--target "gpt-5.1-thinking|thinking|${BASE_URL}|OPENAI_API_KEY")
TARGET_ARGS+=(--target "gpt-4o|instant
|${BASE_URL}|OPENAI_API_KEY")
TARGET_ARGS+=(--target "gpt-5.4|instant|${BASE_URL}|OPENAI_API_KEY")
TARGET_ARGS+=(--target "gpt-5.1|instant|${BASE_URL}|OPENAI_API_KEY")

mkdir -p "$OUT_ROOT"

python get_answer.py \
  --data_jsonl "$DATA_JSONL" \
  --out_dir "$OUT_ROOT" \
  --sleep "${SLEEP:-0.2}" \
  --max_tokens "${MAX_TOKENS:-8192}" \
  --retries "${RETRIES:-4}" \
  --temperature "${TEMPERATURE:-0.2}" \
  --top_p "${TOP_P:-0.95}" \
  --limit "${LIMIT:-0}" \
  --resume "${RESUME:-1}" \
  --check_models "${CHECK_MODELS:-1}" \
  "${TARGET_ARGS[@]}"

echo "[OK] outputs in: $OUT_ROOT/"