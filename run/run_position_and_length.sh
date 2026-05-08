#!/usr/bin/env bash
set -euo pipefail

export LIMIT="${LIMIT:-100}"

TYPE="${TYPE:-<256|512|1k|1.5k|2k>}"

case "$TYPE" in
  256)
    DATA_JSONL="${DATA_JSONL:-<PATH_TO_POSITION_256_DATA_DIR_OR_JSONL>}"
    ;;
  512)
    DATA_JSONL="${DATA_JSONL:-<PATH_TO_POSITION_512_DATA_DIR_OR_JSONL>}"
    ;;
  1k)
    DATA_JSONL="${DATA_JSONL:-<PATH_TO_POSITION_1K_DATA_DIR_OR_JSONL>}"
    ;;
  1.5k)
    DATA_JSONL="${DATA_JSONL:-<PATH_TO_POSITION_1_5K_DATA_DIR_OR_JSONL>}"
    ;;
  2k)
    DATA_JSONL="${DATA_JSONL:-<PATH_TO_POSITION_2K_DATA_DIR_OR_JSONL>}"
    ;;
  *)
    echo "[ERROR] TYPE must be one of: 256, 512, 1k, 1.5k, 2k"
    exit 1
    ;;
esac

OUT_ROOT_BASE="${OUT_ROOT_BASE:-model_outputs}"

BASE_URL="${BASE_URL:-<OPENAI_COMPATIBLE_BASE_URL>}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-<YOUR_OPENAI_COMPATIBLE_API_KEY>}"

SLEEP="${SLEEP:-0.2}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
RETRIES="${RETRIES:-4}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_P="${TOP_P:-0.95}"
RESUME="${RESUME:-1}"
CHECK_MODELS="${CHECK_MODELS:-1}"

TARGET_ARGS=()

TARGET_ARGS+=(--target "gpt-5.1-thinking|thinking|${BASE_URL}|OPENAI_API_KEY")
TARGET_ARGS+=(--target "gpt-4o|instant|${BASE_URL}|OPENAI_API_KEY")
TARGET_ARGS+=(--target "gpt-5.4|instant|${BASE_URL}|OPENAI_API_KEY")
TARGET_ARGS+=(--target "gpt-5.1|instant|${BASE_URL}|OPENAI_API_KEY")

python get_answer_position_and_length.py \
  --data_jsonl "$DATA_JSONL" \
  --out_dir "$OUT_ROOT_BASE" \
  --sleep "$SLEEP" \
  --max_tokens "$MAX_TOKENS" \
  --retries "$RETRIES" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --limit "$LIMIT" \
  --resume "$RESUME" \
  --check_models "$CHECK_MODELS" \
  "${TARGET_ARGS[@]}"

echo "[OK] outputs in: $OUT_ROOT_BASE/len/"
echo "layout: $OUT_ROOT_BASE/len/<len_tag>/<model>__<group>.jsonl"