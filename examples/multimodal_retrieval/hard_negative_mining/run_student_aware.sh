#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEXUS_ROOT="${NEXUS_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

: "${TEACHER_MODEL:?Set TEACHER_MODEL to a teacher embedding model path.}"
: "${STUDENT_MODEL:?Set STUDENT_MODEL to a student base model or LoRA adapter path.}"
: "${INPUT_TRAIN_JSONL:?Set INPUT_TRAIN_JSONL to a Nexus train.jsonl path.}"

OUTPUT_DIR="${OUTPUT_DIR:-$NEXUS_ROOT/outputs/hard_negative_mining/student_aware}"
CANDIDATE_TRAIN_JSONL="${CANDIDATE_TRAIN_JSONL:-}"
CANDIDATE_SOURCE="${CANDIDATE_SOURCE:-all_pos}"
DEVICES="${DEVICES:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TOP_K="${TOP_K:-200}"
NUM_HARD_NEGATIVES="${NUM_HARD_NEGATIVES:-3}"
QUERY_MAX_LEN="${QUERY_MAX_LEN:-2048}"
PASSAGE_MAX_LEN="${PASSAGE_MAX_LEN:-4096}"
STUDENT_QUERY_MAX_LEN="${STUDENT_QUERY_MAX_LEN:-$QUERY_MAX_LEN}"
STUDENT_PASSAGE_MAX_LEN="${STUDENT_PASSAGE_MAX_LEN:-$PASSAGE_MAX_LEN}"
SCORE_BLOCK_SIZE="${SCORE_BLOCK_SIZE:-4096}"
SCORE_DEVICE="${SCORE_DEVICE:-}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
STUDENT_TORCH_DTYPE="${STUDENT_TORCH_DTYPE:-$TORCH_DTYPE}"
BACKBONE_LOAD_STRATEGY="${BACKBONE_LOAD_STRATEGY:-prefer_base_model}"
STUDENT_BACKBONE_LOAD_STRATEGY="${STUDENT_BACKBONE_LOAD_STRATEGY:-$BACKBONE_LOAD_STRATEGY}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
MAX_CANDIDATE_EXAMPLES="${MAX_CANDIDATE_EXAMPLES:-0}"
FILTER_DUPLICATE_TEXT="${FILTER_DUPLICATE_TEXT:-false}"
FILTER_DUPLICATE_MEDIA="${FILTER_DUPLICATE_MEDIA:-true}"
REQUIRE_EXISTING_MEDIA="${REQUIRE_EXISTING_MEDIA:-true}"

mkdir -p "$OUTPUT_DIR"
cd "$NEXUS_ROOT"
export PYTHONPATH="$NEXUS_ROOT:${PYTHONPATH:-}"
PYTHON="${PYTHON:-python3}"

args=(
  --mode student_aware
  --teacher_model "$TEACHER_MODEL"
  --student_model "$STUDENT_MODEL"
  --input_train_jsonl "$INPUT_TRAIN_JSONL"
  --output_train_jsonl "$OUTPUT_DIR/train.student_aware_hardneg.jsonl"
  --max_examples "$MAX_EXAMPLES"
  --max_candidate_examples "$MAX_CANDIDATE_EXAMPLES"
  --candidate_source "$CANDIDATE_SOURCE"
  --top_k "$TOP_K"
  --num_hard_negatives "$NUM_HARD_NEGATIVES"
  --batch_size "$BATCH_SIZE"
  --devices "$DEVICES"
  --query_max_len "$QUERY_MAX_LEN"
  --passage_max_len "$PASSAGE_MAX_LEN"
  --student_query_max_len "$STUDENT_QUERY_MAX_LEN"
  --student_passage_max_len "$STUDENT_PASSAGE_MAX_LEN"
  --score_block_size "$SCORE_BLOCK_SIZE"
  --torch_dtype "$TORCH_DTYPE"
  --student_torch_dtype "$STUDENT_TORCH_DTYPE"
  --backbone_load_strategy "$BACKBONE_LOAD_STRATEGY"
  --student_backbone_load_strategy "$STUDENT_BACKBONE_LOAD_STRATEGY"
  --report_path "$OUTPUT_DIR/report.json"
  --manifest_path "$OUTPUT_DIR/manifest.json"
  --audit_sample_path "$OUTPUT_DIR/audit.jsonl"
  --audit_sample_size "100"
  --write_scores
)

if [[ -n "$CANDIDATE_TRAIN_JSONL" ]]; then
  args+=(--candidate_train_jsonl "$CANDIDATE_TRAIN_JSONL")
fi
if [[ -n "$SCORE_DEVICE" ]]; then
  args+=(--score_device "$SCORE_DEVICE")
fi
if [[ "$FILTER_DUPLICATE_TEXT" == "true" ]]; then
  args+=(--filter_duplicate_text)
fi
if [[ "$FILTER_DUPLICATE_MEDIA" == "true" ]]; then
  args+=(--filter_duplicate_media)
fi
if [[ "$REQUIRE_EXISTING_MEDIA" != "true" ]]; then
  args+=(--no-require_existing_media)
fi

"$PYTHON" -m Nexus.data_mining.multimodal_hard_negative "${args[@]}"

echo "Student-aware hard-negative mining output: $OUTPUT_DIR"
