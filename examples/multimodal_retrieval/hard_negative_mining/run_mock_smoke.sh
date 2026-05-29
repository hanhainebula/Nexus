#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEXUS_ROOT="${NEXUS_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
INPUT_TRAIN_JSONL="${INPUT_TRAIN_JSONL:-$SCRIPT_DIR/data/train_tiny.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$NEXUS_ROOT/outputs/hard_negative_mining/mock_smoke}"
CANDIDATE_SOURCE="${CANDIDATE_SOURCE:-all_pos}"
TOP_K="${TOP_K:-5}"
NUM_HARD_NEGATIVES="${NUM_HARD_NEGATIVES:-2}"

mkdir -p "$OUTPUT_DIR"
cd "$NEXUS_ROOT"
export PYTHONPATH="$NEXUS_ROOT:${PYTHONPATH:-}"
PYTHON="${PYTHON:-python3}"

"$PYTHON" -m Nexus.data_mining.multimodal_hard_negative \
  --mode mock \
  --input_train_jsonl "$INPUT_TRAIN_JSONL" \
  --output_train_jsonl "$OUTPUT_DIR/train.mock_hardneg.jsonl" \
  --max_examples 0 \
  --candidate_source "$CANDIDATE_SOURCE" \
  --top_k "$TOP_K" \
  --num_hard_negatives "$NUM_HARD_NEGATIVES" \
  --report_path "$OUTPUT_DIR/report.json" \
  --manifest_path "$OUTPUT_DIR/manifest.json" \
  --audit_sample_path "$OUTPUT_DIR/audit.jsonl" \
  --audit_sample_size 10 \
  --write_scores \
  --no-require_existing_media

echo "Mock hard-negative mining output: $OUTPUT_DIR"
