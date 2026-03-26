#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${REPO_ROOT}/outputs/mm_embedder_qwen25vl_lora}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/examples/multimodal_retrieval/data/eval}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${REPO_ROOT}/outputs/mm_eval}"

cd "${REPO_ROOT}"

"${PYTHON_BIN}" -m Nexus.evaluation.multimodal_retrieval \
  --eval_name "${EVAL_NAME:-mmeb_local}" \
  --dataset_dir "${DATASET_DIR}" \
  --eval_output_dir "${EVAL_OUTPUT_DIR}" \
  --eval_output_path "${EVAL_OUTPUT_DIR}/summary.md" \
  --embedder_name_or_path "${MODEL_NAME_OR_PATH}" \
  --processor_name_or_path "${MODEL_NAME_OR_PATH}" \
  --embedder_batch_size "${EMBED_BATCH_SIZE:-8}" \
  --embedder_query_max_length "${QUERY_MAX_LEN:-512}" \
  --embedder_passage_max_length "${PASSAGE_MAX_LEN:-1024}" \
  --pooling_method "${POOLING_METHOD:-last_token}" \
  --normalize_embeddings True \
  --search_top_k "${SEARCH_TOP_K:-1000}"
