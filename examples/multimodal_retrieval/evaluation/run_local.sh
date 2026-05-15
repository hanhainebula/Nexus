#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${REPO_ROOT}"

"${PYTHON_BIN}" -m Nexus.evaluation.multimodal_retrieval \
  --eval_config "${REPO_ROOT}/examples/multimodal_retrieval/evaluation/eval_config.json" \
  --model_config "${REPO_ROOT}/examples/multimodal_retrieval/evaluation/model_config.json"
