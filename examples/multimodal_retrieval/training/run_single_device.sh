#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${REPO_ROOT}"

if [[ "${REQUIRE_EXPLICIT_GPUS:-1}" == "1" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Set CUDA_VISIBLE_DEVICES to an idle GPU before training."
  echo "Use tools/multimodal_retrieval/check_idle_gpus.py first on shared machines."
  exit 1
fi

"${PYTHON_BIN}" -m Nexus.training.embedder.multimodal_retrieval \
  --model_config "${REPO_ROOT}/examples/multimodal_retrieval/training/model_config.json" \
  --data_config "${REPO_ROOT}/examples/multimodal_retrieval/training/data_config.json" \
  --training_config "${REPO_ROOT}/examples/multimodal_retrieval/training/training_config.json"
