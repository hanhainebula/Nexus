#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="${ENV_NAME:-nexus-mmeb}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
RUN_VALIDATION="${RUN_VALIDATION:-1}"

echo "Creating isolated conda environment: ${ENV_NAME}"
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install -e "${REPO_ROOT}[eval,multimodal]"

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  echo "Installing flash-attn inside ${ENV_NAME}"
  conda run -n "${ENV_NAME}" python -m pip install flash-attn --no-build-isolation
fi

echo "Collecting key package versions inside ${ENV_NAME}"
conda run -n "${ENV_NAME}" python -c "import accelerate, transformers; print('transformers', transformers.__version__); print('accelerate', accelerate.__version__)"

if [[ "${RUN_VALIDATION}" == "1" ]]; then
  echo "Running validate_stack.sh inside ${ENV_NAME}"
  conda run -n "${ENV_NAME}" bash "${REPO_ROOT}/tools/multimodal_retrieval/validate_stack.sh"
fi

echo "Environment ready: ${ENV_NAME}"
echo "Do not install multimodal dependencies into the local base environment."
