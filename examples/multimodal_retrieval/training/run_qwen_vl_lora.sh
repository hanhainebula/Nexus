#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
TRAIN_DATA="${TRAIN_DATA:-${REPO_ROOT}/examples/multimodal_retrieval/data/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/mm_embedder_qwen25vl_lora}"
PRECISION_MODE="${PRECISION_MODE:-bf16}"

PRECISION_ARGS=()
case "${PRECISION_MODE}" in
  bf16)
    TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
    PRECISION_ARGS+=(--bf16)
    ;;
  fp16)
    TORCH_DTYPE="${TORCH_DTYPE:-float16}"
    PRECISION_ARGS+=(--fp16)
    ;;
  fp32)
    TORCH_DTYPE="${TORCH_DTYPE:-float32}"
    ;;
  *)
    echo "Unsupported PRECISION_MODE=${PRECISION_MODE}. Use one of: bf16, fp16, fp32."
    exit 1
    ;;
esac

cd "${REPO_ROOT}"

if [[ "${REQUIRE_EXPLICIT_GPUS:-1}" == "1" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Set CUDA_VISIBLE_DEVICES to idle GPUs only before training."
  echo "Use tools/multimodal_retrieval/check_idle_gpus.py to inspect shared GPUs first."
  exit 1
fi

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

"${TORCHRUN_BIN}" --nproc_per_node="${NPROC_PER_NODE:-1}" \
  -m Nexus.training.embedder.multimodal_retrieval \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --processor_name_or_path "${MODEL_NAME_OR_PATH}" \
  --train_data "${TRAIN_DATA}" \
  --output_dir "${OUTPUT_DIR}" \
  --overwrite_output_dir \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
  --learning_rate "${LEARNING_RATE:-2e-5}" \
  --train_group_size "${TRAIN_GROUP_SIZE:-8}" \
  --query_max_len "${QUERY_MAX_LEN:-512}" \
  --passage_max_len "${PASSAGE_MAX_LEN:-1024}" \
  --temperature "${TEMPERATURE:-0.02}" \
  --sentence_pooling_method "${POOLING_METHOD:-last_token}" \
  --normalize_embeddings True \
  --negatives_cross_device \
  --use_chat_template True \
  --use_lora True \
  --lora_r "${LORA_R:-64}" \
  --lora_alpha "${LORA_ALPHA:-128}" \
  --lora_dropout "${LORA_DROPOUT:-0.05}" \
  --torch_dtype "${TORCH_DTYPE}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}" \
  --logging_steps "${LOGGING_STEPS:-10}" \
  --save_steps "${SAVE_STEPS:-500}" \
  "${PRECISION_ARGS[@]}"
