#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NEXUS_ROOT=${NEXUS_ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}
MODEL=${MODEL:?Set MODEL=/path/to/model-or-checkpoint}
DATA_BASEDIR=${DATA_BASEDIR:?Set DATA_BASEDIR=/path/to/MMEB/vlm2vec_eval}
RUN_ROOT=${RUN_ROOT:-$NEXUS_ROOT/outputs/mmeb_v2/hmdb51_full_$(date +%Y%m%d_%H%M%S)}
NPROC=${NPROC:-8}
BATCH_SIZE=${BATCH_SIZE:-8}
MASTER_PORT=${MASTER_PORT:-29741}
TORCHRUN=${TORCHRUN:-torchrun}
CONFIG=${CONFIG:-$SCRIPT_DIR/configs/hmdb51_full.yaml}
OUT_SUBDIR=${OUT_SUBDIR:-hmdb51_full}

export PYTHONPATH="$NEXUS_ROOT:${PYTHONPATH:-}"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/generated_media_cache" "$RUN_ROOT/outputs"
cd "$NEXUS_ROOT"

echo "[MMEB v2] task=hmdb51_full"
echo "[MMEB v2] nexus=$NEXUS_ROOT"
echo "[MMEB v2] model=$MODEL"
echo "[MMEB v2] data=$DATA_BASEDIR"
echo "[MMEB v2] config=$CONFIG"
echo "[MMEB v2] run_root=$RUN_ROOT"

"$TORCHRUN" \
  --nproc_per_node="$NPROC" \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=localhost \
  --master_port="$MASTER_PORT" \
  --max_restarts=0 \
  -m Nexus.evaluation.mmeb_v2.eval_embedding \
  --normalize true \
  --per_device_eval_batch_size "$BATCH_SIZE" \
  --model_name_or_path "$MODEL" \
  --dataset_config "$CONFIG" \
  --encode_output_path "$RUN_ROOT/outputs/$(basename "$MODEL")/$OUT_SUBDIR" \
  --data_basedir "$DATA_BASEDIR" \
  --generated_media_basedir "$RUN_ROOT/generated_media_cache" \
  --force_recompute true \
  --video_fps ${VIDEO_FPS:-1} \
  --max_video_frames ${MAX_VIDEO_FRAMES:-64} \
  2>&1 | tee "$RUN_ROOT/logs/hmdb51_full.log"

find "$RUN_ROOT/outputs/$(basename "$MODEL")/$OUT_SUBDIR" -maxdepth 1 -name "*score.json" -print -exec cat {} \;
