#!/usr/bin/env bash

set -euo pipefail

TARGET_ROOT="${1:-data/vlm2vec_eval/MMEB-V2}"
LOG_DIR="${2:-$TARGET_ROOT/logs}"
DOWNLOAD_LOG="$LOG_DIR/download.log"

mkdir -p "$TARGET_ROOT" "$LOG_DIR"

if command -v aria2c >/dev/null 2>&1; then
  DOWNLOADER="aria2c"
elif command -v curl >/dev/null 2>&1; then
  DOWNLOADER="curl"
else
  echo "Neither aria2c nor curl is available." >&2
  exit 1
fi

FILES=(
  "image-tasks/mmeb_v1.tar.gz|https://huggingface.co/datasets/TIGER-Lab/MMEB-V2/resolve/main/image-tasks/mmeb_v1.tar.gz"
  "video-tasks/frames/video_cls.tar.gz|https://huggingface.co/datasets/TIGER-Lab/MMEB-V2/resolve/main/video-tasks/frames/video_cls.tar.gz"
  "video-tasks/frames/video_ret.tar.gz|https://huggingface.co/datasets/TIGER-Lab/MMEB-V2/resolve/main/video-tasks/frames/video_ret.tar.gz"
  "video-tasks/frames/video_mret.tar.gz|https://huggingface.co/datasets/TIGER-Lab/MMEB-V2/resolve/main/video-tasks/frames/video_mret.tar.gz"
  "video-tasks/frames/video_qa.tar.gz-00|https://huggingface.co/datasets/TIGER-Lab/MMEB-V2/resolve/main/video-tasks/frames/video_qa.tar.gz-00"
  "video-tasks/frames/video_qa.tar.gz-01|https://huggingface.co/datasets/TIGER-Lab/MMEB-V2/resolve/main/video-tasks/frames/video_qa.tar.gz-01"
  "video-tasks/frames/video_qa.tar.gz-02|https://huggingface.co/datasets/TIGER-Lab/MMEB-V2/resolve/main/video-tasks/frames/video_qa.tar.gz-02"
  "video-tasks/frames/video_qa.tar.gz-03|https://huggingface.co/datasets/TIGER-Lab/MMEB-V2/resolve/main/video-tasks/frames/video_qa.tar.gz-03"
  "video-tasks/frames/video_qa.tar.gz-04|https://huggingface.co/datasets/TIGER-Lab/MMEB-V2/resolve/main/video-tasks/frames/video_qa.tar.gz-04"
  "visdoc-tasks/visdoc-tasks.data.tar.gz|https://huggingface.co/datasets/TIGER-Lab/MMEB-V2/resolve/main/visdoc-tasks/visdoc-tasks.data.tar.gz"
  "visdoc-tasks/visdoc-tasks.images.tar.gz|https://huggingface.co/datasets/TIGER-Lab/MMEB-V2/resolve/main/visdoc-tasks/visdoc-tasks.images.tar.gz"
)

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$DOWNLOAD_LOG"
}

download_with_aria2c() {
  local url="$1"
  local out_dir="$2"
  local out_name="$3"
  aria2c \
    --continue=true \
    --max-tries=5 \
    --retry-wait=10 \
    --file-allocation=none \
    --summary-interval=30 \
    --max-connection-per-server=8 \
    --split=8 \
    --dir="$out_dir" \
    --out="$out_name" \
    "$url"
}

download_with_curl() {
  local url="$1"
  local out_dir="$2"
  local out_name="$3"
  curl \
    -L \
    --fail \
    --retry 5 \
    -C - \
    -o "$out_dir/$out_name" \
    "$url"
}

log "Starting MMEB v2 eval media download into $TARGET_ROOT"
log "Downloader: $DOWNLOADER"

for item in "${FILES[@]}"; do
  rel_path="${item%%|*}"
  url="${item#*|}"
  out_dir="$TARGET_ROOT/$(dirname "$rel_path")"
  out_name="$(basename "$rel_path")"
  mkdir -p "$out_dir"
  log "Downloading $rel_path"
  if [[ "$DOWNLOADER" == "aria2c" ]]; then
    download_with_aria2c "$url" "$out_dir" "$out_name" 2>&1 | tee -a "$DOWNLOAD_LOG"
  else
    download_with_curl "$url" "$out_dir" "$out_name" 2>&1 | tee -a "$DOWNLOAD_LOG"
  fi
  log "Finished $rel_path"
done

log "All MMEB v2 eval media downloads completed."
