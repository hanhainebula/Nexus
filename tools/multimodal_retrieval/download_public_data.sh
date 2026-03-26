#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
RAW_ROOT="${RAW_ROOT:-$DATA_ROOT/raw}"
NEXUS_ROOT="${NEXUS_ROOT:-$DATA_ROOT/nexus}"
MANIFEST="${MANIFEST:-$REPO_ROOT/docs/multimodal_retrieval/MMEB_v2_manifest.json}"
TRAIN_MODALITY="${TRAIN_MODALITY:-image}"
SOURCE_NAMES="${SOURCE_NAMES:-}"
SKIP_MEDIA="${SKIP_MEDIA:-0}"
DOWNLOAD="${DOWNLOAD:-0}"
CONVERT_TRAIN="${CONVERT_TRAIN:-0}"
EXTRACT_ARCHIVES="${EXTRACT_ARCHIVES:-0}"
DRY_RUN="${DRY_RUN:-1}"
MAX_ROWS="${MAX_ROWS:-}"

ARGS=(
  "--manifest" "$MANIFEST"
  "--raw-root" "$RAW_ROOT"
  "--nexus-root" "$NEXUS_ROOT"
  "--python-bin" "$PYTHON_BIN"
)

IFS=',' read -r -a MODALITIES <<< "$TRAIN_MODALITY"
for modality in "${MODALITIES[@]}"; do
  [[ -n "$modality" ]] || continue
  ARGS+=("--train-modality" "$modality")
done

IFS=',' read -r -a SOURCES <<< "$SOURCE_NAMES"
for source_name in "${SOURCES[@]}"; do
  [[ -n "$source_name" ]] || continue
  ARGS+=("--source-name" "$source_name")
done

if [[ "$SKIP_MEDIA" == "1" ]]; then
  ARGS+=("--skip-media")
fi
if [[ "$DOWNLOAD" == "1" ]]; then
  ARGS+=("--download")
fi
if [[ "$CONVERT_TRAIN" == "1" ]]; then
  ARGS+=("--convert-train")
fi
if [[ "$EXTRACT_ARCHIVES" == "1" ]]; then
  ARGS+=("--extract-archives")
fi
if [[ "$DRY_RUN" == "1" ]]; then
  ARGS+=("--dry-run")
fi
if [[ -n "$MAX_ROWS" ]]; then
  ARGS+=("--max-rows" "$MAX_ROWS")
fi

echo "PYTHON_BIN=$PYTHON_BIN"
echo "RAW_ROOT=$RAW_ROOT"
echo "NEXUS_ROOT=$NEXUS_ROOT"
echo "TRAIN_MODALITY=$TRAIN_MODALITY"
echo "SOURCE_NAMES=${SOURCE_NAMES:-<all>}"
echo "DOWNLOAD=$DOWNLOAD CONVERT_TRAIN=$CONVERT_TRAIN SKIP_MEDIA=$SKIP_MEDIA EXTRACT_ARCHIVES=$EXTRACT_ARCHIVES DRY_RUN=$DRY_RUN"

"$PYTHON_BIN" "$REPO_ROOT/tools/multimodal_retrieval/prepare_public_data.py" "${ARGS[@]}"
