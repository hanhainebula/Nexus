# Multimodal Retrieval Tools

This directory contains helper scripts for preparing MMEB/VLM2Vec-style data for Nexus.

## Scripts

### `convert_vlm2vec_train_to_nexus.py`

Convert training data into Nexus train JSONL.

The input may be:

- a single `json` / `jsonl` / `parquet` file
- a local directory containing nested `json` / `jsonl` / `parquet` shards
- a Hugging Face dataset name

Example:

```bash
python tools/multimodal_retrieval/convert_vlm2vec_train_to_nexus.py \
  --input /path/to/train.jsonl \
  --source-format llavahound_caption \
  --llavahound-mode caption_retrieval \
  --video-root /path/to/raw/videos \
  --output converted/train.jsonl
```

### `convert_vlm2vec_eval_to_nexus.py`

Convert VLM2Vec pair-style evaluation data into the Nexus local evaluation layout:

- `corpus.jsonl`
- `<split>_queries.jsonl`
- `<split>_qrels.jsonl`

Example:

```bash
python tools/multimodal_retrieval/convert_vlm2vec_eval_to_nexus.py \
  --input /path/to/vlm2vec_eval_export \
  --split test \
  --sequence-mode video \
  --output-dir converted_eval/MSR-VTT
```

The converter now also understands legacy pair-style rows such as HatefulMemes-style records with fields like:

- `qry`
- `qry_image_path`
- `pos_text`
- `neg_text`

This makes it possible to turn binary classification-style multimodal metadata into Nexus local-eval triples without writing a one-off conversion script.

### `export_mmeb_v2_inventory.py`

Generate a markdown inventory from the local `VLM2Vec` references.

Example:

```bash
python tools/multimodal_retrieval/export_mmeb_v2_inventory.py \
  --output docs/multimodal_retrieval/MMEB_v2_inventory_generated.md
```

The script auto-discovers `VLM2Vec` from:

- `VLM2VEC_ROOT`
- a sibling repo such as `../VLM2Vec`

You can still pass `--vlm2vec-root` explicitly when needed.

### `export_mmeb_v2_manifest.py`

Generate a machine-readable JSON manifest for the MMEB v2 eval sets and public train sources:

```bash
python tools/multimodal_retrieval/export_mmeb_v2_manifest.py \
  --output docs/multimodal_retrieval/MMEB_v2_manifest.json
```

The exported manifest now separates:

- metadata HF sources
- media HF sources
- parser family
- download patterns for train sources

This matters because several MMEB eval tasks use metadata from one repo and images or videos from another.

### `hf_dataset_manager.py`

Plan or download public dataset files through the Hugging Face HTTP API without relying on `git clone`:

```bash
python tools/multimodal_retrieval/hf_dataset_manager.py \
  --repo TIGER-Lab/MMEB-train \
  --output-root /path/to/raw/vlm2vec_train/MMEB-train \
  --include 'VOC2007/original-*' \
  --include 'images_zip/VOC2007.zip'
```

### `prepare_public_data.py`

Manifest-driven planning, selective download, and train conversion:

```bash
python tools/multimodal_retrieval/prepare_public_data.py \
  --manifest docs/multimodal_retrieval/MMEB_v2_manifest.json \
  --raw-root /path/to/data/raw \
  --nexus-root /path/to/data/nexus \
  --train-modality image \
  --source-name VOC2007 \
  --download \
  --extract-archives \
  --convert-train
```

### `prepare_mmeb_v2_train_data.py`

Convert already-downloaded public train sources into Nexus JSONL and emit stage-specific `data_config.json` files:

```bash
python tools/multimodal_retrieval/prepare_mmeb_v2_train_data.py \
  --manifest docs/multimodal_retrieval/MMEB_v2_manifest.json \
  --raw-root /path/to/data/raw \
  --output-root /path/to/data/nexus/train_ready \
  --stage stage_a \
  --write-stage-configs-dir /path/to/data/nexus/stage_configs
```

### `prepare_mmeb_v2_train_data.py`

Convert a modality/stage subset and emit stage-level `data_config.json` files for Nexus training:

```bash
python tools/multimodal_retrieval/prepare_mmeb_v2_train_data.py \
  --raw-root /path/to/data/raw \
  --output-root /path/to/data/nexus/train \
  --stage stage_a \
  --local-only \
  --allow-missing \
  --write-stage-configs-dir /path/to/data/nexus/configs
```

### `prepare_exploration_train_data.py`

Build an exploration-oriented train JSONL at an exact target size before a smoke finetune:

```bash
python tools/multimodal_retrieval/prepare_exploration_train_data.py \
  --input /path/to/data/nexus/train_ready \
  --output /path/to/data/nexus/exploration/train_1000.jsonl \
  --target-size 1000 \
  --fill-mode repeat_to_target
```

This keeps the "expand to 1000 rows" step in data preparation instead of burying oversampling logic inside the training loop.

### `repair_mmeb_train_media_paths.py`

Rewrite converted Nexus train JSONL rows so their media paths match a real local image layout:

```bash
python tools/multimodal_retrieval/repair_mmeb_train_media_paths.py \
  --input /path/to/data/nexus/train_ready/HatefulMemes.jsonl \
  --output /path/to/data/nexus/train_ready_fixed/HatefulMemes.jsonl \
  --image-root /path/to/local/MMEB/HatefulMemes \
  --fail-on-unresolved
```

This is useful when converted metadata uses repo-style paths such as `images/HatefulMemes/Train/HatefulMemes_image_542.jpg`, but the locally extracted media lives under an archive layout such as `image_542.jpg`.

### `prepare_mmeb_v2_eval_data.py`

Batch-convert MMEB v2 eval subsets from a local raw-data mirror into the Nexus local-eval layout and emit per-dataset `eval_config.json` files:

```bash
python tools/multimodal_retrieval/prepare_mmeb_v2_eval_data.py \
  --manifest docs/multimodal_retrieval/MMEB_v2_manifest.json \
  --raw-root /path/to/data/raw \
  --output-root /path/to/data/nexus/eval_ready \
  --datasets HatefulMemes MSVD \
  --write-eval-configs-dir /path/to/data/nexus/eval_configs \
  --allow-missing \
  --local-only
```

This script understands both:

- VLM2Vec-style local mirrors such as `image-tasks/`, `video-tasks/`, and `visdoc-tasks/`
- repo-cache layouts under `raw-root/vlm2vec_eval/`

For video datasets it prefers `frame_root` as the effective `image_root`, which matches how VLM2Vec eval loaders consume frame sequences during retrieval evaluation.

New robustness notes:

- Generated `eval_config.json` files now place `cache_path` inside the chosen `output-root`, instead of defaulting to the repo root.
- `--write-configs-only` lets you refresh `eval_config.json` files without re-running dataset conversion or touching remote metadata.
- When `--overwrite` is used, conversion now writes to a staging directory first and only replaces the final dataset directory after the new conversion succeeds.

### `check_idle_gpus.py`

Inspect shared GPUs before starting training:

```bash
python tools/multimodal_retrieval/check_idle_gpus.py
```

If direct GPU probing fails inside an isolated environment, dump `nvidia-smi` output first and pass it back in:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits > /tmp/gpus.csv
python tools/multimodal_retrieval/check_idle_gpus.py --input /tmp/gpus.csv
```

### `create_conda_env.sh` and `environment.yml`

Use these to create an isolated runtime instead of installing into the local `base` environment.

Recommended command:

```bash
ENV_NAME=nexus-mmeb-stage1 \
bash tools/multimodal_retrieval/create_conda_env.sh
```

By default the script will:

- create a fresh isolated conda environment
- install `.[eval,multimodal]`
- print key package versions
- run `validate_stack.sh` inside that environment

If you only want the installation step and prefer to validate later, disable the final check:

```bash
ENV_NAME=nexus-mmeb-stage1 \
RUN_VALIDATION=0 \
bash tools/multimodal_retrieval/create_conda_env.sh
```

### `download_public_data.sh`

Wrapper around `prepare_public_data.py` for common shell-driven workflows.

By default it runs in dry-run mode and only plans the work:

```bash
DATA_ROOT=/path/to/storage \
DRY_RUN=1 \
bash tools/multimodal_retrieval/download_public_data.sh
```

### `validate_stack.sh`

Run syntax checks, multimodal tests, inventory export, and conversion smoke tests:

```bash
PYTHON_BIN=/path/to/python \
bash tools/multimodal_retrieval/validate_stack.sh
```

If `VLM2Vec` is not available locally, the script skips the inventory-export step by default and continues with the remaining checks. To require that step, set:

```bash
VALIDATE_REQUIRE_VLM2VEC=1 \
PYTHON_BIN=/path/to/python \
bash tools/multimodal_retrieval/validate_stack.sh
```

### `validate_backbone_matrix.py`

Validate whether the current environment can really instantiate and reload the backbone families that Nexus claims to support:

```bash
PYTHONPATH=/path/to/Nexus \
python tools/multimodal_retrieval/validate_backbone_matrix.py \
  --output-dir /tmp/backbone_matrix/current_env \
  --label current_env
```

This script:

- checks whether each family-specific `transformers` class exists in the current environment
- builds a tiny local checkpoint for each available family
- runs the real `save_pretrained -> load_multimodal_backbone -> from_pretrained` path
- writes both `report.json` and `summary.md`

The current matrix includes:

- `Qwen2-VL`
- `Qwen2.5-VL`
- `Qwen3-VL`
- `Qwen3.5`
- `Llava-Next`

Related processor note:

- `Qwen2-VL`, `Qwen2.5-VL`, `Qwen3-VL`, and `Qwen3.5` expose Qwen-style `min_pixels` / `max_pixels` overrides through `AutoProcessor`.
- `Llava-Next` uses `size` / `crop_size` instead. Nexus now translates Qwen-style pixel budgets into `size` / `crop_size` when `model_type=llava_next`, while still respecting explicit `size` / `crop_size` overrides when they are provided.

If you need to require every requested family to be present and load successfully, add:

```bash
PYTHONPATH=/path/to/Nexus \
python tools/multimodal_retrieval/validate_backbone_matrix.py \
  --output-dir /tmp/backbone_matrix \
  --fail-on-unavailable \
  --fail-on-failure
```

## Environment

Run these tools inside an isolated environment. Do not install dependencies into the local `base` environment.
For the final Stage 2 competitive path, prefer a fresh environment that satisfies the multimodal extra requirements, especially if the chosen backbone moves from `Qwen2-VL` smoke validation to the `Qwen3-VL` or `Qwen3.5` families.
