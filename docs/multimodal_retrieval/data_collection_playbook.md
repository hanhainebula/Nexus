# Public Data Collection Playbook

Last updated: 2026-03-25

## Scope

This document turns the current MMEB v2 public-data understanding into an executable collection plan for Nexus.

It does not assume that all datasets must be downloaded immediately. The goal is to make storage layout, ownership, and post-processing consistent before large downloads start.

## Principles

- Use an isolated environment instead of the local `base` environment.
- Check available disk space before downloading large archives.
- Keep raw downloads and Nexus-converted outputs separate.
- Keep media roots configurable instead of rewriting everything to absolute paths.
- Do not start GPU jobs until data conversion smoke tests pass.

## Recommended Directory Layout

```text
<DATA_ROOT>/
  raw/
    vlm2vec_train/
      MMEB-train/
      ShareGPTVideo/
      colpali_train_set/
      VisRAG/
    vlm2vec_eval/
      MMEB-V2/
  nexus/
    train/
    eval/
  logs/
```

## Public Training Sources

### Image

Source:

- `TIGER-Lab/MMEB-train`

Recommended subsets for the first unified recipe:

- `ImageNet_1K`
- `N24News`
- `HatefulMemes`
- `VOC2007`
- `SUN397`
- `OK-VQA`
- `A-OKVQA`
- `DocVQA`
- `InfographicsVQA`
- `ChartQA`
- `Visual7W`
- `VisDial`
- `CIRR`
- `VisualNews_t2i`
- `VisualNews_i2t`
- `MSCOCO_t2i`
- `MSCOCO_i2t`
- `NIGHTS`
- `WebQA`
- `MSCOCO`

### Video

Source:

- `ShareGPTVideo/train_video_and_instruction`

Recommended entries:

- `video_caption_300k`
- `video_caption_300k-video`
- `video_qa_240k`

Post-processing:

- unpack frame archives
- generate `video_qa_240k.jsonl` if needed
- decide whether final Nexus training should use `video_path` or `video_frames`

### Visdoc

Sources:

- `vidore/colpali_train_set`
- `openbmb/VisRAG-Ret-Train-In-domain-data`

## Public Evaluation Source

Source:

- `TIGER-Lab/MMEB-V2`

Expected contents:

- image tasks
- video tasks
- visual document tasks

## Collection Workflow

### Step 1: Prepare environment

- create or activate an isolated environment
- install `git-lfs`
- verify that `datasets`, `PyYAML`, and `transformers` are available if conversion will follow immediately

### Step 2: Download raw sources

Use:

- `tools/multimodal_retrieval/download_public_data.sh`
- `tools/multimodal_retrieval/prepare_public_data.py`

Recommended first pass:

- image train
- visdoc train
- one small slice of video train
- full eval metadata only if storage is constrained

### Step 3: Convert to Nexus format

Use:

- `tools/multimodal_retrieval/convert_vlm2vec_train_to_nexus.py`
- `tools/multimodal_retrieval/convert_vlm2vec_eval_to_nexus.py`
- `tools/multimodal_retrieval/prepare_mmeb_v2_train_data.py`

Recommended outputs:

- `nexus/train/*.jsonl`
- `nexus/eval/<dataset_name>/corpus.jsonl`
- `nexus/eval/<dataset_name>/test_queries.jsonl`
- `nexus/eval/<dataset_name>/test_qrels.jsonl`

### Step 4: Smoke validation

Use:

- `tools/multimodal_retrieval/validate_stack.sh`

Then run one tiny local evaluation and one tiny training dry run.

## Ownership Suggestion

If data collection is shared across teammates:

- one person owns raw data download and checksums
- one person owns Nexus conversion outputs
- one person owns training mixtures and sampling configs

## Risks

- video archives are the most storage-heavy part
- Hugging Face and git-lfs download methods may vary by environment
- some sources may change file layout over time
- full benchmark download and preprocessing can take hours

## Current Status

Prepared in this repository:

- human-readable MMEB v2 inventory
- machine-readable MMEB v2 manifest
- Nexus conversion scripts
- HF HTTP planning and selective download tooling
- stage-oriented train-data preparation tooling
- validation script
- GPU-safety helper

Still not executed here:

- full raw data download
- full-scale conversion of all public sources
- full training run
- full MMEB v2 benchmark run
