# MMEB v2 Evaluation Examples

This directory documents how to run MMEB v2 benchmark evaluation from inside Nexus.
It is different from `examples/multimodal_retrieval/`:

| Directory | Purpose |
| --- | --- |
| `examples/multimodal_retrieval/` | Nexus-native training, inference, and local retrieval evaluation with `corpus.jsonl`, `queries.jsonl`, and `qrels.jsonl`. |
| `examples/multimodal_retrieval/evaluation/mmeb_v2/` | MMEB v2 benchmark evaluation with official-style task YAML files and parsers. |

Use this directory when you want to evaluate a base model or a Nexus-finetuned checkpoint on MMEB v2 tasks.

## Entrypoint

The embedded MMEB v2 evaluator lives in:

```text
Nexus/evaluation/mmeb_v2/
```

Run it with:

```bash
python -m Nexus.evaluation.mmeb_v2.eval_embedding
```

For multi-GPU evaluation, launch the same module through `torchrun`:

```bash
torchrun --nproc_per_node=8 -m Nexus.evaluation.mmeb_v2.eval_embedding ...
```

## Environment

Install Nexus and the multimodal dependencies first. A typical editable setup is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r examples/multimodal_retrieval/requirements.txt
pip install -e . --no-deps
```

If your cluster already provides a prepared environment, activate that environment instead.

Before running the examples, set at least these variables:

```bash
export MODEL=/path/to/model-or-checkpoint
export DATA_BASEDIR=/path/to/MMEB/vlm2vec_eval
```

Optional variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `NEXUS_ROOT` | Repository root inferred from the script location. | Nexus checkout to evaluate from. |
| `RUN_ROOT` | `outputs/mmeb_v2/<task>_<timestamp>` under the repository. | Output directory for embeddings, scores, logs, and generated media. |
| `NPROC` | `8` | Number of distributed GPU processes. |
| `BATCH_SIZE` | Task-specific default. | Per-device evaluation batch size. |
| `MASTER_PORT` | `29741` | Torch distributed master port. |
| `TORCHRUN` | `torchrun` | Torch distributed launcher executable. |
| `HF_HOME` | Not set by the script. | Optional Hugging Face cache root. |
| `HF_ENDPOINT` | Not set by the script. | Optional Hugging Face mirror endpoint. |

## Quick Smoke Commands

The scripts below are full-task runs for one representative task in each major modality path. They are not random few-sample tests.

### Image: ImageNet-1K

```bash
MODEL=/path/to/model-or-checkpoint DATA_BASEDIR=/path/to/MMEB/vlm2vec_eval bash examples/multimodal_retrieval/evaluation/mmeb_v2/run_imagenet1k.sh
```

Default config:

```text
examples/multimodal_retrieval/evaluation/mmeb_v2/configs/imagenet1k_only.yaml
```

A previously verified Qwen3-VL-Embedding-2B run produced:

```text
hit@1=0.782
num_pred=1000
num_data=1000
```

### VisDoc: VisRAG_ChartQA

```bash
MODEL=/path/to/model-or-checkpoint DATA_BASEDIR=/path/to/MMEB/vlm2vec_eval bash examples/multimodal_retrieval/evaluation/mmeb_v2/run_visrag_chartqa.sh
```

Default config:

```text
examples/multimodal_retrieval/evaluation/mmeb_v2/configs/visrag_chartqa_only.yaml
```

A previously verified Qwen3-VL-Embedding-2B run produced:

```text
ndcg_linear@5=0.8584972274
num_pred=63
num_data=63
```

VisDoc and VisRAG tasks may materialize page images. The scripts always pass `--generated_media_basedir` and write generated files under `RUN_ROOT/generated_media_cache`.

### Video: HMDB51 Full Task

```bash
MODEL=/path/to/model-or-checkpoint DATA_BASEDIR=/path/to/MMEB/vlm2vec_eval bash examples/multimodal_retrieval/evaluation/mmeb_v2/run_hmdb51_full.sh
```

Default config:

```text
examples/multimodal_retrieval/evaluation/mmeb_v2/configs/hmdb51_full.yaml
```

Default video settings:

```text
VIDEO_FPS=1
MAX_VIDEO_FRAMES=64
BATCH_SIZE=8
```

A previously verified Qwen3-VL-Embedding-2B run produced:

```text
hit@1=0.828
num_pred=1000
num_data=1000
```

For code-only smoke tests, you may reduce `MAX_VIDEO_FRAMES`. For score comparison with official-style results, do not report low-frame debug runs as final benchmark numbers.

## Generic Command Template

```bash
export MODEL=/path/to/model-or-checkpoint
export DATA_BASEDIR=/path/to/MMEB/vlm2vec_eval
export NEXUS_ROOT=/path/to/Nexus
export RUN_ROOT=$NEXUS_ROOT/outputs/mmeb_v2/manual_eval
export PYTHONPATH=$NEXUS_ROOT:${PYTHONPATH:-}
cd $NEXUS_ROOT

mkdir -p $RUN_ROOT/generated_media_cache

torchrun \
  --nproc_per_node=8 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=localhost \
  --master_port=29741 \
  --max_restarts=0 \
  -m Nexus.evaluation.mmeb_v2.eval_embedding \
  --normalize true \
  --per_device_eval_batch_size 32 \
  --model_name_or_path $MODEL \
  --dataset_config examples/multimodal_retrieval/evaluation/mmeb_v2/configs/imagenet1k_only.yaml \
  --encode_output_path $RUN_ROOT/outputs/imagenet1k \
  --data_basedir $DATA_BASEDIR \
  --generated_media_basedir $RUN_ROOT/generated_media_cache \
  --force_recompute true
```

## Evaluating a Finetuned Nexus Checkpoint

After finetuning, point `MODEL` to the checkpoint directory:

```bash
MODEL=/path/to/nexus-finetuned-checkpoint DATA_BASEDIR=/path/to/MMEB/vlm2vec_eval bash examples/multimodal_retrieval/evaluation/mmeb_v2/run_imagenet1k.sh
```

No other change is required if the checkpoint can be loaded by the Nexus multimodal embedder wrapper.

## Config Files

Common configs:

| Config | Purpose |
| --- | --- |
| `configs/imagenet1k_only.yaml` | ImageNet-1K single-task run. |
| `configs/visrag_chartqa_only.yaml` | VisRAG_ChartQA single-task run. |
| `configs/hmdb51_full.yaml` | HMDB51 full-task run. |
| `configs/image.yaml` | Image-task group. |
| `configs/video.yaml` | Video-task group. |
| `configs/visdoc.yaml` | VisDoc/VisRAG task group. |
| `configs/image_retrieval.yaml` | Image retrieval task group. |
| `configs/video_retrieval.yaml` | Video retrieval task group. |
| `configs/visdoc_retrieval.yaml` | VisDoc retrieval task group. |

Prefer the provided config files over hand-written paths. A wrong `image_root`, `video_root`, or `frame_root` can produce incomplete predictions and misleadingly low scores.

## Required Checks After Each Run

Do not report only the main metric. Always check:

| Check | Why it matters |
| --- | --- |
| `num_pred` | Number of evaluated queries. |
| `num_data` | Expected number of queries. |
| `num_pred == num_data` | If false, debug dataset config, media paths, and caches before interpreting the score. |
| Log module path | Logs should contain `Nexus.evaluation.mmeb_v2`, confirming that the embedded Nexus evaluator was used. |
| Output directory | Use a fresh `RUN_ROOT` when comparing runs to avoid stale cached embeddings. |

## Current Validation Status

The embedded evaluator has been checked on one representative full task for each major path:

| Path | Task | Previously verified result | Completeness |
| --- | --- | --- | --- |
| Image | `ImageNet-1K` | `hit@1=0.782` | `num_pred=1000 / num_data=1000` |
| VisDoc | `VisRAG_ChartQA` | `ndcg_linear@5=0.8584972274` | `num_pred=63 / num_data=63` |
| Video | `HMDB51` | `hit@1=0.828` | `num_pred=1000 / num_data=1000` |

These checks validate that the Nexus-integrated evaluator can run image, document, and video MMEB v2 paths with official-style scoring.
