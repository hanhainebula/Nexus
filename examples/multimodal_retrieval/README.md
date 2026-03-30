# Examples for `multimodal_retrieval`

This directory mirrors the `FlagEmbedding` style for a multimodal embedder workflow in Nexus:

1. finetune a dual-encoder multimodal retriever
2. run embedding inference on mixed text/image inputs
3. evaluate on a local retrieval dataset with `corpus.jsonl`, `<split>_queries.jsonl`, and `<split>_qrels.jsonl`

## Training data format

Each line in a training json/jsonl file should provide one query, one positive group, and an optional negative group.

```json
{
  "query": {
    "text": "Find the slide that explains rotary position embeddings.",
    "images": ["./assets/query_page.png"]
  },
  "pos": [
    {
      "text": "RoPE rotates hidden dimensions by position-dependent angles.",
      "image_path": "./assets/positive_page.png"
    }
  ],
  "neg": [
    {
      "text": "This page describes AdamW hyperparameters.",
      "image_path": "./assets/negative_page.png"
    }
  ],
  "pos_scores": [1.0],
  "neg_scores": [0.0]
}
```

Supported multimodal item fields:

- `text`
- `title`
- `image`
- `images`
- `image_path`
- `image_paths`
- `pages`
- `video`
- `video_path`
- `videos`
- `video_paths`
- `video_frames`

Relative media paths are resolved against the source json/jsonl file directory by default. You can also keep media files under separate roots by setting `media_root`, `image_root`, or `video_root` in the train/eval configs. When explicit negatives are missing, Nexus now falls back to sampling negatives from other records in the same dataset.

## Evaluation dataset format

Local evaluation expects:

- `corpus.jsonl`
- `<split>_queries.jsonl`
- `<split>_qrels.jsonl`

Each query/corpus line must contain `_id` plus the multimodal fields above. Each qrels line should contain:

```json
{"query-id": "q1", "corpus-id": "d42", "score": 1}
```

If your converted dataset stores JSONL files and media files under different directories, set the following fields in the evaluation config:

- `media_root`
- `image_root`
- `video_root`

## Quick start

- A tiny local smoke dataset is bundled under [`data/`](./data/) so the example configs have concrete paths to point at.
- The example JSON configs now resolve relative paths against the config file directory, so they can be launched from outside the repo root as well.
- Precision settings must match the GPU you actually use. `bf16` is suitable for datacenter GPUs such as A100/H100, while many consumer GPUs are safer with `PRECISION_MODE=fp16` or `PRECISION_MODE=fp32`.
- If you are setting up a fresh machine for multimodal experiments, start from [`requirements.txt`](./requirements.txt). This is the dependency set we validated against `Qwen2-VL`, `Qwen2.5-VL`, `Qwen3-VL`, `Qwen3.5`, and `Llava-Next`. A typical setup is:

```bash
python3.10 -m venv nexus-mm
source nexus-mm/bin/activate
pip install --upgrade pip
pip install -r examples/multimodal_retrieval/requirements.txt
pip install -e . --no-deps
```

- Config-file entrypoints: [`training/run_single_device.sh`](./training/run_single_device.sh) and [`evaluation/run_local.sh`](./evaluation/run_local.sh)
- CLI-style training: [`training/run_qwen_vl_lora.sh`](./training/run_qwen_vl_lora.sh)
- CLI-style evaluation: [`evaluation/run_local_eval.sh`](./evaluation/run_local_eval.sh)
- Inference: [`inference/encode_demo.py`](./inference/encode_demo.py)
- Data conversion tools: `tools/multimodal_retrieval/`

## Verified backbone families

The current multimodal retrieval stack has already been exercised on these backbone families:

- `Qwen2-VL`
- `Qwen2.5-VL`
- `Qwen3-VL`
- `Qwen3.5`

Notes:

- `Qwen3-VL` and `Qwen3.5` require a newer multimodal runtime than older local environments that only ship legacy `transformers` releases.
- [`requirements.txt`](./requirements.txt) pins the runtime that we actually used to load and exercise the currently supported multimodal backbones on this branch.
- The `backbone_load_strategy` config knob can now be used to compare the default loading path with `prefer_base_model`, which is helpful when validating direct `last_hidden_state` pooling on compatible VLM backbones.
- The training, inference, and evaluation entrypoints all accept `processor_kwargs` and `processor_call_kwargs`, so image-related overrides such as `size`, `min_pixels`, or `max_pixels` can be explored without patching the code.
- On `Qwen2-VL`, `Qwen2.5-VL`, `Qwen3-VL`, and `Qwen3.5`, `min_pixels` / `max_pixels` are forwarded to the underlying image processor directly. For example:

```json
{
  "processor_kwargs": {"max_pixels": 262144},
  "processor_call_kwargs": {"min_pixels": 50176, "max_pixels": 262144}
}
```

- `Llava-Next` uses `size` / `crop_size` instead of Qwen-style pixel-budget names. Nexus now translates `min_pixels` / `max_pixels` into an approximate square `size` / `crop_size` pair when `model_type=llava_next`, but explicit `size` / `crop_size` values still take precedence. For example:

```json
{
  "processor_kwargs": {
    "size": {"shortest_edge": 448},
    "crop_size": {"height": 448, "width": 448}
  }
}
```
