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
- Config-file entrypoints: [`training/run_single_device.sh`](./training/run_single_device.sh) and [`evaluation/run_local.sh`](./evaluation/run_local.sh)
- CLI-style training: [`training/run_qwen_vl_lora.sh`](./training/run_qwen_vl_lora.sh)
- CLI-style evaluation: [`evaluation/run_local_eval.sh`](./evaluation/run_local_eval.sh)
- Inference: [`inference/encode_demo.py`](./inference/encode_demo.py)
- Data conversion tools: `tools/multimodal_retrieval/`
