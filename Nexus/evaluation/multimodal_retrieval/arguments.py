import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

from Nexus.abc.arguments import AbsArguments
from Nexus.abc.evaluation import AbsEvalArguments


def load_config(file_path, config_class):
    if hasattr(config_class, "from_json"):
        return config_class.from_json(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return config_class(**data)


@dataclass
class MultimodalRetrievalEvalArgs(AbsEvalArguments):
    dataset_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset directory containing `corpus.jsonl`, `<split>_queries.jsonl`, and `<split>_qrels.jsonl`."},
    )
    media_root: Optional[str] = field(
        default=None,
        metadata={"help": "Optional default root for relative media paths in the eval jsonl files."},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Optional image root that overrides `media_root` for images/pages."},
    )
    video_root: Optional[str] = field(
        default=None,
        metadata={"help": "Optional video root that overrides `media_root` for raw videos."},
    )
    dataset_names: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Optional list of dataset subdirectories to evaluate.", "nargs": "+"},
    )
    splits: List[str] = field(
        default_factory=lambda: ["test"],
        metadata={"help": "Evaluation splits.", "nargs": "+"},
    )
    corpus_embd_save_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Optional cache directory for corpus embeddings."},
    )
    search_top_k: int = field(default=1000)
    cache_path: Optional[str] = field(default=None)
    token: Optional[str] = field(default_factory=lambda: os.getenv("HF_TOKEN", None))
    ignore_identical_ids: bool = field(default=False)
    force_redownload: bool = field(default=False)
    k_values: List[int] = field(
        default_factory=lambda: [1, 3, 5, 10, 100, 1000],
        metadata={"help": "Metric cutoffs.", "nargs": "+"},
    )
    eval_output_method: str = field(
        default="markdown",
        metadata={"help": "How to persist summary metrics.", "choices": ["json", "markdown"]},
    )
    eval_output_path: str = field(default="./eval_results.md")
    eval_metrics: List[str] = field(
        default_factory=lambda: ["ndcg_at_10", "recall_at_10"],
        metadata={"help": "Summary metrics.", "nargs": "+"},
    )


@dataclass
class MultimodalRetrievalEvalModelArgs(AbsArguments):
    embedder_name_or_path: str = field(metadata={"help": "Path or name of the multimodal embedder.", "required": True})
    processor_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional processor path. Defaults to the model path."},
    )
    model_type: str = field(default="auto", metadata={"help": "Explicit model type override."})
    normalize_embeddings: bool = field(default=True)
    pooling_method: str = field(default="last_token")
    use_fp16: bool = field(default=False)
    devices: Optional[List[str]] = field(default=None, metadata={"help": "Devices to use.", "nargs": "+"})
    query_instruction_for_retrieval: Optional[str] = field(default=None)
    query_instruction_format_for_retrieval: str = field(default="{}{}")
    passage_instruction_for_retrieval: Optional[str] = field(default=None)
    passage_instruction_format_for_retrieval: str = field(default="{}{}")
    trust_remote_code: bool = field(default=False)
    cache_dir: Optional[str] = field(default=None)
    token: Optional[str] = field(default_factory=lambda: os.getenv("HF_TOKEN", None))
    embedder_batch_size: int = field(default=8)
    embedder_query_max_length: int = field(default=512)
    embedder_passage_max_length: int = field(default=1024)
    use_chat_template: bool = field(default=True)
    torch_dtype: Optional[str] = field(default=None)
    attn_implementation: Optional[str] = field(default=None)
