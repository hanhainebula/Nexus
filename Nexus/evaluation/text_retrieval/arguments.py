from dataclasses import dataclass, field
from typing import Optional
import os

from Nexus.abc.arguments import AbsArguments
from Nexus.abc.evaluation import AbsEvalArguments


@dataclass
class TextRetrievalEvalArgs(AbsEvalArguments):
    """
    Base class for evaluation arguments.
    """
    dataset_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "1) If you want to perform evaluation on your own dataset, you can provide the path to the dataset directory (must exists in local). "
            "The dataset directory should contain the following files: corpus.jsonl, <split>_queries.jsonl, <split>_qrels.jsonl, or contain multiple directories, each of which contains the following files: corpus.jsonl, <split>_queries.jsonl, <split>_qrels.jsonl."
            "2) If you want to perform evaluation on the datasets we provide evaluation APIs for, you can provide the path to saving the downloaded dataset. If you provide None, the dataset will be only downloaded to the cache directory."
        }
    )
    force_redownload: bool = field(
        default=False, metadata={"help": "Whether to force redownload the dataset. This is useful when you load dataset from remote and want to update the dataset."}
    )
    dataset_names: Optional[str] = field(
        default=None,
        metadata={
            "help": "The names of the datasets to evaluate. Default: None. If None, all available datasets will be evaluated. The name can be a specific dataset name (BEIR), a specific language (MIRACL), etc.",
            "nargs": "+"
        }
    )
    splits: str = field(
        default="test",
        metadata={"help": "Splits to evaluate. Default: test", "nargs": "+"}
    )
    corpus_embd_save_dir: str = field(
        default=None, metadata={"help": "Path to save corpus embeddings. If None, embeddings are not saved."}
    )
    search_top_k: int = field(
        default=1000, metadata={"help": "Top k for retrieving."}
    )
    rerank_top_k: int = field(default=100, metadata={"help": "Top k for reranking."})
    cache_path: str = field(
        default=None, metadata={"help": "Cache directory for loading datasets."}
    )
    token: str = field(
        default_factory=lambda: os.getenv('HF_TOKEN', None),
        metadata={"help": "The token to use when accessing the model."}
    )
    ignore_identical_ids: bool = field(
        default=False, metadata={"help": "whether to ignore identical ids in search results"}
    )
    # ================ for evaluation ===============
    k_values: int = field(
        default_factory=lambda: [1, 3, 5, 10, 100, 1000],
        metadata={"help": "k values for evaluation. Default: [1, 3, 5, 10, 100, 1000]", "nargs": "+"}
    )
    eval_output_method: str = field(
        default="markdown",
        metadata={"help": "The output method for evaluation results. Available methods: ['json', 'markdown']. Default: markdown.", "choices": ["json", "markdown"]}
    )
    eval_output_path: str = field(
        default="./eval_results.md", metadata={"help": "The path to save evaluation results."}
    )
    eval_metrics: str = field(
        default_factory=lambda: ["ndcg_at_10", "recall_at_10"],
        metadata={"help": "The metrics to evaluate. Default: ['ndcg_at_10', 'recall_at_10']", "nargs": "+"}
    )


@dataclass
class TextRetrievalEvalModelArgs(AbsArguments):
    """
    Base class for model arguments during evaluation.
    """
    embedder_name_or_path: str = field(
        metadata={"help": "The embedder name or path.", "required": True}
    )
    normalize_embeddings: bool = field(
        default=True, metadata={"help": "whether to normalize the embeddings"}
    )
    pooling_method: str = field(
        default="cls", metadata={"help": "The pooling method fot the embedder."}
    )
    use_fp16: bool = field(
        default=True, metadata={"help": "whether to use fp16 for inference"}
    )
    devices: Optional[str] = field(
        default=None, metadata={"help": "Devices to use for inference.", "nargs": "+"}
    )
    query_instruction_for_retrieval: Optional[str] = field(
        default=None, metadata={"help": "Instruction for query"}
    )
    query_instruction_format_for_retrieval: str = field(
        default="{}{}", metadata={"help": "Format for query instruction"}
    )
    trust_remote_code: bool = field(
        default=False, metadata={"help": "Trust remote code"}
    )
    reranker_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The reranker name or path."}
    )
    use_bf16: bool = field(
        default=False, metadata={"help": "whether to use bf16 for inference"}
    )
    query_instruction_for_rerank: Optional[str] = field(
        default=None, metadata={"help": "Instruction for query"}
    )
    query_instruction_format_for_rerank: str = field(
        default="{}{}", metadata={"help": "Format for query instruction"}
    )
    passage_instruction_for_rerank: Optional[str] = field(
        default=None, metadata={"help": "Instruction for passage"}
    )
    passage_instruction_format_for_rerank: str = field(
        default="{}{}", metadata={"help": "Format for passage instruction"}
    )
    cache_dir: str = field(
        default=None, metadata={"help": "Cache directory for models."}
    )
    # ================ for inference ===============
    embedder_batch_size: int = field(
        default=3000, metadata={"help": "Batch size for inference."}
    )
    reranker_batch_size: int = field(
        default=3000, metadata={"help": "Batch size for inference."}
    )
    embedder_query_max_length: int = field(
        default=512, metadata={"help": "Max length for query."}
    )
    embedder_passage_max_length: int = field(
        default=512, metadata={"help": "Max length for passage."}
    )
    reranker_query_max_length: Optional[int] = field(
        default=None, metadata={"help": "Max length for reranking."}
    )
    reranker_max_length: int = field(
        default=512, metadata={"help": "Max length for reranking."}
    )
    normalize: bool = field(
        default=False, metadata={"help": "whether to normalize the reranking scores"}
    )
    prompt: Optional[str] = field(
        default=None, metadata={"help": "The prompt for the reranker."}
    )
    embedder_infer_mode: str = field(
        default=None, metadata={'help':'inference mode of embedder', 'choices':['normal','onnx', 'tensorrt']}
    )
    reranker_infer_mode: str = field(
        default=None, metadata={'help':'inference mode of reranker', 'choices':['normal','onnx', 'tensorrt']}
    )
    embedder_onnx_model_path: str = field(
        default=None, metadata={"help" : "embedder onnx model path"}
    )
    embedder_trt_model_path: str = field(
        default=None, metadata={"help" : "embedder trt model path"}
    )
    reranker_onnx_model_path: str = field(
        default=None, metadata={"help" : "reranker onnx model path"}
    )
    reranker_trt_model_path: str = field(
        default=None, metadata={"help" : "reranker trt model path"}
    )
