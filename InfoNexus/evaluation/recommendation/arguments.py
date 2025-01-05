from dataclasses import dataclass, field
import os
from InfoNexus.abc.evaluation import AbsEvalArguments, AbsEvalDataLoaderArguments
from InfoNexus.training.embedder.recommendation import ModelArguments as RetrieverModelArguments
from InfoNexus.training.reranker.recommendation import ModelArguments as RankerModelArguments
from typing import List, Optional, Dict, Any
import json
from InfoNexus.abc.training.arguments import AbsDataArguments,  AbsModelArguments, AbsTrainingArguments
from InfoNexus.training.embedder.recommendation.arguments import DataArguments


@dataclass
class RecommenderEvalArgs(AbsEvalArguments):
    """
    Base class for evaluation arguments.
    """
    dataset_path: str = field(
        default=None,
        metadata={}
    )
    # type: str = field(
    #     default="hdfs", metadata={"help": "Item data client type: 'file' or 'hdfs'."}
    # )
    eval_output_dir: str = field(
        default="./search_results", metadata={"help": "Path to save results."}
    )

    # ================ for evaluation ===============
    cutoffs: int = field(
        default_factory=lambda: [1, 3, 5, 10, 100, 1000],
        metadata={"help": "k values for evaluation. Default: [1, 3, 5, 10, 100, 1000]", "nargs": "+"}
    )
    
    metrics: str = field(
        default_factory=lambda: ["ndcg_at_10", "recall_at_10"],
        metadata={"help": "The metrics to evaluate. Default: ['ndcg_at_10', 'recall_at_10']", "nargs": "+"}
    )
    # item_info: Optional[Dict[str, Any]] = None
    eval_batch_size: int = field(
        default=256, metadata={"help": "Evaluation batch size."}
    )
    # item_batch_size: int = field(
    #     default=256, metadata={"help": "Item batch size."}
    # )
    def __post_init__(self):
        eval_data_arguments = DataArguments.from_json(self.dataset_path)
        self.type = eval_data_arguments.type
        self.item_info = eval_data_arguments.item_info
        self.item_batch_size = eval_data_arguments.item_batch_size

@dataclass
class RecommenderEvalModelArgs(AbsModelArguments):
    """
    Base class for model arguments during evaluation.
    """        
    retriever_ckpt_path: str = ""
    ranker_ckpt_path: str = None
    
        