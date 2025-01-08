from dataclasses import dataclass, field
from Nexus.abc.evaluation import AbsEvalArguments
from Nexus.abc.training.arguments import AbsModelArguments
from Nexus.training.embedder.recommendation.arguments import DataArguments


@dataclass
class RecommenderEvalArgs(AbsEvalArguments):
    """
    Base class for evaluation arguments.
    """
    retriever_data_path: str = field(
        default=None,
        metadata={}
    )
    ranker_data_path: str = field(
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
        retriever_eval_data_arguments = DataArguments.from_json(self.retriever_data_path)
        self.retriever_item_batch_size = retriever_eval_data_arguments.item_batch_size

@dataclass
class RecommenderEvalModelArgs(AbsModelArguments):
    """
    Base class for model arguments during evaluation.
    """        
    retriever_ckpt_path: str = None
    ranker_ckpt_path: str = None
    
        