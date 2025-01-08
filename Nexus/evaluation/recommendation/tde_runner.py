from loguru import logger
from typing import List, Union, Tuple

from .arguments import RecommenderEvalArgs, RecommenderEvalModelArgs
from .evaluator import RecommenderAbsEvaluator, TDERecommenderEvaluator
from Nexus.abc.evaluation import AbsEvalRunner

from Nexus.training.embedder.recommendation.modeling import BaseRetriever
from Nexus.training.reranker.recommendation.modeling import BaseRanker
from Nexus.training.embedder.recommendation.tde_modeling import TDEModel as RetrieverTDEModel
from Nexus.training.reranker.recommendation.tde_modeling import TDEModel as RankerTDEModel
from .datasets import RecommenderEvalDataLoader


class TDERecommenderEvalRunner(AbsEvalRunner):
    
    """
    Abstract class of evaluation runner.
    
    Args:
        eval_args (AbsEvalArgs): :class:AbsEvalArgs object with the evaluation arguments.
        model_args (AbsEvalModelArgs): :class:AbsEvalModelArgs object with the model arguments.
    """
    def __init__(
        self,
        eval_args: RecommenderEvalArgs,
        model_args: RecommenderEvalModelArgs,
    ):
        self.eval_args = eval_args
        self.model_args = model_args

        logger.info(f"Preparing...")
        self.retriever, self.ranker = self.load_retriever_and_ranker(model_args)
        logger.info(f"Loaded retriever and ranker.")
        self.data_loader = self.load_data_loader()
        logger.info(f"Loaded data.")
        self.evaluator = self.load_evaluator()
        logger.info(f"Loaded evaluator.")
        
    def load_retriever_and_ranker(self, model_args: RecommenderEvalModelArgs) -> Tuple[BaseRetriever, Union[BaseRanker, None]]:
        retriever = None
        if model_args.retriever_ckpt_path is not None:
            retriever = RetrieverTDEModel.from_pretrained(model_args.retriever_ckpt_path)
            retriever.eval()
    
        ranker = None
        if model_args.ranker_ckpt_path is not None:
            ranker = RankerTDEModel.from_pretrained(model_args.ranker_ckpt_path)
            ranker.eval()
            
        if retriever is None and ranker is None:
            raise ValueError("Both retriever and ranker cannot be None. At least one must be provided.")
            
        return retriever, ranker
        
    def load_data_loader(self):
        loader = RecommenderEvalDataLoader(self.eval_args, self.model_args)
        return loader

    def load_evaluator(self) -> RecommenderAbsEvaluator:
        evaluator = TDERecommenderEvaluator(
            retriever_data_loader=self.data_loader.retriever_eval_loader,
            ranker_data_loader=self.data_loader.ranker_eval_loader,
            item_loader=self.data_loader.item_loader,
            config=self.eval_args,
            model_config=self.model_args,
            retriever=self.retriever,
            ranker=self.ranker
        )
        return evaluator

    def run(self):
        """
        Run the whole evaluation.
        """
        self.evaluator(
            retriever=self.retriever,
            ranker=self.ranker,
        )