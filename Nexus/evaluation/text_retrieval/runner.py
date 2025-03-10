import os
import json
import logging
from typing import List, Union, Tuple

from Nexus import TextEmbedder, TextReranker, BaseEmbedderInferenceEngine, BaseRerankerInferenceEngine, AbsInferenceArguments
from .arguments import TextRetrievalEvalArgs, TextRetrievalEvalModelArgs
from .evaluator import TextRetrievalAbsEvaluator
from .searcher import TextRetrievalEvalDenseRetriever, TextRetrievalEvalReranker
from Nexus.abc.evaluation import AbsEvalRunner
from .data_loader import TextRetrievalEvalDataLoader

logger = logging.getLogger(__name__)

# The following code is modified from FlagEmbedding, licensed under the MIT License.
# Source: https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/abc/evaluation/data_loader.py
# Original copyright notice: 
# MIT License
# Copyright (c) 2022 staoxiao

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
class TextRetrievalEvalRunner(AbsEvalRunner):
    
    """
    Abstract class of evaluation runner.
    
    Args:
        eval_args (AbsEvalArgs): :class:AbsEvalArgs object with the evaluation arguments.
        model_args (AbsEvalModelArgs): :class:AbsEvalModelArgs object with the model arguments.
    """
    def __init__(
        self,
        eval_args: TextRetrievalEvalArgs,
        model_args: TextRetrievalEvalModelArgs,
    ):
        self.eval_args = eval_args
        self.model_args = model_args

        self.retriever, self.reranker = self.load_retriever_and_reranker()
        self.data_loader = self.load_data_loader()
        self.evaluator = self.load_evaluator()

    @staticmethod
    def get_models(model_args: TextRetrievalEvalModelArgs) -> Tuple[Union[TextEmbedder, BaseEmbedderInferenceEngine], Union[TextReranker, BaseRerankerInferenceEngine, None]]:
        """Get the embedding and reranker model

        Args:
            model_args (AbsEvalModelArgs): :class:AbsEvalModelArgs object with the model arguments.

        Returns:
            Tuple[FlagAutoModel, Union[FlagAutoReranker, None]]: A :class:FlagAutoModel object of embedding model, and 
                :class:FlagAutoReranker object of reranker model if path provided.
        """
        embedder_infer_mode=model_args.embedder_infer_mode
        base_embedder = TextEmbedder(
            model_name_or_path=model_args.embedder_name_or_path,
            normalize_embeddings=model_args.normalize_embeddings,
            pooling_method=model_args.pooling_method,
            use_fp16=model_args.use_fp16,
            query_instruction_for_retrieval=model_args.query_instruction_for_retrieval,
            query_instruction_format=model_args.query_instruction_format_for_retrieval,
            devices=model_args.devices,
            trust_remote_code=model_args.trust_remote_code,
            cache_dir=model_args.cache_dir,
            batch_size=model_args.embedder_batch_size,
            query_max_length=model_args.embedder_query_max_length,
            passage_max_length=model_args.embedder_passage_max_length,
        )
        if embedder_infer_mode is None:
            embedder = base_embedder
            embedder.model.config._name_or_path = model_args.embedder_name_or_path
            
        else:
            embedder_engine_args = AbsInferenceArguments(
                model_name_or_path=model_args.embedder_name_or_path,
                onnx_model_path=model_args.embedder_onnx_model_path,
                trt_model_path=model_args.embedder_trt_model_path,
                infer_mode=model_args.embedder_infer_mode,
                infer_device=model_args.devices,
                infer_batch_size=model_args.embedder_batch_size
            )
            
            embedder = BaseEmbedderInferenceEngine(embedder_engine_args, model=base_embedder)
            embedder.model.model.config._name_or_path = model_args.embedder_name_or_path
        
        
        reranker = None
        if model_args.reranker_name_or_path is not None:
            base_reranker = TextReranker(
                model_name_or_path=model_args.reranker_name_or_path,
                use_fp16=model_args.use_fp16,
                use_bf16=model_args.use_bf16,
                query_instruction_for_rerank=model_args.query_instruction_for_rerank,
                query_instruction_format=model_args.query_instruction_format_for_rerank,
                passage_instruction_for_rerank=model_args.passage_instruction_for_rerank,
                passage_instruction_format=model_args.passage_instruction_format_for_rerank,
                cache_dir=model_args.cache_dir,
                trust_remote_code=model_args.trust_remote_code,
                devices=model_args.devices,
                normalize=model_args.normalize,
                prompt=model_args.prompt,
                batch_size=model_args.reranker_batch_size,
                query_max_length=model_args.reranker_query_max_length,
                max_length=model_args.reranker_max_length,
            )
            
            reranker_infer_mode = model_args.reranker_infer_mode
            if reranker_infer_mode is None:
                reranker = base_reranker
                reranker.model.config._name_or_path = model_args.reranker_name_or_path
            
            else:
                reranker_engine_args = AbsInferenceArguments(
                    model_name_or_path=model_args.reranker_name_or_path,
                    onnx_model_path=model_args.reranker_onnx_model_path,
                    trt_model_path=model_args.reranker_trt_model_path,
                    infer_mode=model_args.reranker_infer_mode,
                    infer_device=model_args.devices,
                    infer_batch_size=model_args.reranker_batch_size
                )
            
                reranker = BaseRerankerInferenceEngine(reranker_engine_args, model=base_reranker)
                reranker.model.model.config._name_or_path = model_args.reranker_name_or_path
            
        return embedder, reranker

    def load_retriever_and_reranker(self) -> Tuple[TextRetrievalEvalDenseRetriever, Union[TextRetrievalEvalReranker, None]]:
        """Load retriever and reranker for evaluation

        Returns:
            Tuple[EvalDenseRetriever, Union[EvalReranker, None]]: A :class:EvalDenseRetriever object for retrieval, and a
                :class:EvalReranker object if reranker provided.
        """
        embedder, reranker = self.get_models(self.model_args)
        retriever = TextRetrievalEvalDenseRetriever(
            embedder,
            search_top_k=self.eval_args.search_top_k,
            overwrite=self.eval_args.overwrite
        )
        if reranker is not None:
            reranker = TextRetrievalEvalReranker(reranker, rerank_top_k=self.eval_args.rerank_top_k)
        return retriever, reranker

    def load_data_loader(self) -> TextRetrievalEvalDataLoader:
        """Load the data loader

        Returns:
            AbsEvalDataLoader: Data loader object for that specific task.
        """
        data_loader = TextRetrievalEvalDataLoader(
            eval_name=self.eval_args.eval_name,
            dataset_dir=self.eval_args.dataset_dir,
            cache_dir=self.eval_args.cache_path,
            token=self.eval_args.token,
            force_redownload=self.eval_args.force_redownload,
        )
        return data_loader

    def load_evaluator(self) -> TextRetrievalAbsEvaluator:
        """Load the evaluator for evaluation

        Returns:
            AbsEvaluator: the evaluator to run the evaluation.
        """
        evaluator = TextRetrievalAbsEvaluator(
            eval_name=self.eval_args.eval_name,
            data_loader=self.data_loader,
            overwrite=self.eval_args.overwrite,
        )
        return evaluator

    @staticmethod
    def evaluate_metrics(
        search_results_save_dir: str,
        output_method: str = "markdown",
        output_path: str = "./eval_dev_results.md",
        metrics: Union[str, List[str]] = ["ndcg_at_10", "recall_at_10"]
    ):
        """Evaluate the provided metrics and write the results.

        Args:
            search_results_save_dir (str): Path to save the search results.
            output_method (str, optional): Output results to `json` or `markdown`. Defaults to :data:`"markdown"`.
            output_path (str, optional): Path to write the output. Defaults to :data:`"./eval_dev_results.md"`.
            metrics (Union[str, List[str]], optional): metrics to use. Defaults to :data:`["ndcg_at_10", "recall_at_10"]`.

        Raises:
            FileNotFoundError: Eval results not found
            ValueError: Invalid output method
        """
        eval_results_dict = {}
        for model_name in sorted(os.listdir(search_results_save_dir)):
            model_search_results_save_dir = os.path.join(search_results_save_dir, model_name)
            if not os.path.isdir(model_search_results_save_dir):
                continue
            for reranker_name in sorted(os.listdir(model_search_results_save_dir)):
                reranker_search_results_save_dir = os.path.join(model_search_results_save_dir, reranker_name)
                if not os.path.isdir(reranker_search_results_save_dir):
                    continue
                eval_results_path = os.path.join(reranker_search_results_save_dir, 'EVAL', "eval_results.json")
                if os.path.exists(eval_results_path):
                    eval_results = json.load(open(eval_results_path, encoding='utf-8'))
                else:
                    raise FileNotFoundError(f"Eval results not found: {eval_results_path}")

                if model_name not in eval_results_dict:
                    eval_results_dict[model_name] = {}
                eval_results_dict[model_name][reranker_name] = eval_results

        if output_method == "json":
            TextRetrievalAbsEvaluator.output_eval_results_to_json(eval_results_dict, output_path)
        elif output_method == "markdown":
            TextRetrievalAbsEvaluator.output_eval_results_to_markdown(eval_results_dict, output_path, metrics)
        else:
            raise ValueError(f"Invalid output method: {output_method}. Available methods: ['json', 'markdown']")

    def run(self):
        """
        Run the whole evaluation.
        """
        if self.eval_args.dataset_names is None:
            dataset_names = self.data_loader.available_dataset_names()
        else:
            dataset_names = self.data_loader.check_dataset_names(self.eval_args.dataset_names)

        if len(dataset_names) == 0:
            logger.info(f"Running {self.eval_args.eval_name} evaluation on the default dataset.")
            self.evaluator(
                splits=self.eval_args.splits,
                search_results_save_dir=self.eval_args.eval_output_dir,
                retriever=self.retriever,
                reranker=self.reranker,
                corpus_embd_save_dir=self.eval_args.corpus_embd_save_dir,
                ignore_identical_ids=self.eval_args.ignore_identical_ids,
                k_values=self.eval_args.k_values
            )
            logger.info(f"{self.eval_args.eval_name} evaluation completed.")
        else:
            logger.info(f"Running {self.eval_args.eval_name} evaluation on the following dataset names: {dataset_names}")
            for dataset_name in dataset_names:
                logger.info(f"Running {self.eval_args.eval_name} evaluation on: {dataset_name}")
                self.evaluator(
                    splits=self.eval_args.splits,
                    search_results_save_dir=self.eval_args.eval_output_dir,
                    retriever=self.retriever,
                    reranker=self.reranker,
                    corpus_embd_save_dir=self.eval_args.corpus_embd_save_dir,
                    ignore_identical_ids=self.eval_args.ignore_identical_ids,
                    k_values=self.eval_args.k_values,
                    dataset_name=dataset_name,
                )
            logger.info(f"{self.eval_args.eval_name} evaluation on {dataset_names} completed.")

        logger.info("Start computing metrics.")
        self.evaluate_metrics(
            search_results_save_dir=self.eval_args.eval_output_dir,
            output_method=self.eval_args.eval_output_method,
            output_path=self.eval_args.eval_output_path,
            metrics=self.eval_args.eval_metrics
        )
