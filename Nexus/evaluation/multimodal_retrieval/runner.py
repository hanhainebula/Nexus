import json
import logging
import os

from Nexus.abc.evaluation import AbsEvalRunner
from Nexus.evaluation.text_retrieval.evaluator import TextRetrievalAbsEvaluator
from Nexus.inference.embedder.multimodal_retrieval import MultimodalEmbedder

from .arguments import MultimodalRetrievalEvalArgs, MultimodalRetrievalEvalModelArgs
from .data_loader import MultimodalRetrievalEvalDataLoader
from .evaluator import MultimodalRetrievalAbsEvaluator
from .searcher import MultimodalRetrievalEvalDenseRetriever

logger = logging.getLogger(__name__)


class MultimodalRetrievalEvalRunner(AbsEvalRunner):
    def __init__(
        self,
        eval_args: MultimodalRetrievalEvalArgs,
        model_args: MultimodalRetrievalEvalModelArgs,
    ):
        self.eval_args = eval_args
        self.model_args = model_args
        self.retriever = self.load_retriever()
        self.data_loader = self.load_data_loader()
        self.evaluator = self.load_evaluator()

    @staticmethod
    def get_model(model_args: MultimodalRetrievalEvalModelArgs) -> MultimodalEmbedder:
        return MultimodalEmbedder(
            model_name_or_path=model_args.embedder_name_or_path,
            processor_name_or_path=model_args.processor_name_or_path,
            model_type=model_args.model_type,
            normalize_embeddings=model_args.normalize_embeddings,
            use_fp16=model_args.use_fp16,
            query_instruction_for_retrieval=model_args.query_instruction_for_retrieval,
            query_instruction_format=model_args.query_instruction_format_for_retrieval,
            passage_instruction_for_retrieval=model_args.passage_instruction_for_retrieval,
            passage_instruction_format=model_args.passage_instruction_format_for_retrieval,
            devices=model_args.devices,
            pooling_method=model_args.pooling_method,
            trust_remote_code=model_args.trust_remote_code,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            batch_size=model_args.embedder_batch_size,
            query_max_length=model_args.embedder_query_max_length,
            passage_max_length=model_args.embedder_passage_max_length,
            use_chat_template=model_args.use_chat_template,
            torch_dtype=model_args.torch_dtype,
            attn_implementation=model_args.attn_implementation,
        )

    def load_retriever(self) -> MultimodalRetrievalEvalDenseRetriever:
        embedder = self.get_model(self.model_args)
        return MultimodalRetrievalEvalDenseRetriever(
            embedder=embedder,
            search_top_k=self.eval_args.search_top_k,
        )

    def load_data_loader(self) -> MultimodalRetrievalEvalDataLoader:
        return MultimodalRetrievalEvalDataLoader(
            eval_name=self.eval_args.eval_name,
            dataset_dir=self.eval_args.dataset_dir,
            media_root=self.eval_args.media_root,
            image_root=self.eval_args.image_root,
            video_root=self.eval_args.video_root,
            cache_dir=self.eval_args.cache_path,
            token=self.eval_args.token,
            force_redownload=self.eval_args.force_redownload,
        )

    def load_evaluator(self) -> MultimodalRetrievalAbsEvaluator:
        return MultimodalRetrievalAbsEvaluator(
            eval_name=self.eval_args.eval_name,
            data_loader=self.data_loader,
            overwrite=self.eval_args.overwrite,
        )

    def evaluate_metrics(self, search_results_save_dir: str, output_method: str, output_path: str, metrics):
        eval_results_dict = {}
        for model_name in sorted(os.listdir(search_results_save_dir)):
            model_search_results_save_dir = os.path.join(search_results_save_dir, model_name)
            if not os.path.isdir(model_search_results_save_dir):
                continue
            eval_results_path = os.path.join(model_search_results_save_dir, "NoReranker", "EVAL", "eval_results.json")
            if not os.path.exists(eval_results_path):
                continue
            with open(eval_results_path, "r", encoding="utf-8") as f:
                eval_results_dict[model_name] = {"NoReranker": json.load(f)}

        if output_method == "json":
            TextRetrievalAbsEvaluator.output_eval_results_to_json(eval_results_dict, output_path)
        elif output_method == "markdown":
            TextRetrievalAbsEvaluator.output_eval_results_to_markdown(eval_results_dict, output_path, metrics)
        else:
            raise ValueError(f"Invalid output method: {output_method}")

    def run(self):
        if self.eval_args.dataset_names is None:
            dataset_names = self.data_loader.available_dataset_names()
        else:
            dataset_names = self.data_loader.check_dataset_names(self.eval_args.dataset_names)

        if len(dataset_names) == 0:
            self.evaluator(
                splits=self.eval_args.splits,
                search_results_save_dir=self.eval_args.eval_output_dir,
                retriever=self.retriever,
                corpus_embd_save_dir=self.eval_args.corpus_embd_save_dir,
                ignore_identical_ids=self.eval_args.ignore_identical_ids,
                k_values=self.eval_args.k_values,
            )
        else:
            for dataset_name in dataset_names:
                self.evaluator(
                    splits=self.eval_args.splits,
                    search_results_save_dir=self.eval_args.eval_output_dir,
                    retriever=self.retriever,
                    corpus_embd_save_dir=self.eval_args.corpus_embd_save_dir,
                    ignore_identical_ids=self.eval_args.ignore_identical_ids,
                    k_values=self.eval_args.k_values,
                    dataset_name=dataset_name,
                )

        self.evaluate_metrics(
            search_results_save_dir=self.eval_args.eval_output_dir,
            output_method=self.eval_args.eval_output_method,
            output_path=self.eval_args.eval_output_path,
            metrics=self.eval_args.eval_metrics,
        )
