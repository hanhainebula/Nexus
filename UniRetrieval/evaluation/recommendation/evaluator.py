"""
Adapted from https://github.com/AIR-Bench/AIR-Bench/blob/0.1.0/air_benchmark/evaluation_utils/evaluator.py
"""
from collections import defaultdict

import json
from loguru import logger
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Union, Tuple

from UniRetrieval.training.embedder.recommendation.modeling import BaseRetriever
from UniRetrieval.training.reranker.recommendation.modeling import BaseRanker

from UniRetrieval.modules.metrics import get_eval_metrics
from UniRetrieval.abc.evaluation import AbsEvaluator
from .datasets import RecommenderEvalDataLoader
import torch
from torch.utils.data import DataLoader, Dataset
from .arguments import RecommenderEvalArgs
from UniRetrieval.modules.arguments import log_dict


class RecommenderAbsEvaluator(AbsEvaluator):
    """
    Base class of Evaluator.
    
    Args:
        eval_name (str): The experiment name of current evaluation.
        data_loader (AbsEvalDataLoader): The data_loader to deal with data.
        overwrite (bool): If true, will overwrite the existing results.
    """
    def __init__(
        self,
        data_loader: DataLoader,
        item_loader: DataLoader, 
        config: RecommenderEvalArgs,
        *args,
        **kwargs,
    ):
        self.eval_loader = data_loader
        self.item_loader = item_loader
        self.item_vectors = None
        self.item_ids = None
        self.config = config
    
    def __call__(
        self,
        retriever: BaseRetriever,
        ranker: Optional[BaseRanker] = None,
        *args,
        **kwargs,
    ):
        logger.info(f"Retriever evaluation begins.")
        metrics = self.evaluate(retriever)
        logger.info(f"Retriever evaluation done.")
        log_dict(logger, metrics)
        
        if ranker is not None:
            logger.info(f"Ranker evaluation begins.")
            metrics = self.evaluate(ranker)
            logger.info(f"Ranker evaluation done.")
            log_dict(logger, metrics)
        
    
    @torch.no_grad()
    def evaluate(self, model:Union[BaseRetriever, BaseRanker], *args, **kwargs) -> Dict:
        model.eval()
        if model.model_type == "retriever":
            logger.info(f"Updating item vectors...")
            self.item_vectors, self.item_ids = self.update_item_vectors(model)
            logger.info(f"Updating item vectors done...")
        eval_outputs = []
        eval_total_bs = 0
        for eval_step, eval_batch in enumerate(self.eval_loader):
            logger.info(f"Retriever evaluation step {eval_step + 1} begins..")
            eval_batch_size = eval_batch[list(eval_batch.keys())[0]].shape[0]
            metrics = self._eval_batch(model, eval_batch, *args, **kwargs)
            eval_outputs.append((metrics, eval_batch_size))
            eval_total_bs += eval_batch_size
            logger.info(f"Retriever evaluation step {eval_step + 1} done.")
            # log_dict(logger, metrics)
            if eval_step > 50:
                break
        metrics = self.eval_epoch_end(model, eval_outputs)
        self._total_eval_samples = eval_total_bs
        return metrics
    
    def _eval_batch(self, model:Union[BaseRetriever, BaseRanker], batch, *args, **kwargs) -> Dict:
        """ Evaluate the model on a batch, return metrics.

        Args:
            batch (Dict): The input batch.

        Returns:
            Dict: The metrics.
        """
        with torch.no_grad():
            model.eval()
            k = max(self.config.cutoffs) if self.config.cutoffs is not None else None
            if model.model_type == 'retriever':
                outputs = model.eval_step(batch, k=k, item_vectors=self.item_vectors, *args, **kwargs)
            else:
                outputs = model.eval_step(batch, k=k, *args, **kwargs)
            metrics: dict = self.compute_metrics(model, outputs)
            return metrics
    
    @torch.no_grad()
    def update_item_vectors(self, model:Union[BaseRetriever, BaseRanker]):
        model.eval()
        all_item_vectors, all_item_ids = [], []
        for item_batch in self.item_loader:
            item_vector = model.item_encoder(item_batch)
            all_item_vectors.append(item_vector)
            all_item_ids.append(item_batch[model.fiid])
        all_item_vectors = torch.cat(all_item_vectors, dim=0)
        all_item_ids = torch.cat(all_item_ids, dim=0).cpu()
        return all_item_vectors, all_item_ids
    
    def load_config(self, config: Union[Dict, str]) -> RecommenderEvalArgs:
        if config is None:
            return RecommenderEvalArgs()
        if isinstance(config, RecommenderEvalArgs):
            return config
        if isinstance(config, dict):
            config_dict = config
        elif isinstance(config, str):
            with open(config, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Config should be either a dictionary or a path to a JSON file, got {type(config)} instead.")
        return RecommenderEvalArgs.from_dict(config_dict)
    
    @torch.no_grad()
    def compute_metrics(self, model:Union[BaseRetriever, BaseRanker], output: Tuple):
        """ Compute the metrics given the output of the model.

        Args:
            output (Tuple): The output of the model.

        Returns:
            Dict: The computed metrics.
        """
        metrics: list = get_eval_metrics(self.config.metrics, model.model_type)
        cutoffs = self.config.cutoffs
        output_dict = {}
        if model.model_type == "retriever":
            for metric, func in metrics:
                for cutoff in cutoffs:
                    output_dict[f"{metric}@{cutoff}"] = func(*output, cutoff)
        else:
            output_dict = (output[0].cpu(), output[1].cpu())    # (pred, target)
        return output_dict
    
    @torch.no_grad()
    def eval_epoch_end(self, model:Union[BaseRetriever, BaseRanker], outputs: List[Tuple]) -> Dict:
        """ Aggregate the metrics from the evaluation batch.

        Args:
            outputs (List): The output of the evaluation batch. It is a list of tuples, 
                where the first element is the metrics (Dict) and the second element is the batch size.

        Returns:
            Dict: The aggregated metrics.
        """
        if model.model_type == "retriever":
            metric_list, bs = zip(*outputs)
            bs = torch.tensor(bs)
            out = defaultdict(list)
            for o in metric_list:
                for k, v in o.items():
                    out[k].append(v)
            for k, v in out.items():
                metric = torch.tensor(v)
                out[k] = ((metric * bs).sum() / bs.sum()).item()
            return out
        else:
            # ranker: AUC, Logloss
            out = {}
            output, bs = zip(*outputs)
            pred, target = zip(*output)
            pred = torch.cat(pred, dim=0)   # [N] or [N, K]
            target = torch.cat(target, dim=0)   # [N] or [N, K]
            metrics: list = get_eval_metrics(self.config.metrics, model.model_type)
            if pred.dim() == 2 and target.dim() == 2:
                # multi task
                if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                    labels = self.model.module.flabel
                else:
                    labels = self.model.flabel
                for i, task in enumerate(labels):
                    for metric, func in metrics:
                        _m = func(pred[:, i], target[:, i])
                        out[f"{metric}/{task}"] = _m.item() if isinstance(_m, torch.Tensor) else _m
                # overall metrics
                for metric, func in metrics:
                    avg_metric = sum([out[f"{metric}/{label}"] for label in labels]) / len(labels)
                    out[metric] = avg_metric
            else:   # single task
                for metric, func in metrics:
                    _m = func(pred, target)
                    out[metric] = _m if isinstance(_m , torch.Tensor) else _m 
            out = {m: v.item() if (isinstance(v, torch.Tensor) or isinstance(v, np.ndarray)) else v for m, v in out.items()}
            return out
            
    
    