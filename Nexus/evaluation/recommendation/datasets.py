import os
import logging
import datasets
import subprocess
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, Dataset

from typing import List, Optional, Union
from Nexus.training.embedder.recommendation.dataset import ShardedDataset, get_datasets
from Nexus.abc.evaluation import AbsEvalDataLoader, AbsEvalDataLoaderArguments
from Nexus.evaluation.recommendation.arguments import RecommenderEvalArgs, RecommenderEvalModelArgs
import pandas as pd
from Nexus.modules.dataset import get_client
from Nexus.modules.arguments import DataAttr4Model
from loguru import logger
from Nexus.training.embedder.recommendation.dataset import AbsRecommenderEmbedderCollator

from torchrec.distributed import DistributedModelParallel
from dynamic_embedding.wrappers import wrap_dataloader, wrap_dataset
    

class RecommenderEvalDataLoader(AbsEvalDataLoader, DataLoader):
    def __init__(
        self,
        config: RecommenderEvalArgs,
        model_args: RecommenderEvalModelArgs,
    ):
        self.config = config
        self.eval_dataset: ShardedDataset = None
        self.data_attr: DataAttr4Model = None
        self.collator = AbsRecommenderEmbedderCollator()
        self.retriever_eval_loader = None
        self.ranker_eval_loader = None
        self.item_loader = None
        
        if model_args.retriever_ckpt_path is not None:
            (self.retriever_train_dataset, self.retriever_eval_dataset), self.retriever_data_attr = get_datasets(config.retriever_data_path)
            self.retriever_eval_loader = DataLoader(
                self.retriever_eval_dataset, 
                batch_size=config.eval_batch_size,
                collate_fn=self.collator
            )
            self.item_loader = DataLoader(
                self.retriever_train_dataset.item_feat_dataset, 
                batch_size=config.retriever_item_batch_size,
            )
        
        if model_args.ranker_ckpt_path is not None:
            (self.ranker_train_dataset, self.ranker_eval_dataset), self.ranker_data_attr = get_datasets(config.ranker_data_path)
            self.ranker_eval_loader = DataLoader(
                self.ranker_eval_dataset, 
                batch_size=config.eval_batch_size,
                collate_fn=self.collator
            )
            