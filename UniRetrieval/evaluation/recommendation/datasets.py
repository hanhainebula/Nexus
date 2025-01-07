import os
import logging
import datasets
import subprocess
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, Dataset

from typing import List, Optional, Union
from UniRetrieval.training.embedder.recommendation.datasets import ShardedDataset, get_datasets
from UniRetrieval.abc.evaluation import AbsEvalDataLoader, AbsEvalDataLoaderArguments
from UniRetrieval.evaluation.recommendation.arguments import RecommenderEvalArgs
import pandas as pd
from UniRetrieval.modules.dataset import get_client
from UniRetrieval.modules.arguments import DataAttr4Model
from loguru import logger
from UniRetrieval.training.embedder.recommendation.datasets import AbsRecommenderEmbedderCollator

from torchrec.distributed import DistributedModelParallel
from dynamic_embedding.wrappers import wrap_dataloader, wrap_dataset
    
class RecommenderEvalDataLoader(AbsEvalDataLoader, DataLoader):
    def __init__(
        self,
        config: RecommenderEvalArgs,
    ):
        self.config = config
        self.eval_dataset: ShardedDataset = None
        self.data_attr: DataAttr4Model = None
        self.collator = AbsRecommenderEmbedderCollator()
        (self.train_dataset, self.eval_dataset), self.data_attr = get_datasets(config.dataset_path)
        
        self.eval_loader = DataLoader(
            self.eval_dataset, 
            batch_size=config.eval_batch_size,
            collate_fn=self.collator
        )
        self.item_loader = DataLoader(
            self.train_dataset.item_feat_dataset, 
            batch_size=config.item_batch_size,
        )
        
class TDERecommenderEvalDataLoader(AbsEvalDataLoader, DataLoader):
    def __init__(
        self,
        model:DistributedModelParallel, 
        config: RecommenderEvalArgs,
    ):
        self.config = config
        self.eval_dataset: ShardedDataset = None
        self.data_attr: DataAttr4Model = None
        self.collator = AbsRecommenderEmbedderCollator()
        (self.train_dataset, self.eval_dataset), self.data_attr = get_datasets(config.dataset_path)
        
        # eval loader 
        self.eval_loader = DataLoader(
            self.eval_dataset, 
            batch_size=config.eval_batch_size,
            collate_fn=self.collator
        )
        self.eval_loader = wrap_dataloader(self.eval_loader, 
                                           model, model.module.tde_configs_dict)
        
        # item loader
        item_feat_dataset = wrap_dataset(self.train_dataset.item_feat_dataset, 
                                         model, model.module.tde_configs_dict)
        self.item_loader = DataLoader(
            item_feat_dataset, 
            batch_size=config.item_batch_size,
        )
        self.item_loader = wrap_dataloader(self.item_loader, 
                                           model, model.module.tde_configs_dict)