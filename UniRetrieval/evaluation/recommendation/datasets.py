import os
import logging
import datasets
import subprocess
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, Dataset

from typing import List, Optional, Union
from UniRetrieval.training.embedder.recommendation.datasets import DailyDataset, DataAttr4Model, get_datasets
from UniRetrieval.abc.evaluation import AbsEvalDataLoader, AbsEvalDataLoaderArguments
from UniRetrieval.evaluation.recommendation.arguments import RecommenderEvalArgs
import pandas as pd
from UniRetrieval.modules.dataset import get_client
from loguru import logger
from UniRetrieval.training.embedder.recommendation.datasets import AbsRecommenderEmbedderCollator

    
class RecommenderEvalDataLoader(AbsEvalDataLoader, DataLoader):
    def __init__(
        self,
        config: RecommenderEvalArgs,
    ):
        self.config = config
        self.eval_dataset: DailyDataset = None
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