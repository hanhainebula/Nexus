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
            # collate_fn=AbsRecommenderEmbedderCollator
        )
        
        # if config['item_info']:
        #     self.item_col = config['item_info']['key']
        #     self.item_data_client = get_client(config['type'], config['item_info']['url'])
        # else:
        #     self.item_col = None
        #     self.item_data_client = None
            
    # def load_item_file(self):
    #     """Load all item data"""
    #     if self.config['item_info'] is None:
    #         return None
    #     data = self.item_data_client.load_file()
    #     if isinstance(data, pd.DataFrame):
    #         pass
    #     elif isinstance(data, dict):
    #         data = pd.DataFrame.from_dict(data, orient='index', columns=self.config['item_info']['columns'])
    #         # data.set_index(self.config['item_info']['key'], inplace=True)
    #     else:
    #         raise ValueError("Item data must be DataFrame or Dict")
    #     # data = {k: torch.tensor(list(v), dtype=torch.int64) for k, v in data.items()}
    #     if self.config['item_info'].get('use_cols', None) is not None:
    #         data = data[self.config['item_info']['use_cols']]
    #     # data.set_index(self.config['item_info']['key'], inplace=True)
    #     return data