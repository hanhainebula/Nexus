from pathlib import Path
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from UniRetrieval.abc.training.reranker import AbsRerankerModelArguments, AbsRerankerDataArguments, AbsRerankerTrainingArguments
from UniRetrieval.abc.arguments import AbsArguments
from typing import Dict, Tuple
import torch
from accelerate import Accelerator
from loguru import logger as loguru_logger
from UniRetrieval.modules.arguments import DataAttr4Model, Statistics

@dataclass
class TrainingArguments(AbsRerankerTrainingArguments):
    train_batch_size: int = 512
    
    cutoffs: list = field(default_factory=lambda : [1, 5, 10])
    metrics: list = field(default_factory=lambda : ["ndcg", "recall"])
    
    checkpoint_best_ckpt: bool = True   # if true, save best model in earystop callback
    checkpoint_steps: int = 1000    # if none, save model per epoch; else save model by steps
    earlystop_metric: str = "auc"

@dataclass
class ModelArguments(AbsRerankerModelArguments):
    # model_name: str = None
    # embedding_dim: int = 10
    data_config: DataAttr4Model = None
    embedding_dim: int = 10
    mlp_layers: int = field(default=None, metadata={"nargs": "+"})
    prediction_layers: int = field(default=None, metadata={"nargs": "+"})
    # num_neg: int = 50
    activation: str = "relu"
    dropout: float = 0.3
    batch_norm: bool = True
    model_name_or_path: str = ''
    topk: int = 1
    

REQUIRED_DATA_CONFIG = [
    "name",
    "type",
    "url",
    "labels",
    "stats",
    "item_col",
    "context_features",
    "item_features",
    "train_settings/start_date",
    "train_settings/end_date",
    "test_settings/start_date",
    "test_settings/end_date",
]

DEFAULT_CONFIG = {
    "file_format": "auto",
    "date_format": "%Y-%m-%d",
    "user_sequential_info": None,
    "post_process": None,
    "filter_settings": None,
    "item_info": None,
}

@dataclass
class DataArguments(AbsRerankerDataArguments):
    # Required fields without default values
    name: str=None
    type: str=None
    url: str=None
    labels: str = field(default=None, metadata={"nargs": "+"})
    stats: Statistics=None
    item_col: str=None
    context_features: str = field(default=None, metadata={"nargs": "+"})
    item_features: str = field(default=None, metadata={"nargs": "+"})
    item_batch_size: int = 2048 # only used for retriever training

    # Optional fields with default values
    train_settings: Dict[str, datetime] = field(default=None,metadata={"required_keys": ["start_date", "end_date"]})
    test_settings: Dict[str, datetime] = field(default=None,metadata={"required_keys": ["start_date", "end_date"]})
    file_format: str = "auto"
    date_format: str = "%Y-%m-%d"
    user_sequential_info: Optional[Dict[str, Any]] = None
    post_process: Optional[Dict[str, Any]] = None
    filter_settings: Optional[Dict[str, Any]] = None
    item_info: Optional[Dict[str, Any]] = None
    seq_features: str = field(default=None, metadata={"nargs": "+"})


    def __post_init__(self):
        # Validate required keys in dictionaries after initialization
        for attr_name, required_keys in [
            ("train_settings", ["start_date", "end_date"]),
            ("test_settings", ["start_date", "end_date"])
        ]:
            attr_value = getattr(self, attr_name)
            missing_keys = [key for key in required_keys if key not in attr_value]
            if missing_keys:
                raise ValueError(f"Missing required keys in {attr_name}: {missing_keys}")
        
        # Convert string dates to datetime objects using the provided date format
        for settings in [self.train_settings, self.test_settings]:
            for key in ["start_date", "end_date"]:
                if isinstance(settings[key], str):
                    settings[key] = datetime.strptime(settings[key], self.date_format)

        # Remove whitespace from feature names and update seq_features if user_sequential_info is provided
        self.context_features = [feat.strip() for feat in self.context_features]
        self.item_features = [feat.strip() for feat in self.item_features]
        if self.user_sequential_info and 'use_cols' in self.user_sequential_info:
            self.seq_features = [feat.strip() for feat in self.user_sequential_info['use_cols']]