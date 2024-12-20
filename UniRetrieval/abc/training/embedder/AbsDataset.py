import os
import math
import random
import logging
import datasets
import numpy as np
import torch.distributed as dist
from dataclasses import dataclass
from abc import abstractmethod
from torch.utils.data import Dataset
from transformers import (
    PreTrainedTokenizer, 
    DataCollatorWithPadding,
    TrainerCallback,
    TrainerState,
    TrainerControl
)

from UniRetrieval.abc.training.dataset import AbsDataset
from .AbsArguments import AbsEmbedderDataArguments, AbsEmbedderTrainingArguments

logger = logging.getLogger(__name__)


class AbsEmbedderTrainDataset(AbsDataset):
    """Abstract class for training dataset.

    Args:
        args (AbsEmbedderDataArguments): Data arguments.
    """
    
    pass

@dataclass
class AbsEmbedderCollator(DataCollatorWithPadding):
    """
    The abstract embedder collator.
    """
    @abstractmethod
    def __call__(self, features):
        return super().call(features)

@dataclass
class CallbackOutput:
    save_checkpoint: str = None
    stop_training: bool = False

class AbsCallback():
    # TODO 结合 rec studio 的callback组件
    def __init__(self, train_dataset):
        self.train_dataset = train_dataset

    @abstractmethod
    def on_epoch_end(
        self,
        args: AbsEmbedderTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """
        Event called at the end of an epoch.
        """
        self.train_dataset.refresh_epoch()
        
