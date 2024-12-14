from abc import ABC, abstractmethod

import torch

from .dataset import AbsDataset
from .modeling import AbsModel
from .arguments import AbsTrainingArguments


class AbsTrainer(ABC):
    def __init__(
        self,
        model: AbsModel,
        train_args: AbsTrainingArguments,
        train: bool = True
    ):
        self.model = model
        self.train_args = train_args
        self.train_mode = train
    
    @abstractmethod
    def _save(self, *args, **kwargs):
        pass

    @abstractmethod
    def train(self, train_dataset: AbsDataset, eval_dataset: AbsDataset, *args, **kwargs):
        pass

    @torch.no_grad()
    @abstractmethod
    def evaluate(self, eval_dataset: AbsDataset, *args, **kwargs):
        pass
