from abc import ABC, abstractmethod
from typing import Optional

import torch

from .dataset import AbsDataset
from .modeling import AbsModel
from .arguments import AbsTrainingArguments


class AbsTrainer(ABC):
    def __init__(
        self,
        model: AbsModel,
        train_args: AbsTrainingArguments,
    ):
        self.model = model
        self.train_args = train_args

    @abstractmethod
    def save_model(self, output_dir: Optional[str] = None, *args, **kwargs):
        pass

    @abstractmethod
    def train(self, train_dataset: AbsDataset, eval_dataset: AbsDataset, *args, **kwargs):
        pass

    @torch.no_grad()
    @abstractmethod
    def evaluate(self, eval_dataset: AbsDataset, *args, **kwargs):
        pass
