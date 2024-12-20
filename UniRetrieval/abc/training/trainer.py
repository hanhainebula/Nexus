from abc import ABC, abstractmethod
from typing import Optional

import torch

from .dataset import AbsDataset
from .modeling import AbsModel
from .arguments import AbsTrainingArguments
from transformers import Trainer

class AbsTrainer(Trainer):
    @abstractmethod
    def save_model(self, output_dir: Optional[str] = None, *args, **kwargs):
        pass

    @abstractmethod
    def train(self, train_dataset: AbsDataset=None, eval_dataset: AbsDataset=None, *args, **kwargs):
        pass

    @torch.no_grad()
    @abstractmethod
    def evaluate(self, eval_dataset: AbsDataset, *args, **kwargs):
        pass
