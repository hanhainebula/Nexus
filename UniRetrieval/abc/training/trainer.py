from abc import ABC, abstractmethod
from typing import Optional

import torch

from .dataset import AbsDataset
from .modeling import AbsModel
from .arguments import AbsTrainingArguments
from transformers import Trainer

class AbsTrainer(Trainer):
    @abstractmethod
    def _save(self, output_dir: Optional[str] = None, *args, **kwargs):
        return super()._save(output_dir, *args, **kwargs)

    @torch.no_grad()
    @abstractmethod
    def evaluate(self, eval_dataset: AbsDataset, *args, **kwargs):
        return super().evaluate(eval_dataset, **args, **kwargs)