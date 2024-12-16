import torch
from torch import nn, Tensor
from transformers import AutoTokenizer
from transformers.file_utils import ModelOutput

import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Union

from UniRetrieval.abc.training.modeling import AbsModelOutput, AbsModel, AbsReranker

logger = logging.getLogger(__name__)


@dataclass
class RerankerOutput(AbsModelOutput, ModelOutput):
    scores: Optional[Tensor] = None


class AbsRerankerModel(AbsReranker, nn.Module):
    """Abstract class of embedding model for training.

    Args:
        base_model: The base model to train on.
        tokenizer (AutoTokenizer, optional): The tokenizer to use. Defaults to ``None``.
        train_batch_size (int, optional): Batch size used for training. Defaults to ``4``.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self,*args, **kwargs):
        """The computation performed at every call.
        """
        pass

    def compute_loss(self, *args, **kwargs):
        """Compute the loss.
        """
        pass
    
    def save(self, output_dir: str):
        """Save the model.

        Args:
            output_dir (str): Directory for saving the model.
        """
        pass

    def save_pretrained(self, *args, **kwargs):
        """
        save model (and tokenizer if has)
        """
        pass
