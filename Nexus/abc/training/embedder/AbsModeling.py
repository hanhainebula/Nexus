import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoTokenizer

import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Union
from Nexus.abc.training.modeling import AbsModelOutput, AbsEmbedder

logger = logging.getLogger(__name__)


@dataclass
class EmbedderOutput(AbsModelOutput):
    """
    Output information returned by the model.
    """
    pass

class AbsEmbedderModel(AbsEmbedder, nn.Module):
    """Abstract class of embedding model for training.

    Args:
        base_model: The base model to train on.
        tokenizer (AutoTokenizer, optional): The tokenizer to use. Defaults to ``None``.
        negatives_cross_device (bool, optional): If True, will compute cross devices negative loss. Defaults to ``False``.
        temperature (float, optional): Temperature to control the scale of scores. Defaults to ``1.0``.
        sub_batch_size (int, optional): Sub-batch size during encoding. If negative, will not split to sub-batch.
            Defaults to ``-1``.
        kd_loss_type (str, optional): Type of knowledge distillation loss. Defaults to ``"kl_div"``.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_modules()

    def init_modules(self):
        self.loss_function=self.get_loss_function()
        self.score_function=self.get_score_function()
        
    @abstractmethod
    def get_loss_function(self):
        pass
    
    @abstractmethod
    def get_score_function(self):
        pass

    @abstractmethod
    def compute_loss(self, batch, *args, **kwargs):
        """Abstract method compute the loss.

        Args:
            scores (torch.Tensor): Computed score.
            target (torch.Tensor): The target value.
        """
        pass

    @abstractmethod
    def compute_score(self, *args, **kwargs):
        """Abstract method to compute the score.
        """
        pass

    @abstractmethod
    def save(self, output_dir: str):
        """Abstract method to save the model.

        Args:
            output_dir (str): Directory for saving the model.
        """
        pass

    def forward(self, batch, *args, **kwargs):
        """The computation performed at every call.

        Args:
            Datas after datacollator

        Returns:
            EmbedderOutput: Output of the forward call of model.
        """
        return self.compute_loss(batch,*args, **kwargs)