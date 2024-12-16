import os
import math
import random
import logging
import datasets
import numpy as np
import torch.distributed as dist
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import (
    PreTrainedTokenizer, 
    DataCollatorWithPadding,
    BatchEncoding,
    DataCollatorForSeq2Seq
)
from typing import List

from UniRetrieval.abc.training.dataset import AbsDataset
from .AbsArguments import AbsRerankerDataArguments, AbsRerankerTrainingArguments

logger = logging.getLogger(__name__)


class AbsRerankerTrainDataset(AbsDataset):
    """Abstract class for reranker training dataset.

    Args:
        args (AbsRerankerDataArguments): Data arguments.
        tokenizer (PreTrainedTokenizer): Tokenizer to use.
    """
    def __init__(
        self,
        args: AbsRerankerDataArguments):
        pass

@dataclass
class AbsRerankerCollator(DataCollatorWithPadding):
    """
    The abstract reranker collator.
    """
    def __call__(self, features):
        return features

# remove Abs LLM reranker class