import os
from typing import Optional
from dataclasses import dataclass, field

from transformers import TrainingArguments
from UniRetrieval.abc.training.arguments import AbsDataArguments,  AbsModelArguments, AbsTrainingArguments

@dataclass
class AbsRerankerModelArguments(AbsModelArguments):
    """
    Abstract class for reranker model arguments.
    """

    model_name_or_path: str = field(
        metadata={"help": "The model checkpoint for initialization."}
    )




@dataclass
class AbsRerankerDataArguments(AbsDataArguments):
    """
    Abstract class for reranker data arguments.
    """
    train_data: str = field(
        default=None, metadata={
            "help": "One or more paths to training data. `query: str`, `pos: List[str]`, `neg: List[str]` are required in the training data.",
            "nargs": "+"
        }
    )
    cache_path: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the cached data"}
    )


@dataclass
class AbsRerankerTrainingArguments(AbsTrainingArguments, TrainingArguments):
    pass