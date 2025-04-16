import os
from typing import Optional
from dataclasses import dataclass, field
from Nexus.abc.training.arguments import AbsDataArguments,  AbsModelArguments, AbsTrainingArguments
from transformers import TrainingArguments


@dataclass
class AbsEmbedderModelArguments(AbsModelArguments):
    """
    Abstract class for model arguments.
    """

    model_name_or_path: str = field(
        metadata={"help": "The model checkpoint for initialization."}
    )
    cache_dir: str = field(
        default=None,
        metadata={"help": "Where do you want to store the pre-trained models downloaded from s3."}
    )


@dataclass
class AbsEmbedderDataArguments(AbsDataArguments):
    """
    Abstract class for data arguments.
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


    def __post_init__(self):
        if not isinstance(self.train_data, list):
            self.train_data = self.train_data.split()
            
        for train_dir in self.train_data:
            if not os.path.exists(train_dir):
                raise FileNotFoundError(f"cannot find file: {train_dir}, please set a true path")


@dataclass
class AbsEmbedderTrainingArguments(AbsTrainingArguments):
    pass
