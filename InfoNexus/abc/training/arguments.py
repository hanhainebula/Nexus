from dataclasses import dataclass, field

from InfoNexus.abc.arguments import AbsArguments
from transformers import TrainingArguments

@dataclass
class AbsDataArguments(AbsArguments):
    train_data: str = field(
        default=None,
        metadata={"help": "Path to the data directory."}
    )
    eval_data: str = field(
        default=None,
        metadata={"help": "Path to the data directory."}
    )
    data_cache_dir: str = field(
        default=None,
        metadata={"help": "Path to the cache directory."}
    )


@dataclass
class AbsModelArguments(AbsArguments):
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Name or path to the model."}
    )
    model_cache_dir: str = field(
        default=None,
        metadata={"help": "Path to the cache directory."}
    )


@dataclass
class AbsTrainingArguments(AbsArguments, TrainingArguments):
    pass
