from dataclasses import dataclass, field

from Nexus.abc.arguments import AbsArguments


@dataclass
class AbsEvalArguments(AbsArguments):
    eval_name: str = field(
        default=None,
        metadata={"help": "Name of the evaluation task."}
    )
    eval_output_dir: str = field(
        default=None,
        metadata={"help": "Directory to save the evaluation results."}
    )
    overwrite: bool = field(
        default=False,
        metadata={"help": "Overwrite the output directory if it exists."}
    )


@dataclass
class AbsEvalDataLoaderArguments(AbsArguments):
    eval_dataset_name_or_path: str = field(
        default=None,
        metadata={"help": "Name or path to the evaluation dataset."}
    )
    eval_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for evaluation."}
    )
