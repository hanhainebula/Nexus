import os
import logging
from pathlib import Path
from typing import Tuple
from abc import ABC, abstractmethod
from transformers import set_seed, PreTrainedTokenizer


from .AbsArguments import (
    AbsRerankerModelArguments,
    AbsRerankerDataArguments,
    AbsRerankerTrainingArguments
)
from .AbsTrainer import AbsRerankerTrainer
from .AbsModeling import AbsRerankerModel
from .AbsDataset import (
    AbsRerankerTrainDataset, AbsRerankerCollator
)

from Nexus.abc.training.runner import AbsRunner


logger = logging.getLogger(__name__)


class AbsRerankerRunner(AbsRunner):
    """Abstract class to run reranker model fine-tuning.

    Args:
        model_args (AbsRerankerModelArguments): Model arguments
        data_args (AbsRerankerDataArguments): Data arguments.
        training_args (AbsRerankerTrainingArguments): Training arguments.
    """
    def __init__(
        self,
        model_args: AbsRerankerModelArguments,
        data_args: AbsRerankerDataArguments,
        training_args: AbsRerankerTrainingArguments
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args


    @abstractmethod
    def load_model(self, *args, **kwargs) -> AbsRerankerModel:
        pass

    @abstractmethod
    def load_trainer(self) -> AbsRerankerTrainer:
        """Abstract method to load the trainer.

        Returns:
            AbsRerankerTrainer: The loaded trainer instance.
        """
        pass

    @abstractmethod
    def load_dataset(self, *args, **kwargs):
        pass
    
    def load_data_collator(self) -> AbsRerankerCollator:
        """Loads the appropriate data collator.
        """
        pass

    def run(self):
        """
        Executes the training process.
        """
        Path(self.training_args.output_dir).mkdir(parents=True, exist_ok=True)

        # Training
        self.trainer.train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)
        self.trainer.save_model()
