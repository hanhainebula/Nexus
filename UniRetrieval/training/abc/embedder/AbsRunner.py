import os
import logging
from pathlib import Path
from typing import Tuple, Union
from abc import ABC, abstractmethod
from transformers import set_seed, PreTrainedTokenizer


from .AbsArguments import (
    AbsEmbedderModelArguments,
    AbsEmbedderDataArguments,
    AbsEmbedderTrainingArguments
)
from .AbsTrainer import AbsEmbedderTrainer
from .AbsModeling import AbsEmbedderModel
from .AbsDataset import (
    AbsEmbedderTrainDataset, AbsEmbedderCollator)
from UniRetrieval.abc.training.runner import AbsRunner

logger = logging.getLogger(__name__)


class AbsEmbedderRunner(AbsRunner):
    """Abstract class to run embedding model fine-tuning.

    Args:
        model_args (AbsEmbedderModelArguments): Model arguments
        data_args (AbsEmbedderDataArguments): Data arguments.
        training_args (AbsEmbedderTrainingArguments): Training arguments.
    """
    def __init__(
        self,
        model_args: AbsEmbedderModelArguments,
        data_args: AbsEmbedderDataArguments,
        training_args: AbsEmbedderTrainingArguments
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

        if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
        ):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        )
        logger.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            training_args.local_rank,
            training_args.device,
            training_args.n_gpu,
            bool(training_args.local_rank != -1),
            training_args.fp16,
        )
        logger.info("Training/evaluation parameters %s", training_args)
        logger.info("Model parameters %s", model_args)
        logger.info("Data parameters %s", data_args)

        # Set seed
        set_seed(training_args.seed)

        self.tokenizer, self.model = self.load_tokenizer_and_model()
        self.train_dataset = self.load_train_dataset()
        self.data_collator = self.load_data_collator()
        self.trainer = self.load_trainer()
        
    @abstractmethod
    def load_model(self, *args, **kwargs) -> AbsEmbedderModel:
        pass
        
    @abstractmethod
    def load_trainer(self) -> AbsEmbedderTrainer:
        """Abstract method to load the trainer.

        Returns:
            AbsEmbedderTrainer: The loaded trainer instance.
        """
        pass

    @abstractmethod
    def load_train_dataset(self) -> AbsEmbedderTrainDataset:
        """Loads the training dataset based on data arguments.

        Returns:
            AbsEmbedderTrainDataset: The loaded dataset instance.
        """
        pass

    def load_dataset(self, *args, **kwargs) -> AbsEmbedderTrainDataset:
        return self.load_train_dataset()

    @abstractmethod
    def load_data_collator(self) -> AbsEmbedderCollator:
        """Loads the appropriate data collator.

        Returns:
            AbsEmbedderCollator: Loaded data collator.
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


