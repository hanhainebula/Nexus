import logging
import os
from typing import Tuple
from transformers import (
    AutoModel, AutoConfig,
    AutoTokenizer, PreTrainedTokenizer
)
from transformers import set_seed, PreTrainedTokenizer
from UniRetrieval.abc.training.embedder import AbsEmbedderRunner
from .arguments import RetrieverArguments, TrainingArguments, ModelArguments
from .datasets import 
from .modeling import MLPRetriever
from .trainer import RetrieverTrainer

logger = logging.getLogger(__name__)


class RetrieverRunner(AbsEmbedderRunner):
    """
    Finetune Runner for base embedding models.
    """
    
    # TODO 这里args都又包装了一层
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: AbsTextEmbedderDataArguments,
        training_args: TrainingArguments
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

        self.model = self.load_model()
        self.tokenizer=self.model.tokenizer
        self.train_dataset = self.load_dataset()
        self.data_collator = self.load_data_collator()
        self.trainer = self.load_trainer()

    
    def load_model(self) -> MLPRetriever:
        """Load tokenizer and model.

        Returns:
            AbsEmbedderModel: Tokenizer and model instances.
        """
        model=MLPRetriever(
            self.data_args,
            self.model_args,
            model_type='retriever')
        return model

    def load_trainer(self) -> RetrieverTrainer:
        """Load the trainer.

        Returns:
            EncoderOnlyEmbedderTrainer: Loaded trainer instance.
        """
        # TODO data_collator和tokenizer
    
        trainer = RetrieverTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer
        )

        return trainer
    
    def load_dataset(self) -> AbsTextEmbedderTrainDataset:
        """Loads the training dataset based on data arguments.

        Returns:
            : The loaded dataset instance.
        """
        pass


    def load_data_collator(self) -> AbsTextEmbedderCollator:
        """Loads the appropriate data collator.

        Returns:
            : Loaded data collator.
        """
        pass