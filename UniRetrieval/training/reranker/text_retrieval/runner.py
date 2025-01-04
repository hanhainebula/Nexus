import os
import logging
from typing import Optional

import torch
from transformers import (
    AutoModelForSequenceClassification, AutoConfig,
    AutoTokenizer,
    set_seed
)

from UniRetrieval.abc.training.reranker import (
    AbsRerankerRunner, AbsRerankerModel, AbsRerankerTrainDataset, AbsRerankerTrainer
)

from .arguments import TextRerankerModelArguments, TextRerankerDataArguments, TextRerankerTrainingArguments
from .modeling import CrossEncoderModel
from .trainer import TextRerankerTrainer
from .dataset import AbsTextRerankerTrainDataset, AbsTextRerankerCollator

logger = logging.getLogger(__name__)


class TextRerankerRunner(AbsRerankerRunner):
    """
    Encoder only reranker runner for finetuning.
    """

    def __init__(
        self,
        model_args: TextRerankerModelArguments,
        data_args: TextRerankerDataArguments,
        training_args: TextRerankerTrainingArguments,
        model: Optional[AbsRerankerModel] = None,
        train_dataset: Optional[AbsRerankerTrainDataset] = None,
        trainer: Optional[AbsRerankerTrainer] = None,
        loss_function: Optional[torch.nn.Module] = None,
        score_function: Optional[torch.nn.Module] = None
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
        self.loss_function = loss_function
        self.score_function = score_function
        self.model = model if model is not None else self.load_model()
        self.tokenizer=self.model.tokenizer
        self.train_dataset = train_dataset if train_dataset is not None else self.load_dataset()
        self.data_collator = self.load_data_collator()
        self.trainer = trainer if trainer is not None else self.load_trainer()
   
    def load_model(self) -> CrossEncoderModel:
        """Load the tokenizer and model.

        Returns:
            Tuple[PreTrainedTokenizer, AbsEmbedderModel]: Tokenizer and model instances.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.token,
            trust_remote_code=self.model_args.trust_remote_code
        )

        num_labels = 1
        config = AutoConfig.from_pretrained(
            self.model_args.config_name if self.model_args.config_name else self.model_args.model_name_or_path,
            num_labels=num_labels,
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.token,
            trust_remote_code=self.model_args.trust_remote_code,
        )
        logger.info('Config: %s', config)

        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_args.model_name_or_path,
            config=config,
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.token,
            from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
            trust_remote_code=self.model_args.trust_remote_code
        )

        model = CrossEncoderModel(
            base_model,
            tokenizer=tokenizer,
            train_batch_size=self.training_args.per_device_train_batch_size,
            loss_function=self.loss_function,
            score_function=self.score_function
        )

        if self.training_args.gradient_checkpointing:
            model.enable_input_require_grads()
        return model

    def load_trainer(self) -> TextRerankerTrainer:
        """Load the trainer.

        Returns:
            EncoderOnlyRerankerTrainer: Loaded trainer instance.
        """
        trainer = TextRerankerTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer
        )
        return trainer

    def load_dataset(self) -> AbsTextRerankerTrainDataset:
        """Loads the training dataset based on data arguments.

        Returns:
        """
        train_dataset = AbsTextRerankerTrainDataset(
            args=self.data_args,
            tokenizer=self.tokenizer
        )
        return train_dataset

    def load_data_collator(self) -> AbsTextRerankerCollator:
        """Loads the appropriate data collator.

        Returns:
            AbsRerankerCollator: Loaded data collator.
        """
        data_collator = AbsTextRerankerCollator(
            tokenizer=self.tokenizer,
            query_max_len=self.data_args.query_max_len,
            passage_max_len=self.data_args.passage_max_len,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            padding=True,
            return_tensors="pt"
        )
        return data_collator
