import os
import logging
from typing import Tuple
from transformers import (
    AutoModelForSequenceClassification, AutoConfig,
    AutoTokenizer, PreTrainedTokenizer
)
from transformers import set_seed
from UniRetrieval.abc.training.reranker import AbsRerankerRunner, AbsRerankerModel
from .arguments import AbsTextRerankerModelArguments, AbsTextRerankerDataArguments, AbsTextRerankerTrainingArguments
from .modeling import CrossEncoderModel
from .trainer import EncoderOnlyRerankerTrainer
from .datasets import AbsTextRerankerTrainDataset, AbsTextRerankerCollator

logger = logging.getLogger(__name__)


class EncoderOnlyRerankerRunner(AbsRerankerRunner):
    """
    Encoder only reranker runner for finetuning.
    """

    def __init__(
        self,
        model_args: AbsTextRerankerModelArguments,
        data_args: AbsTextRerankerDataArguments,
        training_args: AbsTextRerankerTrainingArguments
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
        )

        if self.training_args.gradient_checkpointing:
            model.enable_input_require_grads()
        return model

    def load_trainer(self) -> EncoderOnlyRerankerTrainer:
        """Load the trainer.

        Returns:
            EncoderOnlyRerankerTrainer: Loaded trainer instance.
        """
        trainer = EncoderOnlyRerankerTrainer(
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
        if self.model_args.model_type == 'encoder':
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
        if self.model_args.model_type == 'encoder':
            RerankerCollator = AbsTextRerankerCollator
            
        data_collator = RerankerCollator(
            tokenizer=self.tokenizer,
            query_max_len=self.data_args.query_max_len,
            passage_max_len=self.data_args.passage_max_len,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            padding=True,
            return_tensors="pt"
        )
        return data_collator

