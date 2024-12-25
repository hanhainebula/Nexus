import logging
import os
from transformers import (
    AutoModel, AutoConfig,
    AutoTokenizer
)
from transformers import set_seed
from UniRetrieval.abc.training.embedder import AbsEmbedderRunner
from . import EncoderOnlyEmbedderModelArguments, EncoderOnlyEmbedderDataArguments, EncoderOnlyEmbedderTrainingArguments
from .datasets import AbsTextEmbedderTrainDataset, AbsTextEmbedderCollator, AbsEmbedderSameDatasetTrainDataset, AbsEmbedderSameDatasetCollator
from .callback import EmbedderTrainerCallbackForDataRefresh
from .modeling import BiEncoderOnlyEmbedderModel
from .trainer import EncoderOnlyEmbedderTrainer
from . arguments import BiEncoderOnlyEmbedderModelArguments

logger = logging.getLogger(__name__)


class EncoderOnlyEmbedderRunner(AbsEmbedderRunner):
    """
    Finetune Runner for base embedding models.
    """
    
    # TODO 这里args都又包装了一层
    def __init__(
        self,
        model_args: EncoderOnlyEmbedderModelArguments,
        data_args: EncoderOnlyEmbedderDataArguments,
        training_args: EncoderOnlyEmbedderTrainingArguments
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.encoder_only_model_args=BiEncoderOnlyEmbedderModelArguments(
            negatives_cross_device=self.training_args.negatives_cross_device,
            temperature=self.training_args.temperature,
            sub_batch_size=self.training_args.sub_batch_size,
            kd_loss_type=self.training_args.kd_loss_type,
            sentence_pooling_method=self.training_args.sentence_pooling_method,
            normalize_embeddings=self.training_args.normalize_embeddings
        )

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

    
    def load_model(self) -> BiEncoderOnlyEmbedderModel:
        """Load tokenizer and model.

        Returns:
            AbsEmbedderModel: Tokenizer and model instances.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.token,
            trust_remote_code=self.model_args.trust_remote_code
        )
        base_model = AutoModel.from_pretrained(
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

        model = BiEncoderOnlyEmbedderModel(
            base_model,
            tokenizer=tokenizer,
            model_args=self.encoder_only_model_args
        )

        if self.training_args.gradient_checkpointing:
            model.enable_input_require_grads()

        if self.training_args.fix_position_embedding:
            for k, v in model.named_parameters():
                if "position_embeddings" in k:
                    logging.info(f"Freeze the parameters for {k}")
                    v.requires_grad = False
                    
        return model

    def load_trainer(self) -> EncoderOnlyEmbedderTrainer:
        """Load the trainer.

        Returns:
            EncoderOnlyEmbedderTrainer: Loaded trainer instance.
        """
        # TODO data_collator和tokenizer
    
        trainer = EncoderOnlyEmbedderTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer
        )
        if self.data_args.same_dataset_within_batch:
            trainer.add_callback(EmbedderTrainerCallbackForDataRefresh(self.train_dataset))
        return trainer
    
    def load_dataset(self) -> AbsTextEmbedderTrainDataset:
        """Loads the training dataset based on data arguments.

        Returns:
            AbsEmbedderTrainDataset: The loaded dataset instance.
        """
        if self.data_args.same_dataset_within_batch:
            train_dataset = AbsEmbedderSameDatasetTrainDataset(
                args=self.data_args,
                default_batch_size=self.training_args.per_device_train_batch_size,
                seed=self.training_args.seed,
                tokenizer=self.tokenizer,
                process_index=self.training_args.process_index,
                num_processes=self.training_args.world_size
            )
            self.training_args.per_device_train_batch_size = 1
            self.training_args.dataloader_num_workers = 0   # avoid multi-processing
        else:
            train_dataset = AbsTextEmbedderTrainDataset(
                args=self.data_args,
                tokenizer=self.tokenizer
            )
        return train_dataset

    def load_data_collator(self) -> AbsTextEmbedderCollator:
        """Loads the appropriate data collator.

        Returns:
            AbsEmbedderCollator: Loaded data collator.
        """
        if self.data_args.same_dataset_within_batch:
            EmbedCollator = AbsEmbedderSameDatasetCollator
        else:
            EmbedCollator = AbsTextEmbedderCollator

        data_collator = EmbedCollator(
            tokenizer=self.tokenizer,
            query_max_len=self.data_args.query_max_len,
            passage_max_len=self.data_args.passage_max_len,
            sub_batch_size=self.training_args.sub_batch_size,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            padding=True,
            return_tensors="pt"
        )
        return data_collator