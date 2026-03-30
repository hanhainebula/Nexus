import logging
import os
import inspect
from typing import Optional

import torch
from transformers import set_seed

from Nexus.abc.training.embedder import (
    AbsEmbedderModel,
    AbsEmbedderRunner,
    AbsEmbedderTrainDataset,
    AbsEmbedderTrainer,
)

from .arguments import (
    MultimodalEmbedderDataArguments,
    MultimodalEmbedderModelArguments,
    MultimodalEmbedderTrainingArguments,
    WrappedMultimodalEmbedderModelArguments,
)
from .dataset import AbsMultimodalEmbedderCollator, AbsMultimodalEmbedderTrainDataset
from .modeling import BiMultimodalEmbedderModel
from .trainer import MultimodalEmbedderTrainer

logger = logging.getLogger(__name__)


class MultimodalEmbedderRunner(AbsEmbedderRunner):
    def __init__(
        self,
        model_args: MultimodalEmbedderModelArguments,
        data_args: MultimodalEmbedderDataArguments,
        training_args: MultimodalEmbedderTrainingArguments,
        model: Optional[AbsEmbedderModel] = None,
        train_dataset: Optional[AbsEmbedderTrainDataset] = None,
        trainer: Optional[AbsEmbedderTrainer] = None,
        loss_function: Optional[torch.nn.Module] = None,
        score_function: Optional[torch.nn.Module] = None,
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.wrapped_model_args = WrappedMultimodalEmbedderModelArguments(
            negatives_cross_device=self.training_args.negatives_cross_device,
            temperature=self.training_args.temperature,
            sub_batch_size=self.training_args.sub_batch_size,
            kd_loss_type=self.training_args.kd_loss_type,
            sentence_pooling_method=self.training_args.sentence_pooling_method,
            normalize_embeddings=self.training_args.normalize_embeddings,
            query_max_len=self.data_args.query_max_len,
            passage_max_len=self.data_args.passage_max_len,
            model_type=self.model_args.model_type,
            use_chat_template=self.model_args.use_chat_template,
            processor_call_kwargs=self.model_args.processor_call_kwargs,
        )

        overwrite_output_dir = getattr(training_args, "overwrite_output_dir", True)
        if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not overwrite_output_dir
        ):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )

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
        logger.info("Training parameters %s", training_args)
        logger.info("Model parameters %s", model_args)
        logger.info("Data parameters %s", data_args)

        set_seed(training_args.seed)

        self.loss_function = loss_function
        self.score_function = score_function
        self.model = model if model is not None else self.load_model()
        self.processor = self.model.processor
        self.train_dataset = train_dataset if train_dataset is not None else self.load_dataset()
        self.data_collator = self.load_data_collator()
        self.trainer = trainer if trainer is not None else self.load_trainer()

    def load_model(self) -> BiMultimodalEmbedderModel:
        model = BiMultimodalEmbedderModel.build(
            model_args=self.model_args,
            wrapped_model_args=self.wrapped_model_args,
            loss_function=self.loss_function,
            score_function=self.score_function,
        )

        if self.training_args.gradient_checkpointing:
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable()

        return model

    def load_trainer(self) -> MultimodalEmbedderTrainer:
        trainer_kwargs = {
            "model": self.model,
            "args": self.training_args,
            "train_dataset": self.train_dataset,
            "data_collator": self.data_collator,
        }
        trainer_init_params = inspect.signature(MultimodalEmbedderTrainer.__init__).parameters
        if "processing_class" in trainer_init_params:
            trainer_kwargs["processing_class"] = self.processor
        elif "tokenizer" in trainer_init_params:
            trainer_kwargs["tokenizer"] = getattr(self.processor, "tokenizer", None)
        return MultimodalEmbedderTrainer(**trainer_kwargs)

    def load_dataset(self) -> AbsMultimodalEmbedderTrainDataset:
        return AbsMultimodalEmbedderTrainDataset(
            args=self.data_args,
            tokenizer=getattr(self.processor, "tokenizer", None),
        )

    def load_data_collator(self) -> AbsMultimodalEmbedderCollator:
        return AbsMultimodalEmbedderCollator(
            tokenizer=getattr(self.processor, "tokenizer", None),
            padding=True,
            return_tensors="pt",
        )
