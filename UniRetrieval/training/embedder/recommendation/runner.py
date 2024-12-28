import logging
from typing import Tuple
from UniRetrieval.abc.training.embedder import AbsEmbedderRunner
from .arguments import TrainingArguments, ModelArguments, DataArguments
from .modeling import MLPRetriever
from .trainer import RetrieverTrainer
from .datasets import AbsRecommenderEmbedderCollator, ConfigProcessor, DailyDataset, DailyDataIterator, DataAttr4Model
from UniRetrieval.modules.optimizer import get_lr_scheduler, get_optimizer
from UniRetrieval.training.embedder.recommendation.callback import ItemVectorCallback


class RetrieverRunner(AbsEmbedderRunner):
    """
    Finetune Runner for base embedding models.
    """
    def __init__(
        self,
        model_config_path: str,
        data_config_path: str,
        train_config_path: str
    ):        
        self.model_config_path = model_config_path
        self.data_config_path = data_config_path
        self.train_config_path = train_config_path
        
        self.data_args = DataArguments.from_json(self.data_config_path)
        self.model_args = ModelArguments.from_json(self.model_config_path)
        self.training_args = TrainingArguments.from_json(self.train_config_path)
        
        self.train_dataset, self.cp_attr = self.load_dataset()
        self.model = self.load_model()
        self.data_collator = self.load_data_collator()
        self.trainer = self.load_trainer()

    def load_dataset(self) -> Tuple[DailyDataset, DataAttr4Model]:
        cp = ConfigProcessor(self.data_config_path)
        train_data_iterator = DailyDataIterator(cp.config, cp.train_files)
        train_data = DailyDataset(train_data_iterator, shuffle=True, attrs=cp.attrs, preload=False)
        return train_data, cp.attrs
    
    def load_model(self) -> MLPRetriever:
        item_loader = self.train_dataset.get_item_loader(self.data_args.item_batch_size)
        model = MLPRetriever(self.cp_attr, self.model_config_path, item_loader=item_loader)
        return model

    def load_trainer(self) -> RetrieverTrainer:    
        
        self.optimizer = get_optimizer(
            self.training_args.optim,
            self.model.parameters(),
            lr=self.training_args.learning_rate,
            weight_decay=self.training_args.weight_decay    
        )
        self.lr_scheduler = get_lr_scheduler()
        # self.training_args.dataloader_num_workers = 0   # avoid multi-processing

        trainer = RetrieverTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            data_collator=self.data_collator,
            optimizers=[self.optimizer, self.lr_scheduler]
        )
        trainer.add_callback(ItemVectorCallback(trainer=trainer))
        # if self.data_args.same_dataset_within_batch:
        #     trainer.add_callback(EmbedderTrainerCallbackForDataRefresh(self.train_dataset))
        return trainer

    def load_data_collator(self) -> AbsRecommenderEmbedderCollator:
        collator = AbsRecommenderEmbedderCollator()
        return collator