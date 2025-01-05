from abc import ABC, abstractmethod

from .dataset import AbsDataset
from .modeling import AbsModel
from .trainer import AbsTrainer
from .arguments import AbsTrainingArguments, AbsDataArguments, AbsModelArguments


class AbsRunner(ABC):
    def __init__(
        self,
        model_args: AbsModelArguments,
        data_args: AbsDataArguments,
        training_args: AbsTrainingArguments,
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

    @abstractmethod
    def load_model(self, *args, **kwargs) -> AbsModel:
        pass

    @abstractmethod
    def load_dataset(self, *args, **kwargs) -> AbsDataset:
        pass

    @abstractmethod
    def load_trainer(self, *args, **kwargs) -> AbsTrainer:
        pass

    @abstractmethod
    def run(self, *args, **kwargs):
        pass
