from abc import ABC, abstractmethod

from .data_loader import AbsEvalDataLoader


class AbsEvaluator(ABC):
    def __init__(
        self,
        eval_name: str,
        data_loader: AbsEvalDataLoader,
        overwrite: bool = False
    ):
        self.eval_name = eval_name
        self.data_loader = data_loader
        self.overwrite = overwrite

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def compute_metrics(*args, **kwargs):
        pass
