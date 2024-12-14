from abc import ABC, abstractmethod
from typing import Any

from .arguments import AbsEvalArguments, AbsEvalDataLoaderArguments
from .data_loader import AbsEvalDataLoader
from .evaluator import AbsEvaluator


class AbsEvalRunner(ABC):
    def __init__(
        self,
        eval_args: AbsEvalArguments,
        eval_data_loader_args: AbsEvalDataLoaderArguments
    ):
        self.eval_args = eval_args
        self.data_loader = self.load_data_loader(eval_data_loader_args)
        self.evaluator = self.load_evaluator()

    def load_data_loader(
        self,
        eval_data_loader_args: AbsEvalDataLoaderArguments
    ) -> AbsEvalDataLoader:
        return AbsEvalDataLoader(
            **eval_data_loader_args.to_dict()
        )

    def load_evaluator(self):
        return AbsEvaluator(
            eval_name=self.eval_args.eval_name,
            data_loader=self.data_loader,
            overwrite=self.eval_args.overwrite
        )

    @abstractmethod
    def run(
        self,
        model: Any,
        *args,
        **kwargs
    ):
        pass
