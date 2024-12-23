from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from .arguments import AbsEvalArguments


class AbsEvalDataLoader(ABC):
    @abstractmethod
    def __init__(
        self,
        *args,
        **kwargs
    ):
        pass
