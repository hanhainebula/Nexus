from abc import ABC, abstractmethod


class AbsEvalDataLoader(ABC):
    @abstractmethod
    def __init__(
        self,
        *args,
        **kwargs
    ):
        pass
