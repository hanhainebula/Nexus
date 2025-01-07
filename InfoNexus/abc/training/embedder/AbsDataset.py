import logging
from dataclasses import dataclass
from abc import abstractmethod


from InfoNexus.abc.training.dataset import AbsDataset

logger = logging.getLogger(__name__)


class AbsEmbedderTrainDataset(AbsDataset):
    """Abstract class for training dataset.

    Args:
        args (AbsEmbedderDataArguments): Data arguments.
    """
    
    pass

@dataclass
class AbsEmbedderCollator():
    """
    The abstract embedder collator.
    """
    @abstractmethod
    def __call__(self, features):
        return super().__call__(features)

@dataclass
class CallbackOutput:
    save_checkpoint: str = None
    stop_training: bool = False

        
