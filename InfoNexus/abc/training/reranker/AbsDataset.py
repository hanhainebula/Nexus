import logging
from dataclasses import dataclass
from InfoNexus.abc.training.dataset import AbsDataset

logger = logging.getLogger(__name__)


class AbsRerankerTrainDataset(AbsDataset):
    """Abstract class for reranker training dataset.

    Args:
        args (AbsRerankerDataArguments): Data arguments.
        tokenizer (PreTrainedTokenizer): Tokenizer to use.
    """
    pass

@dataclass
class AbsRerankerCollator():
    """
    The abstract reranker collator.
    """
    def __call__(self, features):
        return super().__call__(features)

# remove Abs LLM reranker class