from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict

from torch import nn

@dataclass
class AbsModelOutput:
    def to_dict(self):
        return asdict(self)


class AbsModel(ABC, nn.Module):

    @abstractmethod
    def compute_loss(self, *args, **kwargs):
        pass


class AbsEmbedder(ABC, nn.Module):
    
    @abstractmethod
    def compute_score(self, *args, **kwargs):
        pass

    @abstractmethod
    def encode_query(self, *args, **kwargs):
        pass

    @abstractmethod
    def encode_info(self, *args, **kwargs):
        pass


class AbsReranker(ABC, nn.Module):
    
    @abstractmethod
    def compute_score(self, *args, **kwargs):
        pass
