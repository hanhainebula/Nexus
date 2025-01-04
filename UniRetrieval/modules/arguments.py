import json
from typing import Dict, Any, List, Optional, Tuple
import torch
import importlib
from dataclasses import dataclass, field
from UniRetrieval.abc.arguments import AbsArguments

@dataclass
class Statistics:
    columns: List[str] = field(
        default_factory=list,
        metadata={
            'description': 'names of all the statistics'
        },
    )

    @staticmethod
    def from_dict(d: dict) -> "Statistics":
        stat = Statistics(d.pop("columns"))
        for k, v in d.items():
            setattr(stat, k, v)
            stat.columns.append(k)
        return stat


@dataclass
class DataAttr4Model:
    """
    Data attributes for a dataset. Serve the models, especially for the initialization.
    """
    fiid: str = field(
        metadata={
            'description': 'column name of the item ids'
        },
    )

    flabels: List[str] = field(
        metadata={
            'description': 'column names of the labels'
        },
    )

    features: List[str] = field(
        metadata={
            'description': 'column names of the features'
        },
    )

    context_features: List[str] = field(
        metadata={
            'description': 'column names of the context features, serving as the inputs x to the model'
        },
    )

    item_features: List[str] = field(
        metadata={
            'description': 'column names of the item features, such as item category, price, etc'
        },
    )

    seq_features: Dict[str, List[str]] = field(
        metadata={
            'description': 'column names of the sequential features, such as the click history of the users'
        },
    )

    seq_lengths: Dict[str, int] = field(
        metadata={
            'description': 'length of the sequences'
        },
    )

    num_items: int = field(
        metadata={
            'description': 'number of candidate items'
        },
    ) # number of candidate items instead of maximum id of items

    stats: Statistics = field(
        metadata={
            'description': 'statistics of the dataset, describing the number of unique values of each feature'
        },
    )

    @staticmethod
    def from_dict(d: dict):
        if "stats" in d:
            d["stats"] = Statistics.from_dict(d["stats"])
        attr = DataAttr4Model(**d)
        return attr

    def to_dict(self):
        d = self.__dict__
        for k, v in d.items():
            if type(v) == Statistics:
                d[k] = v.__dict__
        return d
    
    
def read_json(json_path: str) -> Dict[str, Any]:
    """Helper function to read a JSON file into a dictionary."""
    with open(json_path, 'r') as f:
        return json.load(f)
    
def get_modules(module_type: str, module_name: str):
    assert module_type in ["loss", "sampler", "encoder", "interaction", "score", "module"], f"{module_type} is not a valid module type"
    try:
        module = importlib.import_module(f"UniRetrieval.modules.{module_type}")
        cls = getattr(module, module_name)
        return cls
    except ImportError as e:
        raise ImportError(f"Could not import {module_name} from UniRetrieval.modules.{module_type}") from e
    

def get_model_cls(model_type: str, model_name: str):
    assert model_type in ["retriever", "ranker"], f"{model_type} is not a valid model type"
    model_type = 'embedder' if (model_type == "retriever") else "reranker"
    try:
        module = importlib.import_module(f"UniRetrieval.training.{model_type}.recommendation.modeling")
        cls = getattr(module, model_name)
        return cls
    except ImportError as e:
        raise ImportError(f"Could not import {model_name} from UniRetrieval.training.{model_type}.recommendation.modeling") from e

def get_seq_data(d: dict, seq_name: Optional[str]):
    """ Get sequence data from a batch.

    Args:
        d (Dict[str: Any]): A dictionary containing the batch of data.
        seq_names (Optional[str]): The names of the sequence to extract. If None, use the default sequence name 'seq'.

    Returns:
       Dict: A dictionary containing the sequence data. If no sequence data, return an empty dictionary.
    
    """
    if seq_name is not None:
        return d[seq_name]
    if "seq" in d:
        return d['seq']
    else:
        return {}


def split_batch(batch: dict, data_attr: DataAttr4Model) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict], Dict[str, torch.Tensor]]:
    context_feat = {}; item_feat = {}; seq_feat_dict = {}
    for k, v in batch.items():
        if k in data_attr.context_features:
            context_feat[k] = v
        elif k in data_attr.item_features:
            item_feat[k] = v
        elif k in data_attr.seq_features:
            seq_feat_dict[k] = get_seq_data(batch, k)
    return context_feat, item_feat, seq_feat_dict


def batch_to_device(batch, device) -> Dict:
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
        elif isinstance(value, dict):
            batch[key] = batch_to_device(value, device)
    return batch


def log_dict(logger, d: Dict):
    """Log a dictionary of values."""
    output_list = [f"{k}={v}" for k, v in d.items()]
    logger.info(", ".join(output_list))