import json
from typing import Dict, Any, Tuple
import torch
import importlib
from dataclasses import dataclass, field
from UniRetrieval.abc.arguments import AbsArguments


@dataclass
class Statistics(AbsArguments):
    request_id: int
    user_id: int
    device_id: int
    age: int
    gender: int
    province: int
    video_id: int
    author_id: int
    category_level_one: int
    category_level_two: int
    upload_type: int

@dataclass
class DataAttr4Model(AbsArguments):
    """
    Data attributes for a dataset. Serve for models
    """
    fiid: str
    num_items: int  # number of candidate items instead of maximum id of items
    stats: Statistics
    flabels: str = field(default=None, metadata={"nargs": "+"})
    features: str = field(default=None, metadata={"nargs": "+"})
    context_features: str = field(default=None, metadata={"nargs": "+"})
    item_features: str = field(default=None, metadata={"nargs": "+"})
    seq_features: str = field(default=None, metadata={"nargs": "+"})
    
    
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

def get_seq_data(d: dict):
    if "seq" in d:
        return d['seq']
    else:
        return {}


def split_batch(batch: dict, data_config: DataAttr4Model) -> Tuple[Dict, Dict, Dict]:
    context_feat = {}; item_feat = {}
    seq_feat = get_seq_data(batch)
    for k, v in batch.items():
        if k in data_config.context_features:
            context_feat[k] = v
        elif k in data_config.item_features:
            item_feat[k] = v
    return context_feat, seq_feat, item_feat


def batch_to_device(batch, device) -> Dict:
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
        elif isinstance(value, dict):
            batch[key] = batch_to_device(value, device)
    return batch
