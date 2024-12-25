import json
from typing import Dict, Any
from typing import Dict, Tuple
import torch
from accelerate import Accelerator
from loguru import logger as loguru_logger
import importlib
from UniRetrieval.training.embedder.recommendation.arguments import TrainingArguments, DataAttr4Model

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
    try:
        module = importlib.import_module(f"UniRetrieval.modules.{model_type}s")
        cls = getattr(module, model_name)
        return cls
    except ImportError as e:
        raise ImportError(f"Could not import {model_name} from UniRetrieval.modules.{model_type}s") from e


def get_seq_data(d: dict):
    if "seq" in d:
        return d['seq']
    else:
        return {}


def split_batch(batch: dict, data_attr: DataAttr4Model) -> Tuple[Dict, Dict, Dict]:
    context_feat = {}; item_feat = {}
    seq_feat = get_seq_data(batch)
    for k, v in batch.items():
        if k in data_attr.context_features:
            context_feat[k] = v
        elif k in data_attr.item_features:
            item_feat[k] = v
    return context_feat, seq_feat, item_feat


def batch_to_device(batch, device) -> Dict:
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
        elif isinstance(value, dict):
            batch[key] = batch_to_device(value, device)
    return batch

def get_logger(config: TrainingArguments):
    accelerator = Accelerator()
    logger = loguru_logger
    if accelerator.is_local_main_process:
        if config.logging_dir is not None:
            logger.add(f"{config.logging_dir}/train.log", level='INFO')
        elif config.checkpoint_dir is not None:
            logger.add(f"{config.checkpoint_dir}/train.log", level='INFO')
    return logger