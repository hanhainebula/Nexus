from operator import itemgetter
from typing import Dict, Tuple

import torch
from rs4industry.data.dataset import DataAttr4Model

import importlib

from accelerate import Accelerator
from loguru import logger as loguru_logger
from rs4industry.config import TrainingArguments


def get_logger(config: TrainingArguments):
    accelerator = Accelerator()
    logger = loguru_logger
    if accelerator.is_local_main_process:
        if config.logging_dir is not None:
            logger.add(f"{config.logging_dir}/train.log", level='INFO')
        elif config.checkpoint_dir is not None:
            logger.add(f"{config.checkpoint_dir}/train.log", level='INFO')
    return logger

def get_modules(module_type: str, module_name: str):
    assert module_type in ["loss", "sampler", "encoder", "interaction", "score", "module"], f"{module_type} is not a valid module type"
    # import the module {module_name} from "rs4industry.model.{module_type}"
    try:
        # from "rs4industry.model.{module_type}" import {module_name}
        module = importlib.import_module(f"rs4industry.model.{module_type}")
        cls = getattr(module, module_name)
        # module = importlib.import_module(module_name, package=pkg)
        return cls
    except ImportError as e:
        raise ImportError(f"Could not import {module_name} from rs4industry.model.{module_type}") from e
    

def get_model_cls(model_type: str, model_name: str):
    assert model_type in ["retriever", "ranker"], f"{model_type} is not a valid model type"
    try:
        module = importlib.import_module(f"rs4industry.model.{model_type}s")
        cls = getattr(module, model_name)
        return cls
    except ImportError as e:
        raise ImportError(f"Could not import {model_name} from rs4industry.model.{model_type}s") from e


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
