import os
from typing import Dict, List, Union

import torch
import torch.distributed as dist
from UniRetrieval.modules.embedding import TDEMultiFeatEmbedding, MultiFeatEmbedding

from torchrec import JaggedTensor


__all__ = []


MEMORY_IO_REGISTERED = False


def register_memory_io():
    global MEMORY_IO_REGISTERED
    if not MEMORY_IO_REGISTERED:
        mem_io_path = os.getenv("TDE_MEMORY_IO_PATH")
        if mem_io_path is None:
            raise RuntimeError("env TDE_MEMORY_IO_PATH must set for unittest")

        torch.ops.tde.register_io(mem_io_path)
        MEMORY_IO_REGISTERED = True


def init_dist():
    if not dist.is_initialized():
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "13579"
        dist.init_process_group("nccl")


def convert_to_tde_model(model:torch.nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, MultiFeatEmbedding):
            model.set_submodule(
                name,
                TDEMultiFeatEmbedding(
                    multi_feat_embedding=model.get_submodule(name),
                    tde_table_configs=model.model_config.tde_settings['table_configs']
                )
            )
            
    return model

def convert_jt_to_tensor(data: dict):
    for k in data:
        if isinstance(data[k], JaggedTensor):
            data[k] = data[k].to_padded_dense()
            if data[k].shape[0] == 1:
                data[k] = data[k].squeeze(0)
        elif isinstance(data[k], dict):
            data[k] = convert_jt_to_tensor(data[k])
            
    return data