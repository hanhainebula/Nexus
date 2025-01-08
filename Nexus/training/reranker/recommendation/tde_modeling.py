from dataclasses import dataclass
import os
import json
from typing import Dict, Union, Tuple, Optional, List
import itertools

import torch
import torch.distributed as dist


from .modeling import BaseRanker
from Nexus.training.reranker.recommendation.arguments import DataAttr4Model, ModelArguments
from Nexus.modules.arguments import get_model_cls

from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL
from torchrec import EmbeddingBagConfig, EmbeddingConfig, EmbeddingCollection, EmbeddingBagCollection
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.planner import Topology, EmbeddingShardingPlanner
from torchrec.distributed.types import EmbeddingModuleShardingPlan
from torchrec.distributed.model_parallel import DistributedModelParallel, get_default_sharders

from torchrec.distributed.embedding_types import ShardedEmbeddingModule
from torchrec.inference.state_dict_transform import (
    state_dict_gather,
    state_dict_to_device,
)
from torchrec.distributed.types import ModuleSharder, ShardingType, ShardingPlan, ShardingEnv

from dynamic_embedding.utils import convert_to_tde_model
from dynamic_embedding.wrappers import attach_id_transformer_group

    
class TDEModel(torch.nn.Module):
    
    def __init__(self, base_model:BaseRanker):
        super().__init__()
        self.base_model = base_model 
        self.base_model = convert_to_tde_model(self.base_model)
        
    # use attribute or method from base_model if it is not found in TDEModel
    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.base_model, name)
    
    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)
    
    @staticmethod
    def _init_process(backend=None):
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        if backend is None:
            if torch.cuda.is_available():
                device: torch.device = torch.device(f"cuda:{rank}")
                backend = "nccl"
                torch.cuda.set_device(device)
            else:
                device: torch.device = torch.device("cpu")
                backend = "gloo"
        elif backend == 'nccl':
            device: torch.device = torch.device(f"cuda:{rank}")
            torch.cuda.set_device(device)
        elif backend == 'gloo':
            device: torch.device = torch.device("cpu")
            
        if not dist.is_initialized():
            dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
            pg = dist.group.WORLD
        else:
            pg = dist.new_group(backend=backend)

        return rank, world_size, device, backend, pg
    
    @staticmethod
    def _prepare_model(model:BaseRanker):
        
        def _get_tde_configs_dict(model:BaseRanker) -> Dict:
            """Get the embedding configurations from the model.

            Returns:
                Dict[str, List[EmbeddingConfig]]: a dictionary of embedding configurations.
            Example:
                {"emb": [EmbeddingConfig, EmbeddingConfig], "emb2": [EmbeddingConfig, EmbeddingConfig]}
            """
            embedding_configs_dict = {}
            for name, module in model.named_modules():
                if isinstance(module, (EmbeddingCollection, EmbeddingBagCollection)):
                    embedding_configs_dict[name] = module._embedding_configs
            return embedding_configs_dict
        
        def _get_tde_feature_names(model:BaseRanker) -> List[str]:
            """Get the feature names from the model.
            
            Returns:
                List[str]: a list of feature names.

            Example:
                ["userId", "movieId", "rating", "timestamp"]
            """
            feature_names = []
            for _, module in model.named_modules():
                if isinstance(module, (EmbeddingCollection, EmbeddingBagCollection)):
                    flatten_feature_names = list(itertools.chain(*[config.feature_names for config in module._embedding_configs]))
                    feature_names.extend(flatten_feature_names)
            return feature_names
        
        # get tde configs 
        # get configs using TDEModel instead of base_model to make sure 'base_model' is included in path
        model.tde_configs_dict = _get_tde_configs_dict(model)
        model.tde_feature_names = _get_tde_feature_names(model)
        model.tde_settings = model.model_config.tde_settings
        
        rank, world_size, device, backend, pg = TDEModel._init_process()
        topology = Topology(
            world_size=world_size,
            local_world_size=world_size,
            compute_device=device.type,
        )
        
        sharding_type = model.tde_settings["sharding_type"]
        planner = EmbeddingShardingPlanner(
            topology=topology,
            constraints={
                f"{feature_name}": ParameterConstraints(
                    sharding_types=getattr(ShardingType, sharding_type.upper()).value)
                for feature_name in model.tde_feature_names
            }
        )
        plan: ShardingPlan = planner.collective_plan(model, get_default_sharders(), pg)
        
        model = DistributedModelParallel(
            module=model,
            env=ShardingEnv.from_process_group(pg),
            device=device,
            plan=plan,
            sharders=get_default_sharders(),
        )
        attach_id_transformer_group(
            model.module.tde_settings["ps_url"],
            model, 
            model.module.tde_configs_dict)
        
        return model
    
    # save -> shard -> load  
    def save(self, checkpoint_dir: str, **kwargs):
        self.save_checkpoint(checkpoint_dir)
        self.base_model.save_configurations(checkpoint_dir)
    
    def save_checkpoint(self, checkpoint_dir: str):
        path = os.path.join(checkpoint_dir, "model.pt")
        torch.save(self.state_dict(), path)
        
    @staticmethod
    def from_pretrained(checkpoint_dir:str, 
                        model_class_or_name:Optional[Union[type, str]]=None):
        
        # create raw model 
        config_path = os.path.join(checkpoint_dir, "model_config.json")
        with open(config_path, "r", encoding="utf-8") as config_path:
            config_dict = json.load(config_path)
            
        # data config 
        data_config = DataAttr4Model.from_dict(config_dict['data_config'])
        del config_dict['data_config']
        
        # model config 
        model_config = ModelArguments.from_dict(config_dict)
        
        # model class 
        if model_class_or_name is None:
            model_class_or_name = config_dict['model_name']
        if isinstance(model_class_or_name, str):
            model_cls = get_model_cls(config_dict['model_type'], model_class_or_name)
        else:
            model_cls = model_class_or_name
        del config_dict['model_type'], config_dict['model_name']
        
        # create model         
        model = model_cls(data_config, model_config)

        # wrap TDEModel 
        model = TDEModel(model)
        model = TDEModel._prepare_model(model)
        
        # load state_dict
        ckpt_path = os.path.join(checkpoint_dir, "model.pt")
        state_dict = torch.load(ckpt_path, weights_only=False, map_location='cpu')

        model.load_state_dict(state_dict)
        
        if "item_vectors" in state_dict:
            model.item_vectors = state_dict["item_vectors"]
            del state_dict['item_vectors']
        
        return model
