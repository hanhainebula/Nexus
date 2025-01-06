#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os 
import json 
import queue
import threading
from typing import Dict, List, Union, Optional
import itertools
import numpy as np 
from functools import partial

import torch
import torch.distributed as dist
from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL
from torchrec import EmbeddingBagConfig, EmbeddingConfig, EmbeddingCollection, EmbeddingBagCollection
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.planner import Topology, EmbeddingShardingPlanner
from torchrec.distributed.types import EmbeddingModuleShardingPlan
from torchrec.distributed.model_parallel import DistributedModelParallel, get_default_sharders

from torchrec.distributed.embedding_types import ShardedEmbeddingModule
from torchrec.datasets.utils import Batch
from torchrec import KeyedJaggedTensor, JaggedTensor
from torchrec.inference.state_dict_transform import (
    state_dict_gather,
    state_dict_to_device,
)
from torchrec.distributed.types import ModuleSharder, ShardingType, ShardingPlan, ShardingEnv


from torchrec_dynamic_embedding.id_transformer_group import IDTransformerGroup
from torchrec_dynamic_embedding.id_transformer_collection import IDTransformerCollection
from torchrec_dynamic_embedding.ps import PSCollection
from dynamic_embedding.utils import convert_jt_to_tensor

from UniRetrieval.training.reranker.recommendation.modeling import BaseRanker
from UniRetrieval.training.embedder.recommendation.modeling import BaseRetriever
from UniRetrieval.training.reranker.recommendation.arguments import ModelArguments, DataAttr4Model
from UniRetrieval.modules.arguments import get_model_cls, get_modules, split_batch
from UniRetrieval.modules.embedding import TDEMultiFeatEmbedding, MultiFeatEmbedding


# Similar to torch.utils.data._utils.pin_memory._pin_memory_loop
def transform_loop(dataloader, transform_fn, out_queue, done_event):
    # This setting is thread local, and prevents the copy in pin_memory from
    # consuming all CPU cores.
    torch.set_num_threads(1)

    for data in dataloader:
        if done_event.is_set():
            break
        transformed_data = transform_fn(data)

        while not done_event.is_set():
            try:
                out_queue.put(transformed_data, timeout=MP_STATUS_CHECK_INTERVAL)
                break
            except queue.Full:
                continue
        # save memory
        del transformed_data

    if not done_event.is_set():
        done_event.set()
        
def _get_data_device(data):
        device_dict = {}
        for feat in data:
            if type(data[feat]) == dict:
                device_dict[feat] = _get_data_device(data[feat])
            elif isinstance(data[feat], (torch.Tensor, JaggedTensor)):
                device_dict[feat] = data[feat].device
        return device_dict
    
def _to_device(data, device_or_device_dict: Union[torch.device, Dict]):
    for feat in data:
        if type(data[feat]) == dict:
            data[feat] = _to_device(data[feat], 
                device_or_device_dict[feat] if type(device_or_device_dict) == dict \
                else device_or_device_dict)
        elif isinstance(data[feat], (torch.Tensor, JaggedTensor)):
            data[feat] = data[feat].to(
                device_or_device_dict[feat] if type(device_or_device_dict) == dict \
                else device_or_device_dict)
    return data 

def _transform_fn(data:Dict, feat2path:Dict[str, str], id_transformer_group: IDTransformerGroup):
    """
    data: Dict
    """

    # combine tensors sharing the same feature name
    global_kjts, global_feat_lens, global_feat_original_key = {}, {}, {}
    for feature_name in feat2path:
        all_feat_tensors, all_feat_lens, all_feat_original_key = [], [], []
        for feat in data:
            if isinstance(data[feat], dict):
                for sub_feat in data[feat]:
                    if sub_feat == feature_name:
                        if data[feat][sub_feat].dim() > 1:
                            all_feat_tensors.extend(list(data[feat][sub_feat]))
                            all_feat_lens.append(data[feat][sub_feat].shape[0])
                        else:
                            all_feat_tensors.extend([data[feat][sub_feat]])
                            all_feat_lens.append(1)
                        all_feat_original_key.append([feat, sub_feat])
            else:
                if feat == feature_name:
                    if data[feat].dim() > 1:
                        all_feat_tensors.extend(list(data[feat]))
                        all_feat_lens.append(data[feat].shape[0])
                    else:
                        all_feat_tensors.extend([data[feat]])
                        all_feat_lens.append(1)
                    all_feat_original_key.append(feat)
        
        if len(all_feat_tensors) == 0:
            continue
        else:
            all_feat_tensors = JaggedTensor.from_dense(all_feat_tensors)
            global_kjts[feat2path[feature_name]] = KeyedJaggedTensor.from_jt_dict({feature_name : all_feat_tensors})
            global_feat_lens[feat2path[feature_name]] = all_feat_lens
            global_feat_original_key[feat2path[feature_name]] = all_feat_original_key

    # transform
    cache_kjts, fetch_handles = id_transformer_group.transform(global_kjts)
    
    # revert cache_kjts to original data
    for feature_name in feat2path:
        if feat2path[feature_name] in cache_kjts:
            all_feat_tensors = cache_kjts[feat2path[feature_name]][feature_name].to_dense()
            all_feat_lens = global_feat_lens[feat2path[feature_name]]
            cum_all_feat_lens = np.cumsum([0] + all_feat_lens).tolist()
            all_feat_original_key = global_feat_original_key[feat2path[feature_name]] 
            for idx, feat_original_key in enumerate(all_feat_original_key):
                if type(feat_original_key) == list:
                        data[feat_original_key[0]][feat_original_key[1]] = \
                            JaggedTensor.from_dense(all_feat_tensors[cum_all_feat_lens[idx] : cum_all_feat_lens[idx + 1]])
                else:
                    data[feat_original_key] = \
                        JaggedTensor.from_dense(all_feat_tensors[cum_all_feat_lens[idx] : cum_all_feat_lens[idx + 1]])

    return data, fetch_handles


class PrefetchDataLoaderIter:
    def __init__(self, dataloader, transform_fn, num_prefetch=0):
        self._data_queue = queue.Queue(maxsize=num_prefetch)
        self._done_event = threading.Event()
        self._transform_thread = threading.Thread(
            target=transform_loop,
            args=(dataloader, transform_fn, self._data_queue, self._done_event),
        )
        self._transform_thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        return self._get_data()

    def __del__(self):
        self._done_event.set()

    def _get_data(self):
        if self._done_event.is_set():
            raise StopIteration
        if not self._transform_thread.is_alive():
            raise RuntimeError("Transform thread exited unexpectedly")
        data, handles = self._data_queue.get()
        for handle in handles:
            handle.wait()
        return data


class DataLoaderIter:
    def __init__(self, dataloader, transform_fn):
        self.transform_fn = transform_fn
        self._dataloader_iter = iter(dataloader)     

    def __next__(self):
        data = next(self._dataloader_iter)
        data_device_dict = _get_data_device(data)
        data = _to_device(data, torch.device('cpu'))
        data, handles = self.transform_fn(data)
        for handle in handles:
            handle.wait()
        data = _to_device(data, data_device_dict)
        data = convert_jt_to_tensor(data)
        
        return data


class DataLoader:
    def __init__(
        self,
        id_transformer_group: IDTransformerGroup,
        dataloader,
        *,
        configs_dict: Dict = None,
        paths: List[str] = None,
        num_prefetch=0,
    ):
        self._id_transformer_group = id_transformer_group
        
        self._configs_dict = configs_dict
        self._paths = paths
        all_feature_names = []
        for configs in self._configs_dict.values():
            for emb_config in configs:
                all_feature_names.extend(emb_config.feature_names)
        all_feature_names = list(set(all_feature_names))   
        self._feat2path = {} 
        for feature_name in all_feature_names:
            for path, configs in self._configs_dict.items():
                for emb_config in configs:
                    if feature_name in emb_config.feature_names:
                        self._feat2path[feature_name] = path
                        break
        
        self._data_queue = queue.Queue(maxsize=num_prefetch)
        self._done_event = threading.Event()

        self._dataloader = dataloader
        self._num_prefetch = num_prefetch
        
        # attribute of dataloader
        self.dataset = dataloader.dataset
    
    def __iter__(self):
        if self._num_prefetch > 0:
            return PrefetchDataLoaderIter(
                self._dataloader, 
                partial(_transform_fn, feat2path=self._feat2path, id_transformer_group=self._id_transformer_group),
                num_prefetch=self._num_prefetch
            )
        else:
            return DataLoaderIter(self._dataloader, 
                                  partial(_transform_fn, feat2path=self._feat2path, id_transformer_group=self._id_transformer_group))

    def __len__(self):
        return len(self._dataloader)
    
    
class Dataset:
    def __init__(
        self,
        id_transformer_group: IDTransformerGroup,
        dataset,
        configs_dict: Dict = None,
        paths: List[str] = None,
    ):
        self._dataset = dataset 
        self._id_transformer_group = id_transformer_group
        
        self._configs_dict = configs_dict
        self._paths = paths
        all_feature_names = []
        for configs in self._configs_dict.values():
            for emb_config in configs:
                all_feature_names.extend(emb_config.feature_names)
        all_feature_names = list(set(all_feature_names))   
        self._feat2path = {} 
        for feature_name in all_feature_names:
            for path, configs in self._configs_dict.items():
                for emb_config in configs:
                    if feature_name in emb_config.feature_names:
                        self._feat2path[feature_name] = path
                        break 
        
    def __getitem__(self, index:Union[int, torch.Tensor]):
        data = self._dataset[index]
        if isinstance(index, torch.Tensor):
            data_device_dict = _get_data_device(data)
            data = _to_device(data, torch.device('cpu'))
            data, handles = _transform_fn(data, self._feat2path, self._id_transformer_group)
            for handle in handles:
                handle.wait()
            data = _to_device(data, data_device_dict)
            data = convert_jt_to_tensor(data)
        return data

    def __len__(self):
        return len(self._dataset)
class BatchWrapper:
    
    def __init__(
        self,
        id_transformer_group: IDTransformerGroup,
        configs_dict: Dict = None,
        paths: List[str] = None,
    ): 
        self._id_transformer_group = id_transformer_group
        
        self._configs_dict = configs_dict
        self._paths = paths
        all_feature_names = []
        for configs in self._configs_dict.values():
            for emb_config in configs:
                all_feature_names.extend(emb_config.feature_names)
        all_feature_names = list(set(all_feature_names))   
        self._feat2path = {} 
        for feature_name in all_feature_names:
            for path, configs in self._configs_dict.items():
                for emb_config in configs:
                    if feature_name in emb_config.feature_names:
                        self._feat2path[feature_name] = path
                        break
                    
    def __call__(self, data:dict):
        data_device_dict = _get_data_device(data)
        data = _to_device(data, torch.device('cpu'))
        data, handles = _transform_fn(data, self._feat2path, self._id_transformer_group)
        for handle in handles:
            handle.wait()
        data = _to_device(data, data_device_dict)
        data = convert_jt_to_tensor(data)
        return data

class TDEModel(torch.nn.Module):
    
    def __init__(self, base_model:Union[BaseRanker, BaseRetriever]):
        super().__init__()
        self.base_model = base_model 
        self.base_model = self.convert_to_tde_model(self.base_model)
        
    
    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.base_model, name)
    
    def convert_to_tde_model(self, model:Union[BaseRanker, BaseRetriever]):
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
    
    def _prepare_model(self, model):
        
        def _get_tde_configs_dict(model) -> Dict:
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
        
        def _get_tde_feature_names(model) -> List[str]:
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
        self.tde_configs_dict = _get_tde_configs_dict(model)
        self.tde_feature_names = _get_tde_feature_names(model)
        self.tde_settings = model.model_config.tde_settings
        
        rank, world_size, device, backend, pg = self._init_process()
        topology = Topology(
            world_size=world_size,
            local_world_size=world_size,
            compute_device=device.type,
        )
        
        sharding_type = self.tde_settings["sharding_type"]
        planner = EmbeddingShardingPlanner(
            topology=topology,
            constraints={
                f"{feature_name}": ParameterConstraints(
                    sharding_types=getattr(ShardingType, sharding_type.upper()).value)
                for feature_name in self.tde_feature_names
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
            self.tde_settings["ps_url"],
            model, 
            self.tde_configs_dict)
        
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
            model_class_or_name = config_dict['model_name_or_path']
        if isinstance(model_class_or_name, str):
            model_cls = get_model_cls(config_dict['model_type'], model_class_or_name)
        else:
            model_cls = model_class_or_name
        del config_dict['model_type'], config_dict['model_name_or_path']
        
        # create model         
        model = model_cls(data_config, model_config)

        # wrap TDEModel 
        model = TDEModel(model)
        model = model._prepare_model(model)
        
        # load state_dict
        ckpt_path = os.path.join(checkpoint_dir, "model.pt")
        state_dict = torch.load(ckpt_path, weights_only=False, map_location='cpu')

        model.load_state_dict(state_dict)
        
        if "item_vectors" in state_dict:
            model.item_vectors = state_dict["item_vectors"]
            del state_dict['item_vectors']
        
        return model
        

def attach_id_transformer_group(
    url: str,
    module: DistributedModelParallel,
    configs_dict: Dict[str, Union[List[EmbeddingBagConfig], List[EmbeddingConfig]]],
    *,
    eviction_config=None,
    transform_config=None,
    ps_config=None,
    parallel=True
):
    id_transformer_group = IDTransformerGroup(
        url,
        module,
        configs_dict,
        eviction_config=eviction_config,
        transform_config=transform_config,
        ps_config=ps_config,
        parallel=parallel,
    )
    # Attach the id transformer group to module for saving.
    module._id_transformer_group = id_transformer_group

def wrap_dataloader(
    dataloader,
    module: DistributedModelParallel,
    configs_dict: Dict[str, Union[List[EmbeddingBagConfig], List[EmbeddingConfig]]],
    *,
    num_prefetch=0,
):
    """
    DataLoader to transform data from global id to cache id.

    Args:
        url: configuration for PS, e.g. redis://127.0.0.1:6379/?prefix=model.
        dataloader: dataloader to transform.
        module: DMP module that need dynamic embedding.
        configs_dict: a dictionary that maps the module path of the sharded module to its embedding
            configs or embeddingbag configs. The plan of `module` should contain the module path
            in `configs_dict`.
        eviction_config: configuration for eviction policy. Default is `{"type": "mixed_lru_lfu"}`
        transform_config: configuration for the transformer. Default is `{"type": "naive"}`
        transform_config: configuration for the ps. Default is `{"chunk_size": 8 * 1024 * 1024}
        parallel: Whether the IDTransformerCollections will run paralell. When set to True,
            IDTransformerGroup will start a thread for each IDTransformerCollection.
        num_prefetch: number of samples to prefetch.

    Return:
        DataLoader: the dataloader to transform data.
        DistributedModelParallel: model with id_transformer_group attached.

    Example:
        class Model(nn.Module):
            def __init__(self, config1, config2):
                super().__init__()
                self.emb1 = EmbeddingCollection(tables=config1, device=torch.device("meta"))
                self.emb2 = EmbeddingCollection(tables=config2, device=torch.device("meta"))
                ...

            def forward(self, kjt1, kjt2):
                ...

        m = Model(config1, config2)
        m = DistributedModelParallel(m)
        dataloader, m = tde.wrap("redis://127.0.0.1:6379/", dataloader, m, { "emb1": config1, "emb2": config2 })

        for label, kjt1, kjt2 in dataloader:
            output = m(kjt1, kjt2)
            ...
    """
    paths = list(configs_dict.keys())
    return DataLoader(
            id_transformer_group=module._id_transformer_group,
            dataloader=dataloader,
            configs_dict=configs_dict,
            paths=paths,
            num_prefetch=num_prefetch,
        )
    

def wrap_dataset(
    dataset,
    module: DistributedModelParallel,
    configs_dict: Dict[str, Union[List[EmbeddingBagConfig], List[EmbeddingConfig]]],
):  
    paths = list(configs_dict.keys())
    return Dataset(
        module._id_transformer_group,
        dataset,
        configs_dict,
        paths, 
    )