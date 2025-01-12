import torch
import numpy as np

import os
from tqdm import tqdm, trange
from typing import cast, Any, List, Union, Optional, Tuple, Type
import onnx
import onnxruntime as ort
import tensorrt as trt
import pandas as pd
import torch
import numpy as np
from dataclasses import dataclass, field
from Nexus.abc.inference import AbsReranker, InferenceEngine, AbsInferenceArguments
import pycuda.driver as cuda
import os
import sys
sys.path.append('.')
import time

from pandas import DataFrame
import yaml
import torch
import numpy as np

import tensorrt as trt
import pycuda.driver as cuda

# from inference.inference.inference_engine import InferenceEngine
from Nexus.training.reranker.recommendation.modeling import BaseRanker

import redis
import os
import time
import re
import importlib 
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy 
from pandas import DataFrame

import torch
import numpy as np
import pandas as pd
import onnxruntime as ort
import yaml
import json 
import redis 

import onnx
import onnxruntime as ort

import tensorrt as trt
import pycuda.driver as cuda
from .base import BaseRerankerInferenceEngine

from dynamic_embedding.wrappers import BatchWrapper
from torchrec.distributed import DistributedModelParallel

from Nexus.training.reranker.recommendation.tde_modeling import TDEModel

class TDERerankerInferenceEngine(BaseRerankerInferenceEngine):

    def __init__(self, config: dict) -> None:
        # super().__init__(config)

        # load config
        self.config = config
        with open(os.path.join(config['model_ckpt_path'], 'model_config.json'), 'r', encoding='utf-8') as f:
            self.model_ckpt_config = json.load(f)
        self.feature_config = deepcopy(self.model_ckpt_config['data_config'])
        with open(config['feature_cache_config_path'], 'r') as f:
            self.feature_cache_config = yaml.safe_load(f)

        # load cache protos 
        self.key_temp2proto = {}
        for key_temp, proto_dict in self.feature_cache_config['key_temp2proto'].items():
            proto_spec = importlib.util.spec_from_file_location(
                f'{key_temp}_proto_module', proto_dict['module_path'])
            proto_module = importlib.util.module_from_spec(proto_spec)
            proto_spec.loader.exec_module(proto_module)
            self.key_temp2proto[key_temp] = getattr(proto_module, proto_dict['class_name'])

        # connect to redis for feature cache
        self.redis_client = redis.Redis(host=self.feature_cache_config['host'], 
                                        port=self.feature_cache_config['port'], 
                                        db=self.feature_cache_config['db'])    
            
        # put seq into context_features
        # self.feature_config is deepcopy of self.model_ckpt_config['data_config']
        if 'seq_features' in self.model_ckpt_config['data_config']:
            self.feature_config['context_features'].append(self.model_ckpt_config['data_config']['seq_features'])
            
        # load model
        self.model:DistributedModelParallel = self.load_model()
        self.model.module.model_config.topk = self.config['output_topk']
        
        # get batch wrapper 
        self.batch_wrapper = BatchWrapper(self.model._id_transformer_group,
                                          self.model.module.tde_configs_dict,
                                          list(self.model.module.tde_configs_dict.keys()))
        
    def _batch_to_tensor(self, batch_data:dict):
        '''
        move batch data to device
        Args:
            batch_data: dict
        Returns:
            batch_data: dict
        '''
        for key, value in batch_data.items():
            if isinstance(value, dict):
                batch_data[key] = self._batch_to_tensor(value)
            else:
                batch_data[key] = torch.tensor(value)
        return batch_data
    
    def _batch_to_device(self, batch_data:dict, device):
        '''
        move batch data to device
        Args:
            batch_data: dict
        Returns:
            batch_data: dict
        '''
        for key, value in batch_data.items():
            if isinstance(value, dict):
                batch_data[key] = self._batch_to_device(value, device)
            else:
                batch_data[key] = value.to(device)
        return batch_data
        
            
    def batch_inference(self, batch_infer_df:pd.DataFrame, batch_candidates_df:pd.DataFrame):
        '''
        batch inference
        Args:
            batch_infer_df: pd.DataFrame: batch of infer request.
            batch_candidates_df: pd.DataFrame: candidates of the batch request.
        Returns:
            batch_outputs: np.ndarray
        '''
        # batch_st_time = time.time()

        # get user_context features 
        batch_user_context_dict = self.get_user_context_features(batch_infer_df)
        
        # get candidates features
        batch_candidates_dict = self.get_candidates_features(batch_candidates_df)
        # TODO: Cross Features

        feed_dict = {}
        feed_dict.update(batch_user_context_dict)
        feed_dict.update(batch_candidates_dict)
        feed_dict = self._batch_to_tensor(feed_dict)
        feed_dict = self.batch_wrapper(feed_dict)
        
        # split candidates and user_context features
        batch_user_context_dict, batch_candidates_dict = {}, {}
        for key, value in feed_dict.items():
            if key.startswith('candidates_'):
                batch_candidates_dict[key[len('candidates_') : ]] = value
            else:
                batch_user_context_dict[key] = value
        batch_user_context_dict = self._batch_to_device(batch_user_context_dict, self.model.device)
        batch_candidates_dict = self._batch_to_device(batch_candidates_dict, self.model.device)
        
        batch_outputs_idx = self.model.module.predict(
            batch_user_context_dict, batch_candidates_dict, 
            topk=self.config['output_topk'], gpu_mem_save=True) # topk idx
        # batch_outputs = batch_outputs_idx
        batch_outputs = []
        for row_idx, output_idx in enumerate(batch_outputs_idx):
            batch_outputs.append(
                np.array(batch_candidates_df.iloc[row_idx][self.feature_config['fiid']])[output_idx.cpu()]) # get topk item id
        batch_outputs = np.stack(batch_outputs, axis=0)
        batch_ed_time = time.time()
        # print(f'batch time: {batch_ed_time - batch_st_time}s')
        return batch_outputs
    
    def load_model(self):
        model = TDEModel.from_pretrained(self.config['model_ckpt_path'])
        return model 
    
    def get_user_context_features(self, batch_infer_df: DataFrame):
        '''
        get user and context features from redis
        Args:
            batch_infer_df: pd.DataFrame
        Returns:
            user_context_dict: dict
        '''
        # batch_infer_df : [B, M] 
        '''
        context_features:
        [
            "user_id",
            "device_id",
            "age",
            "gender",
            "province", 
            {"user_seq_effective_50" : ["video_id", "author_id", "category_level_two", "category_level_one", "upload_type"]}
        ]
        ''' 
        
        batch_infer_df = batch_infer_df.rename(mapper=(lambda col: col.strip(' ')), axis='columns')
        
        # user and context side features 
        context_features = [sub_feat for feat in self.feature_config['context_features'] 
                            for sub_feat in (list(feat.keys()) if isinstance(feat, dict) else [feat])]
        user_context_dict = self._row_get_features(
            batch_infer_df, 
            context_features, 
            [self.feature_cache_config['features'][feat] for feat in context_features])
        
        # expand features with nested keys
        for feat in self.feature_config['context_features']:
            if isinstance(feat, dict):
                for feat_name, feat_fields in feat.items():
                    if isinstance(user_context_dict[feat_name][0], Iterable):
                        cur_dict = defaultdict(list) 
                        # the case is for sequence features 
                        for proto_seq in user_context_dict[feat_name]:
                            for field in feat_fields: 
                                cur_list = [getattr(proto, field) for proto in proto_seq]
                                cur_dict[field].append(cur_list)
                        
                        user_context_dict[feat_name] = cur_dict
                    else:
                        cur_dict = {}
                        for proto in user_context_dict[feat_name]:
                            for field in feat_fields:
                                cur_dict[field].append(getattr(proto, field))
                        
                        user_context_dict[feat_name] = cur_dict
                        
        return user_context_dict
    
    def get_candidates_features(self, batch_candidates_df:pd.DataFrame):
        '''
        get candidates features from redis
        Args:
            batch_candidates_df (pd.DataFrame): shape = [B, N], each row is a list of candidates.
        Returns:
            candidates_dict: dict
        '''
        batch_candidates_df = batch_candidates_df.rename(mapper=(lambda col: col.strip(' ')), axis='columns')

        # candidates side features 
        # flatten candidates
        flattened_candidates_df = defaultdict(list) 
        num_candidates = len(batch_candidates_df.iloc[0, 0])
        for row in batch_candidates_df.itertuples():
            for col in batch_candidates_df.columns: 
                if (not isinstance(getattr(row, col), np.ndarray)) and (not isinstance(getattr(row, col), list)):
                    raise ValueError('All elements of each columns of batch_candidates_df should be np.ndarray or list')
                if num_candidates != len(getattr(row, col)):
                    raise ValueError('All features of one candidates should have equal length!')    
                flattened_candidates_df[col].extend(
                    getattr(row, col).tolist() if isinstance(getattr(row, col), np.ndarray) else getattr(row, col))

        flattened_candidates_df = pd.DataFrame(flattened_candidates_df)
        # get flattened candidate features
        flattened_candidates_dict = self._row_get_features(
            flattened_candidates_df, 
            self.feature_config['item_features'], 
            [self.feature_cache_config['features'][feat] for feat in self.feature_config['item_features']])
        # fold candidate features
        candidates_dict = {}
        for key, value in flattened_candidates_dict.items():
            candidates_dict['candidates_' + key] = [value[i * num_candidates : (i + 1) * num_candidates] \
                                                    for i in range(len(batch_candidates_df))]

        return candidates_dict 
    