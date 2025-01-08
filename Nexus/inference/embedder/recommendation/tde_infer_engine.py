import os
from tqdm import tqdm, trange
from typing import cast, Any, List, Union, Optional, Tuple, Type
import onnx
import onnxruntime as ort
import tensorrt as trt
import pandas as pd
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from dataclasses import dataclass, field
from Nexus.abc.inference import AbsEmbedder, InferenceEngine, AbsInferenceArguments
import os
import sys
sys.path.append('.')
import argparse
import time

from pandas import DataFrame
import yaml
import torch
import numpy as np

import tensorrt as trt

# from inference.inference.inference_engine import InferenceEngine
from Nexus.inference.utils import gen_item_index, gen_i2i_index
from Nexus.training.embedder.recommendation.modeling import BaseRetriever

import faiss
import redis
import os
import time
import re
import importlib 
from collections import defaultdict
from collections.abc import Iterable
import argparse
from copy import deepcopy 
from pandas import DataFrame

from tqdm import tqdm
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

from dynamic_embedding.wrappers import BatchWrapper
from torchrec.distributed import DistributedModelParallel
from Nexus.training.embedder.recommendation.tde_modeling import TDEModel


class TDEEmbedderInferenceEngine(InferenceEngine):

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
        
        # load model
        self.model:DistributedModelParallel = self.load_model()
        self.model.module.model_config.topk = self.config['output_topk']
        
        # get batch wrapper
        self.batch_wrapper = BatchWrapper(self.model._id_transformer_group,
                                          self.model.module.tde_configs_dict,
                                          list(self.model.module.tde_configs_dict.keys()))
        
        # put seq into context_features
        # self.feature_config is deepcopy of self.model_ckpt_config['data_config']
        if 'seq_features' in self.model_ckpt_config['data_config']:
            self.feature_config['context_features'].append(self.model_ckpt_config['data_config']['seq_features'])
        
        self.retrieve_index_config = config['retrieve_index_config']

        if config['stage'] == 'retrieve':
            if self.retrieve_index_config.get('gen_item_index', True):
                gen_item_index(os.path.join(config['model_ckpt_path'], 'item_vectors.pt'), 
                               self.retrieve_index_config['item_index_path'],
                               self.retrieve_index_config['item_ids_path'])
            if config['retrieval_mode'] == 'u2i':
                self.item_index = faiss.read_index(self.retrieve_index_config['item_index_path'])
                self.item_index.nprobe = self.retrieve_index_config['nprobe']
                self.item_ids_table = np.load(self.retrieve_index_config['item_ids_path'])
            elif config['retrieval_mode'] == 'i2i':
                if self.retrieve_index_config.get('gen_i2i_index', True):
                    gen_i2i_index(config['output_topk'],
                                  config['model_ckpt_path'], 
                                  self.retrieve_index_config['i2i_redis_host'],
                                  self.retrieve_index_config['i2i_redis_port'],
                                  self.retrieve_index_config['i2i_redis_db'],
                                  item_index_path=self.retrieve_index_config['item_index_path'])
                self.i2i_redis_client = redis.Redis(host=self.retrieve_index_config['i2i_redis_host'], 
                                    port=self.retrieve_index_config['i2i_redis_port'], 
                                    db=self.retrieve_index_config['i2i_redis_db'])
                
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
                
    def batch_inference(self, batch_infer_df:DataFrame):
        """
        Perform batch inference for a given batch of data.
        Args:
            batch_infer_df (DataFrame): A pandas DataFrame containing the batch of data to perform inference on.

        Returns:
            (np.ndarray):
                shape [batch_size, output_topk], the recommended items based on the provided inputs.
                - If retrieval_mode is 'u2i', returns a list of item IDs corresponding to the top-k recommendations for each user.
                - If retrieval_mode is 'i2i', returns a list of lists of item IDs representing the recommendations for each sequence of video IDs.
        """
        # iterate infer data 
        batch_st_time = time.time()

        # get user_context features 
        batch_user_context_dict = self.get_user_context_features(batch_infer_df)
        
        feed_dict = {}
        feed_dict.update(batch_user_context_dict)
        feed_dict = self._batch_to_tensor(feed_dict)
        feed_dict = self.batch_wrapper(feed_dict)
        feed_dict = self._batch_to_device(feed_dict, self.model.device)
        
        if self.config['retrieval_mode'] == 'u2i':
            batch_user_embedding = self.model.module.encode_query(feed_dict)
            user_embedding_np = batch_user_embedding[:batch_infer_df.shape[0]].detach().cpu().numpy()
            # user_embedding_noise = np.random.normal(loc=0.0, scale=0.01, size=user_embedding_np.shape)
            # user_embedding_np = user_embedding_np + user_embedding_noise
            D, I = self.item_index.search(user_embedding_np, self.config['output_topk'])
            batch_outputs = I
        elif self.config['retrieval_mode'] == 'i2i':
            seq_video_ids = feed_dict['seq_video_id']
            batch_outputs = self.get_i2i_recommendations(seq_video_ids)
        
        # print(batch_outputs.shape)
        batch_ed_time = time.time()
        # print(f'batch time: {batch_ed_time - batch_st_time}s')
      
        if self.config['retrieval_mode'] == 'u2i':
            return self.item_ids_table[batch_outputs]
        else:
            return batch_outputs
        
    def load_model(self):
        model = TDEModel.from_pretrained(self.config['model_ckpt_path'])
        return model
    
    def get_i2i_recommendations(self, seq_video_ids_batch):
        pipeline = self.i2i_redis_client.pipeline()
        for seq_video_ids in seq_video_ids_batch:
            for video_id in seq_video_ids:
                redis_key = f'item:{video_id}'
                pipeline.get(redis_key)
        results = pipeline.execute()

        all_top10_items = []

        for result in results:
            if result:
                top10_items = result.decode('utf-8').split(',')
                all_top10_items.extend(top10_items)
            else:
                print('Redis returned None for a key')

        all_top10_items = list(map(int, all_top10_items))
        all_top10_items = np.array(all_top10_items).reshape(seq_video_ids_batch.shape[0], -1)

        return all_top10_items
    
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
    
    def _row_get_features(self, row_df:pd.DataFrame, feats_list, feats_cache_list):
        # each row represents one entry 
        # row_df: [B, M]
        res_dict = defaultdict(list)
        # key_temp list 
        feats_key_temp_list = list(set([cache['key_temp'] for cache in feats_cache_list]))
        
        # get all keys and values related to these rows in one time 
        with self.redis_client.pipeline() as pipe:
            feats_all_key_and_temp = set()
            for row in row_df.itertuples():
                for key_temp in feats_key_temp_list:
                    key_feats = re.findall('{(.*?)}', key_temp) 
                    cur_key = key_temp
                    for key_feat in key_feats:
                        cur_key = cur_key.replace(f'{{{key_feat}}}', str(getattr(row, key_feat)))
                    feats_all_key_and_temp.add((cur_key, key_temp))
            feats_all_key_and_temp = list(feats_all_key_and_temp)

            redis_st_time = time.time()
            for key, _ in feats_all_key_and_temp:
                pipe.get(key)
            feats_all_values = pipe.execute()
            redis_ed_time = time.time()
            # print(f'redis time : {(redis_ed_time - redis_st_time)}s')
        
        parse_st_time = time.time()
        feats_k2p = {}
        for (key, key_temp), value in zip(feats_all_key_and_temp, feats_all_values):
            value_proto = self.key_temp2proto[key_temp]()
            value_proto.ParseFromString(value)
            feats_k2p[key] = value_proto
        parse_ed_time = time.time()
        # print(f'parse time : {(parse_ed_time - parse_st_time)}s')

        # get feats from these values
        for row in row_df.itertuples():
            cur_all_values = dict()
            # get all protos that this row needs 
            for key_temp in feats_key_temp_list:
                key_feats = re.findall('{(.*?)}', key_temp) 
                cur_key = key_temp
                for key_feat in key_feats:
                    cur_key = cur_key.replace(f'{{{key_feat}}}', str(getattr(row, key_feat)))
                cur_all_values[key_temp] = feats_k2p[cur_key]

            # get features from these protos 
            for feat, cache in zip(feats_list, feats_cache_list):
                res_dict[feat].append(getattr(cur_all_values[cache['key_temp']], cache['field']))
        
        return res_dict

    def get_ort_session(self) -> ort.InferenceSession:
        pass

    def get_trt_session(self):
        pass
    
    def convert_to_onnx(self):
        pass

    