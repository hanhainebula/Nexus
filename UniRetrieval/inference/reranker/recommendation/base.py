import torch
import numpy as np
from tqdm import tqdm, trange
from typing import Any, List, Union, Tuple, Optional,Dict, Literal
import math
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from UniRetrieval.abc.inference import AbsReranker
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
from UniRetrieval.abc.inference import AbsReranker, InferenceEngine, AbsInferenceArguments
import pycuda.driver as cuda
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
import pycuda.driver as cuda
import pycuda.autoinit

# from inference.inference.inference_engine import InferenceEngine
from UniRetrieval.inference.utils import gen_item_index, gen_i2i_index
from UniRetrieval.training.reranker.recommendation.modeling import BaseRanker

import faiss
import redis
import os
import time
import re
import importlib 
from collections import defaultdict
from collections.abc import Iterable
from abc import abstractmethod
import argparse
from copy import deepcopy 
from pandas import DataFrame
import pdb

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
import pycuda.autoinit

class BaseRerankerInferenceEngine(InferenceEngine):

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
        
        print('\n\n\n\n',cuda.Context.get_current(),'\n\n\n\n')
        
        # load model session
        self.convert_to_onnx()
        if config['infer_mode'] == 'ort':
            self.ort_session = self.get_ort_session()
            print(f'Session is using : {self.ort_session.get_providers()}')
        if config['infer_mode'] == 'trt':
            # pdb.set_trace()
            self.engine = self.get_trt_session()    
            
        # put seq into context_features
        # self.feature_config is deepcopy of self.model_ckpt_config['data_config']
        if 'seq_features' in self.model_ckpt_config['data_config']:
            self.feature_config['context_features'].append({'seq_effective_50' : self.model_ckpt_config['data_config']['seq_features']})
                
    def batch_inference(self, batch_infer_df:pd.DataFrame, batch_candidates_df:pd.DataFrame):
        '''
        batch inference
        Args:
            batch_infer_df: pd.DataFrame: batch of infer request.
            batch_candidates_df: pd.DataFrame: candidates of the batch request.
        Returns:
            batch_outputs: np.ndarray
        '''
        batch_st_time = time.time()

        # get user_context features 
        batch_user_context_dict = self.get_user_context_features(batch_infer_df)
        
        # get candidates features
        batch_candidates_dict = self.get_candidates_features(batch_candidates_df)
        # TODO: Cross Features

        feed_dict = {}
        feed_dict.update(batch_user_context_dict)
        feed_dict.update(batch_candidates_dict)
        feed_dict['output_topk'] = self.config['output_topk']
        for key in feed_dict:
            feed_dict[key] = np.array(feed_dict[key])
            
        if self.config['infer_mode'] == 'ort':
            batch_outputs_idx = self.ort_session.run(
                output_names=["output"],
                input_feed=feed_dict
            )[0]
        elif self.config['infer_mode'] == 'trt':
            batch_outputs_idx = self.infer_with_trt(feed_dict)
        # batch_outputs = batch_outputs_idx
        batch_outputs = []
        for row_idx, output_idx in enumerate(batch_outputs_idx):
            batch_outputs.append(
                np.array(batch_candidates_df.iloc[row_idx][self.feature_config['fiid']])[output_idx])
        batch_outputs = np.stack(batch_outputs, axis=0)
        batch_ed_time = time.time()
        # print(f'batch time: {batch_ed_time - batch_st_time}s')
        return batch_outputs
    
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

    def convert_to_onnx(self):
        """convert pytorch checkpoint to onnx model and then convert onnx model to ort session.
        
        Args:
            None
        Return: 
            onnxruntime.InferenceSession: The ONNX Runtime session object.
        """
        model = BaseRanker.from_pretrained(self.config['model_ckpt_path'])
        checkpoint = torch.load(os.path.join(self.config['model_ckpt_path'], 'model.pt'),
                                map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        
        model.eval()
        model.forward = model.predict 

        input_names = []
        dynamic_axes = {} 

        # user/context input
        context_input = {}
        for feat in self.feature_config['context_features']:
            if isinstance(feat, str):
                context_input[feat] = torch.randint(self.feature_config['stats'][feat], (5,))
                input_names.append(feat)
                dynamic_axes[feat] = {0: "batch_size"}
        seq_input = {}
        for field in self.feature_config['seq_features']:
            seq_input[field] = torch.randint(self.feature_config['stats'][field], (5, 50))
            input_names.append('seq_' + field)
            dynamic_axes['seq_' + field] = {0: "batch_size"}
        context_input['seq'] = seq_input
                        
        # candidates input 
        candidates_input = {} 
        for feat in self.feature_config['item_features']:
            if isinstance(feat, str):
                candidates_input[feat] = torch.randint(self.feature_config['stats'][feat], (5, 16))
                input_names.append('candidates_' + feat)
                dynamic_axes['candidates_' + feat] = {0: "batch_size", 1: "num_candidates"}

        output_topk = self.config['output_topk']
        input_names.append('output_topk')

        model_onnx_path = os.path.join(self.config['model_ckpt_path'], 'model_onnx.pb')
        torch.onnx.export(
            model,
            (context_input, candidates_input, output_topk), 
            model_onnx_path,
            input_names=input_names,
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=15,
            verbose=True
        )
    
    def convert_to_onnx_static(self):
        """Convert PyTorch checkpoint to ONNX model with static shapes.

        Args:
            None
        Return:
            None
        """
        model = BaseRanker.from_pretrained(self.config['model_ckpt_path'])
        checkpoint = torch.load(os.path.join(self.config['model_ckpt_path'], 'model.pt'),
                                map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)

        model.eval()
        model.forward = model.predict

        input_names = []
        # 设置静态形状
        static_batch_size = 128  # 固定的 batch size
        static_seq_length = 50  # 根据输入的实际序列长度设置
        static_num_candidates = 50  # 根据输入的实际候选数量设置

        # user/context input
        context_input = {}
        for feat in self.feature_config['context_features']:
            if isinstance(feat, str):
                context_input[feat] = torch.randint(self.feature_config['stats'][feat], (static_batch_size,))
                input_names.append(feat)

        # 获取序列长度
        seq_input = {}
        for field in self.feature_config['seq_features']:
            seq_tensor = torch.randint(self.feature_config['stats'][field], (static_batch_size, 50))  # 假设初始序列长度为 50
            # static_seq_length = seq_tensor.shape[1]  # 根据实际输入设置序列长度
            seq_input[field] = seq_tensor
            input_names.append('seq_' + field)
        context_input['seq'] = seq_input

        # 获取候选数量
        candidates_input = {}
        for feat in self.feature_config['item_features']:
            if isinstance(feat, str):
                candidates_tensor = torch.randint(self.feature_config['stats'][feat], (static_batch_size, 50))  
                # static_num_candidates = candidates_tensor.shape[1]  # 根据实际输入设置候选数量
                candidates_input[feat] = candidates_tensor
                input_names.append('candidates_' + feat)

        output_topk = self.config['output_topk']
        input_names.append('output_topk')

        # 打印静态形状信息
        print(f"Static shapes - batch_size: {static_batch_size}, seq_length: {static_seq_length}, num_candidates: {static_num_candidates}")
        model_onnx_path = os.path.join(self.config['model_ckpt_path'], 'model_onnx.pb')
        torch.onnx.export(
            model,
            (context_input, candidates_input, output_topk),
            model_onnx_path,
            input_names=input_names,
            output_names=["output"],
            opset_version=15,
            verbose=False
        )
    
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
            {"seq_effective_50" : ["video_id", "author_id", "category_level_two", "category_level_one", "upload_type"]}
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
        
        for feat in self.feature_config['context_features']:
            if isinstance(feat, dict):
                for feat_name, feat_fields in feat.items():
                    cur_dict = defaultdict(list) 
                    if isinstance(user_context_dict[feat_name][0], Iterable):
                        for seq in user_context_dict[feat_name]:
                            for field in feat_fields: 
                                cur_list = [getattr(proto, field) for proto in seq]
                                cur_dict[feat_name + '_' + field].append(cur_list)
                        user_context_dict.update(cur_dict)  
                        del user_context_dict[feat_name]
                    else:
                        for proto in user_context_dict[feat_name]:
                            for field in feat_fields:
                                cur_dict[feat_name + '_' + field].append(getattr(proto, field))
                        del user_context_dict[feat_name]
 
        batch_user_context_dict = user_context_dict
        
        for k, v in list(batch_user_context_dict.items()):
            if k.startswith('seq_effective_50_'):
                batch_user_context_dict['seq_' + k[len('seq_effective_50_'):]] = v
                del batch_user_context_dict[k]
        
        return batch_user_context_dict 
    
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
            for key_temp in feats_key_temp_list:
                key_feats = re.findall('{(.*?)}', key_temp) 
                cur_key = key_temp
                for key_feat in key_feats:
                    cur_key = cur_key.replace(f'{{{key_feat}}}', str(getattr(row, key_feat)))
                cur_all_values[key_temp] = feats_k2p[cur_key]

            for feat, cache in zip(feats_list, feats_cache_list):
                res_dict[feat].append(getattr(cur_all_values[cache['key_temp']], cache['field']))
        
        return res_dict

    def get_normal_session(self):
        pass
    
    def get_onnx_session(self):
        pass
    
    def get_ort_session(self) -> ort.InferenceSession:
        model_onnx_path = os.path.join(self.config['model_ckpt_path'], 'model_onnx.pb')
        # print graph
        onnx_model = onnx.load(model_onnx_path)
        onnx.checker.check_model(onnx_model)
        print("=" * 25 + 'comp graph : ' + "=" * 25)
        print(onnx.helper.printable_graph(onnx_model.graph))

        if self.config['infer_device'] == 'cpu':
            providers = ["CPUExecutionProvider"]
        elif isinstance(self.config['infer_device'], int):
            providers = [("CUDAExecutionProvider", {"device_id": self.config['infer_device']})]
        return ort.InferenceSession(model_onnx_path, providers=providers)
    
    
    def get_trt_session(self):
        model_onnx_path = os.path.join(self.config['model_ckpt_path'], 'model_onnx.pb')
        trt_engine_path = os.path.join(self.config['model_ckpt_path'], 'model_trt.engine')
        # print('model_onnx_path:',model_onnx_path)
        # Set the GPU device
        
        cuda.Device(self.config['infer_device']).make_context()
        print('\n\n\n\n',cuda.Context.get_current(),'\n\n\n\n')
        # Build or load the engine
        if not os.path.exists(trt_engine_path):
            serialized_engine = self.build_engine(model_onnx_path, trt_engine_path)
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(serialized_engine)
        else:
            print(f'Loading engine from {trt_engine_path}')
            engine = self.load_engine(trt_engine_path)
        
        
        return engine
    
    def load_engine(self, engine_file_path):
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
        with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
        
    
    def load_engine_old(self, trt_engine_path):
        device=self.config['infer_device']
        if not isinstance(device, int):
            device=0
        # cuda.Device(device).make_context()
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(trt_engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    
    # def build_engine(self, onnx_file_path, engine_file_path):
        
    #     TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
        
    #     with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
    #         with open(onnx_file_path, 'rb') as model:
    #             if not parser.parse(model.read()):
    #                 for error in range(parser.num_errors):
    #                     print('\n\n\n\n', parser.get_error(error), '\n\n\n\n')
    #                 raise RuntimeError('Failed to parse ONNX model')

    #         config = builder.create_builder_config()
    #         config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    #     # 检查并确保所有输入张量的形状是静态的
    #         for i in range(network.num_inputs):
    #             input_tensor = network.get_input(i)
    #             input_shape = input_tensor.shape
    #             if -1 in input_shape:  # 如果存在动态维度
    #                 raise RuntimeError(f"Input {input_tensor.name} has dynamic shape {input_shape}. Static shape is required.")
                
    #         # Build and serialize the engine
    #         serialized_engine = builder.build_serialized_network(network, config)
    #         with open(engine_file_path, 'wb') as f:
    #             f.write(serialized_engine)
    #         return serialized_engine
    
    
    def build_engine(self, onnx_file_path, engine_file_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        print('\n\n\n\n',cuda.Context.get_current(),'\n\n\n\n')
        
        onnx_model = onnx.load(onnx_file_path)
        onnx.checker.check_model(onnx_model)
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            # Parse ONNX model
            with open(onnx_file_path, 'rb') as model:
                # print('\n\n\n\n',model.read(),'\n\n\n\n')
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        print('\n\n\n\n',parser.get_error(error),'\n\n\n\n')
                    raise RuntimeError('Failed to parse ONNX model')

            # Create builder config
            config = builder.create_builder_config()
            print('builder 1:',builder)
            print('config 1:',config)
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

            print('builder 2:',builder)
            print('config 2:',config)
            
            # Create optimization profile
            profile = builder.create_optimization_profile()
            
            print('builder 3:',profile)

            for i in range(network.num_inputs):
                input_name = network.get_input(i).name
                input_shape = network.get_input(i).shape
                # Set the min, opt, and max shapes for the input
                min_shape = [1 if dim == -1 else dim for dim in input_shape]
                opt_shape = [self.config['infer_batch_size'] if dim == -1 else dim for dim in input_shape]
                max_shape = [self.config['infer_batch_size'] if dim == -1 else dim for dim in input_shape]
                profile.set_shape(input_name, tuple(min_shape), tuple(opt_shape), tuple(max_shape))
            config.add_optimization_profile(profile)
            print('config 3:',config)
            # Build and serialize the engine
            serialized_engine = builder.build_serialized_network(network, config)
            print('serialized_engine:',serialized_engine)
            with open(engine_file_path, 'wb') as f:
                f.write(serialized_engine)
            return serialized_engine
    
    
    def infer_with_trt(self, inputs):
        
        
        with self.engine.create_execution_context() as context:
            stream = cuda.Stream()
            bindings = [0] * self.engine.num_io_tensors

            input_memory = []
            output_buffers = {}
            for i in range(self.engine.num_io_tensors):
                tensor_name = self.engine.get_tensor_name(i)
                # print('tensor_name:', tensor_name)
                # print('self.engine.get_tensor_mode(tensor_name):', self.engine.get_tensor_mode(tensor_name))
                # print('tensor shape:', self.engine.get_tensor_shape(tensor_name))
                
                dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
                if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    # print('inputs[tensor_name]:', inputs[tensor_name])
                    # print('inputs[tensor_name].type:', type(inputs[tensor_name]))
                    # print('inputs[tensor_name].dtype:', inputs[tensor_name].dtype)
                    if -1 in tuple(self.engine.get_tensor_shape(tensor_name)):  # dynamic
                        print(f'dynamic :{tensor_name}')
                        context.set_input_shape(tensor_name, tuple(self.engine.get_tensor_profile_shape(tensor_name, 0)[2]))
                        # context.set_input_shape(tensor_name, tuple(inputs[tensor_name].shape))
                    input_mem = cuda.mem_alloc(inputs[tensor_name].nbytes)
                    bindings[i] = int(input_mem)
                    context.set_tensor_address(tensor_name, int(input_mem))
                    cuda.memcpy_htod_async(input_mem, inputs[tensor_name], stream)
                    input_memory.append(input_mem)
                else:  # output
                    print('at else.****************')
                    print('dtype:',dtype)
                    shape = tuple(self.engine.get_tensor_shape(tensor_name))
                    pdb.set_trace()
                    output_buffer = np.empty(shape, dtype=dtype)
                    output_buffer = np.ascontiguousarray(output_buffer)
                    output_memory = cuda.mem_alloc(output_buffer.nbytes)
                    bindings[i] = int(output_memory)
                    context.set_tensor_address(tensor_name, output_memory)
                    cuda.memcpy_htod_async(output_memory, output_buffer, stream)
                    output_buffers[tensor_name] = (output_buffer, output_memory)

            context.execute_async_v3(stream_handle=stream.handle)
            stream.synchronize()

            for tensor_name, (output_buffer, output_memory) in output_buffers.items():
                cuda.memcpy_dtoh(output_buffer, output_memory)
        
        
        return output_buffers['output'][0]
    
    def get_tensorrt_session(self):
        pass
    
    def inference(self):
        pass
    
    def load_model(self):
        pass
    