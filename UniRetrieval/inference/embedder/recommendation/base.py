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
from UniRetrieval.abc.inference import AbsEmbedder, InferenceEngine, AbsInferenceArguments
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
from UniRetrieval.inference.utils import gen_item_index, gen_i2i_index
from UniRetrieval.training.embedder.recommendation.modeling import BaseRetriever

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
import pycuda.autoinit

class BaseEmbedder(AbsEmbedder):
    """
    Base embedder for encoder only models.

    Args:
        model_name_or_path (str): If it's a path to a local model, it loads the model from the path. Otherwise tries to download and
            load a model from HuggingFace Hub with the name.
        normalize_embeddings (bool, optional): If True, normalize the embedding vector. Defaults to :data:`True`.
        use_fp16 (bool, optional): If true, use half-precision floating-point to speed up computation with a slight performance 
            degradation. Defaults to :data:`True`.
        query_instruction_for_retrieval (Optional[str], optional): Query instruction for retrieval tasks, which will be used with
            with :attr:`query_instruction_format`. Defaults to :data:`None`.
        query_instruction_format (str, optional): The template for :attr:`query_instruction_for_retrieval`. Defaults to :data:`"{}{}"`.
        devices (Optional[Union[str, int, List[str], List[int]]], optional): Devices to use for model inference. Defaults to :data:`None`.
        pooling_method (str, optional): Pooling method to get embedding vector from the last hidden state. Defaults to :data:`"cls"`.
        trust_remote_code (bool, optional): trust_remote_code for HF datasets or models. Defaults to :data:`False`.
        cache_dir (Optional[str], optional): Cache directory for the model. Defaults to :data:`None`.
        batch_size (int, optional): Batch size for inference. Defaults to :data:`256`.
        query_max_length (int, optional): Maximum length for query. Defaults to :data:`512`.
        passage_max_length (int, optional): Maximum length for passage. Defaults to :data:`512`.
        convert_to_numpy (bool, optional): If True, the output embedding will be a Numpy array. Otherwise, it will be a Torch Tensor. 
            Defaults to :data:`True`.
    
    Attributes:
        DEFAULT_POOLING_METHOD: The default pooling method when running the model.
    """
    
    DEFAULT_POOLING_METHOD = "cls"

    def __init__(
        self,
        model_name_or_path: str,
        normalize_embeddings: bool = True,
        use_fp16: bool = True,
        query_instruction_for_retrieval: Optional[str] = None,
        query_instruction_format: str = "{}{}", # specify the format of query_instruction_for_retrieval
        devices: Optional[Union[str, List[str]]] = None, # specify devices, such as "cuda:0" or ["cuda:0", "cuda:1"]
        # Additional parameters for BaseEmbedder
        pooling_method: str = "cls",
        trust_remote_code: bool = False,
        cache_dir: Optional[str] = None,
        # inference
        batch_size: int = 256,
        query_max_length: int = 512,
        passage_max_length: int = 512,
        convert_to_numpy: bool = True,
        *args,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        query_instruction_format = query_instruction_format.replace('\\n', '\n')
        self.model_name_or_path = model_name_or_path
        self.normalize_embeddings = normalize_embeddings
        self.use_fp16 = use_fp16
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.query_instruction_format = query_instruction_format
        self.target_devices = self.get_target_devices(devices)

        self.batch_size = batch_size
        self.query_max_length = query_max_length
        self.passage_max_length = passage_max_length
        self.convert_to_numpy = convert_to_numpy
        
        for k in kwargs:
            setattr(self, k, kwargs[k])
            
        self.kwargs = kwargs
        
        self.pool = None

        self.pooling_method = pooling_method

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir
        )
        
    @staticmethod
    def get_detailed_instruct(instruction_format: str, instruction: str, sentence: str):
        """Combine the instruction and sentence along with the instruction format.

        Args:
            instruction_format (str): Format for instruction.
            instruction (str): The text of instruction.
            sentence (str): The sentence to concatenate with.

        Returns:
            str: The complete sentence with instruction
        """
        return instruction_format.format(instruction, sentence)
    

    def encode_query(
        self,
        queries: Union[List[str], str],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        convert_to_numpy: Optional[bool] = None,
        **kwargs: Any
    ):
        """encode the queries using the instruction if provided.

        Args:
            queries (Union[List[str], str]): Input queries to encode.
            batch_size (Optional[int], optional): Number of sentences for each iter. Defaults to :data:`None`.
            max_length (Optional[int], optional): Maximum length of tokens. Defaults to :data:`None`.
            convert_to_numpy (Optional[bool], optional): If True, the output embedding will be a Numpy array. Otherwise, it will 
                be a Torch Tensor. Defaults to :data:`None`.

        Returns:
            Union[torch.Tensor, np.ndarray]: Return the embedding vectors in a numpy array or tensor.
        """
        if batch_size is None: batch_size = self.batch_size
        if max_length is None: max_length = self.query_max_length
        if convert_to_numpy is None: convert_to_numpy = self.convert_to_numpy

        return self.encode(
            queries,
            batch_size=batch_size,
            max_length=max_length,
            convert_to_numpy=convert_to_numpy,
            instruction=self.query_instruction_for_retrieval,
            instruction_format=self.query_instruction_format,
            **kwargs
        )


    def encode_info(
        self,
        corpus: Union[List[str], str],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        convert_to_numpy: Optional[bool] = None,
        **kwargs: Any
    ):
        """encode the corpus using the instruction if provided.

        Args:
            corpus (Union[List[str], str]): Input corpus to encode.
            batch_size (Optional[int], optional): Number of sentences for each iter. Defaults to :data:`None`.
            max_length (Optional[int], optional): Maximum length of tokens. Defaults to :data:`None`.
            convert_to_numpy (Optional[bool], optional): If True, the output embedding will be a Numpy array. Otherwise, it will 
                be a Torch Tensor. Defaults to :data:`None`.

        Returns:
            Union[torch.Tensor, np.ndarray]: Return the embedding vectors in a numpy array or tensor.
        """
        passage_instruction_for_retrieval = self.kwargs.get("passage_instruction_for_retrieval", None)
        passage_instruction_format = self.kwargs.get("passage_instruction_format", "{}{}")

        if batch_size is None: batch_size = self.batch_size
        if max_length is None: max_length = self.passage_max_length
        if convert_to_numpy is None: convert_to_numpy = self.convert_to_numpy

        return self.encode(
            corpus,
            batch_size=batch_size,
            max_length=max_length,
            convert_to_numpy=convert_to_numpy,
            instruction=passage_instruction_for_retrieval,
            instruction_format=passage_instruction_format,
            **kwargs
        )

    def encode(
        self,
        sentences: Union[List[str], str],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        convert_to_numpy: Optional[bool] = None,
        instruction: Optional[str] = None,
        instruction_format: Optional[str] = None,
        **kwargs: Any
    ):
        """encode the input sentences with the embedding model.

        Args:
            sentences (Union[List[str], str]): Input sentences to encode.
            batch_size (Optional[int], optional): Number of sentences for each iter. Defaults to :data:`None`.
            max_length (Optional[int], optional): Maximum length of tokens. Defaults to :data:`None`.
            convert_to_numpy (Optional[bool], optional): If True, the output embedding will be a Numpy array. Otherwise, it will 
                be a Torch Tensor. Defaults to :data:`None`.
            instruction (Optional[str], optional): The text of instruction. Defaults to :data:`None`.
            instruction_format (Optional[str], optional): Format for instruction. Defaults to :data:`None`.

        Returns:
            Union[torch.Tensor, np.ndarray]: return the embedding vectors in a numpy array or tensor.
        """
        if batch_size is None: batch_size = self.batch_size
        if max_length is None: max_length = self.passage_max_length
        if convert_to_numpy is None: convert_to_numpy = self.convert_to_numpy

        if instruction is not None:
            if isinstance(sentences, str):
                sentences = self.get_detailed_instruct(instruction_format, instruction, sentences)
            else:
                sentences = [self.get_detailed_instruct(instruction_format, instruction, sentence) for sentence in
                             sentences]

        if isinstance(sentences, str) or len(self.target_devices) == 1:
            return self.encode_single_device(
                sentences,
                batch_size=batch_size,
                max_length=max_length,
                convert_to_numpy=convert_to_numpy,
                device=self.target_devices[0],
                **kwargs
            )

        if self.pool is None:
            self.pool = self.start_multi_process_pool(AbsEmbedder._encode_multi_process_worker)
        embeddings = self.encode_multi_process(
            sentences,
            self.pool,
            batch_size=batch_size,
            max_length=max_length,
            convert_to_numpy=convert_to_numpy,
            **kwargs
        )
        return embeddings


    @torch.no_grad()
    def encode_single_device(
        self,
        sentences: Union[List[str], str],
        batch_size: int = 256,
        max_length: int = 512,
        convert_to_numpy: bool = True,
        device: Optional[str] = None,
        **kwargs: Any
    ):
        """Encode input sentences by a single device.

        Args:
            sentences (Union[List[str], str]): Input sentences to encode.
            batch_size (int, optional): Number of sentences for each iter. Defaults to :data:`256`.
            max_length (int, optional): Maximum length of tokens. Defaults to :data:`512`.
            convert_to_numpy (bool, optional): If True, the output embedding will be a Numpy array. Otherwise, it will 
                be a Torch Tensor. Defaults to :data:`True`.
            device (Optional[str], optional): Device to use for encoding. Defaults to None.

        Returns:
            Union[torch.Tensor, np.ndarray]: return the embedding vectors in a numpy array or tensor.
        """
        if device is None:
            device = self.target_devices[0]

        if device == "cpu": self.use_fp16 = False
        if self.use_fp16: self.model.half()

        self.model.to(device)
        self.model.eval()

        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True

        # tokenize without padding to get the correct length
        all_inputs = []
        for start_index in trange(0, len(sentences), batch_size, desc='pre tokenize',
                                  disable=len(sentences) < 256):
            sentences_batch = sentences[start_index:start_index + batch_size]
            inputs_batch = self.tokenizer(
                sentences_batch,
                truncation=True,
                max_length=max_length,
                **kwargs
            )
            inputs_batch = [{
                k: inputs_batch[k][i] for k in inputs_batch.keys()
            } for i in range(len(sentences_batch))]
            all_inputs.extend(inputs_batch)

        # sort by length for less padding
        length_sorted_idx = np.argsort([-len(x['input_ids']) for x in all_inputs])
        all_inputs_sorted = [all_inputs[i] for i in length_sorted_idx]

        # adjust batch size
        flag = False
        while flag is False:
            try:
                inputs_batch = self.tokenizer.pad(
                    all_inputs_sorted[: batch_size],
                    padding=True,
                    return_tensors='pt',
                    **kwargs
                ).to(device)
                last_hidden_state = self.model(**inputs_batch, return_dict=True).last_hidden_state
                embeddings = self.pooling(last_hidden_state, inputs_batch['attention_mask'])
                flag = True
            except RuntimeError as e:
                batch_size = batch_size * 3 // 4
            except torch.OutofMemoryError as e:
                batch_size = batch_size * 3 // 4

        # encode
        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Inference Embeddings",
                                disable=len(sentences) < 256):
            inputs_batch = all_inputs_sorted[start_index:start_index + batch_size]
            inputs_batch = self.tokenizer.pad(
                inputs_batch,
                padding=True,
                return_tensors='pt',
                **kwargs
            ).to(device)
            last_hidden_state = self.model(**inputs_batch, return_dict=True).last_hidden_state
            embeddings = self.pooling(last_hidden_state, inputs_batch['attention_mask'])
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            embeddings = cast(torch.Tensor, embeddings)

            if convert_to_numpy:
                embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)

        if convert_to_numpy:
            all_embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            all_embeddings = torch.cat(all_embeddings, dim=0)

        # adjust the order of embeddings
        all_embeddings = all_embeddings[np.argsort(length_sorted_idx)]

        # return the embeddings
        if input_was_string:
            return all_embeddings[0]
        return all_embeddings

    def pooling(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ):
        """The pooling function.

        Args:
            last_hidden_state (torch.Tensor): The last hidden state of the model.
            attention_mask (Optional[torch.Tensor], optional): Attention mask. Defaults to :data:`None`.

        Raises:
            NotImplementedError: pooling method not implemented.

        Returns:
            torch.Tensor: The embedding vectors after pooling.
        """
        if self.pooling_method == 'cls':
            return last_hidden_state[:, 0]
        elif self.pooling_method == 'mean':
            s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d
        else:
            raise NotImplementedError(f"pooling method {self.pooling_method} not implemented")
        
        
@dataclass
class SessionParams():
    name: str=field(
        default='query',
        metadata={
            'help':'Name of Session Inputs/Outputs'
        }
    )


class NormalSession():
    """
    A class used to represent a Normal Session for text retrieval.
    Attributes
    ----------
    model : object
        The model used for encoding queries and corpus.
    Methods
    -------
    get_inputs():
        Returns a list of SessionParams representing the input names.
    get_outputs():
        Returns a list of SessionParams representing the output names.
    run(output_names, input_feed, run_options=None):
        Executes the session with the given inputs and returns the embeddings.
    """
    def __init__(self, model):
        self.model=model

    def get_inputs(self):
        return [SessionParams(
            name='sentences'
        )]
        
    def get_outputs(self):
        return [SessionParams(
            name='embeddings'
        )]

    def run(self, output_names, input_feed, run_options=None):
        sentences=input_feed['sentences']            
        embeddings = self.model.encode(sentences)
                    
        return [embeddings]
        
    


# class BaseEmbedderInferenceEngine(InferenceEngine):
class BaseEmbedderInferenceEngine(InferenceEngine):

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
        
        # load model session
        self.convert_to_onnx()
        if config['infer_mode'] == 'ort':
            self.ort_session = self.get_ort_session()
            print(f'Session is using : {self.ort_session.get_providers()}')
        if config['infer_mode'] == 'trt':
            self.engine = self.get_trt_session()    
            
        # put seq into context_features
        # self.feature_config is deepcopy of self.model_ckpt_config['data_config']
        if 'seq_features' in self.model_ckpt_config['data_config']:
            self.feature_config['context_features'].append({'seq_effective_50': self.model_ckpt_config['data_config']['seq_features']})

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
        for key in feed_dict:
            feed_dict[key] = np.array(feed_dict[key])

        if self.config['retrieval_mode'] == 'u2i':
            if self.config['infer_mode'] == 'ort':
                batch_user_embedding = self.ort_session.run(
                    output_names=["output"],
                    input_feed=feed_dict
                )[0]
            elif self.config['infer_mode'] == 'trt':
                batch_user_embedding = self.infer_with_trt(feed_dict)

            user_embedding_np = batch_user_embedding[:batch_infer_df.shape[0]]
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
        
    def infer_with_trt(self, inputs):
        with self.engine.create_execution_context() as context:
            stream = cuda.Stream()
            bindings = [0] * self.engine.num_io_tensors

            input_memory = []
            output_buffers = {}
            for i in range(self.engine.num_io_tensors):
                tensor_name = self.engine.get_tensor_name(i)
                dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
                if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    if -1 in tuple(self.engine.get_tensor_shape(tensor_name)):  # dynamic
                        context.set_input_shape(tensor_name, tuple(self.engine.get_tensor_profile_shape(tensor_name, 0)[2]))
                    input_mem = cuda.mem_alloc(inputs[tensor_name].nbytes)
                    bindings[i] = int(input_mem)
                    context.set_tensor_address(tensor_name, int(input_mem))
                    cuda.memcpy_htod_async(input_mem, inputs[tensor_name], stream)
                    input_memory.append(input_mem)
                else:  # output
                    shape = tuple(context.get_tensor_shape(tensor_name))
                    output_buffer = np.empty(shape, dtype=dtype)
                    output_buffer = np.ascontiguousarray(output_buffer)
                    output_memory = cuda.mem_alloc(output_buffer.nbytes)
                    bindings[i] = int(output_memory)
                    context.set_tensor_address(tensor_name, int(output_memory))
                    output_buffers[tensor_name] = (output_buffer, output_memory)

            context.execute_async_v3(stream_handle=stream.handle)
            stream.synchronize()

            for tensor_name, (output_buffer, output_memory) in output_buffers.items():
                cuda.memcpy_dtoh(output_buffer, output_memory)

        return output_buffers['output'][0]
    
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
        model = BaseRetriever.from_pretrained(self.config['model_ckpt_path'])
        checkpoint = torch.load(os.path.join(self.config['model_ckpt_path'], 'model.pt'),
                                map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        
        model.eval()
        model.forward = model.encode_query

        input_names = []
        dynamic_axes = {} 

        # user/context input
        context_input = {}
        for feat in self.feature_config['context_features']:
            if isinstance(feat, str):
                context_input[feat] = torch.randint(self.feature_config['stats'][feat], (5,))
                input_names.append(feat)
                dynamic_axes[feat] = {0: "batch_size"}
        # TODO: need to be modified for better universality 
        # special process for seq feature 
        # TODO: how to get the length of the seq 
        seq_input = {}
        for field in self.feature_config['seq_features']:
            seq_input[field] = torch.randint(self.feature_config['stats'][field], (5, 50))
            input_names.append('seq_' + field)
            dynamic_axes['seq_' + field] = {0: "batch_size"}
        context_input['seq'] = seq_input

        model_onnx_path = os.path.join(self.config['model_ckpt_path'], 'model_onnx.pb')
        torch.onnx.export(
            model,
            (context_input, {}), 
            model_onnx_path,
            input_names=input_names,
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=15,
            verbose=True
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

        # Set the GPU device
        cuda.Device(self.config['infer_device']).make_context()

        # Build or load the engine
        if not os.path.exists(trt_engine_path):
            serialized_engine = self.build_engine(model_onnx_path, trt_engine_path)
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(serialized_engine)
        else:
            engine = self.load_engine(trt_engine_path)

        return engine
    
    def load_engine(self, engine_file_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    
    def build_engine(self, onnx_file_path, engine_file_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            # Parse ONNX model
            with open(onnx_file_path, 'rb') as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    raise RuntimeError('Failed to parse ONNX model')

            # Create builder config
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

            # Create optimization profile
            profile = builder.create_optimization_profile()
            for i in range(network.num_inputs):
                input_name = network.get_input(i).name
                input_shape = network.get_input(i).shape
                # Set the min, opt, and max shapes for the input
                min_shape = [1 if dim == -1 else dim for dim in input_shape]
                opt_shape = [self.config['infer_batch_size'] if dim == -1 else dim for dim in input_shape]
                max_shape = [self.config['infer_batch_size'] if dim == -1 else dim for dim in input_shape]
                profile.set_shape(input_name, tuple(min_shape), tuple(opt_shape), tuple(max_shape))
            config.add_optimization_profile(profile)

            # Build and serialize the engine
            serialized_engine = builder.build_serialized_network(network, config)
            with open(engine_file_path, 'wb') as f:
                f.write(serialized_engine)
            return serialized_engine
        
    def get_tensorrt_session(self):
        pass
    
    def inference(self):
        pass
    
    def load_model(self):
        pass
    