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
        
    


class BaseEmbedderInferenceEngine(InferenceEngine):
    def __init__(self, infer_args: AbsInferenceArguments):
        super().__init__(infer_args)
        # normal model
        self.load_model()
        # session
        self.session = self.get_inference_session()
        
    def load_model(self, use_fp16=False):
        self.model = BaseEmbedder(model_name_or_path=self.config["model_name_or_path"],use_fp16=use_fp16, batch_size=self.config['infer_batch_size'], devices=self.config['infer_device'])

    def get_normal_session(self):
        if not self.model:
            self.load_model()
        return NormalSession(self.model)

    def get_onnx_session(self) -> ort.InferenceSession:
        onnx_model_path = self.config["onnx_model_path"]
        return ort.InferenceSession(onnx_model_path)

    def get_tensorrt_session(self) -> trt.ICudaEngine:
        engine_file_path=self.config['trt_model_path']
        if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
            print("Reading engine from file {}".format(engine_file_path))
            TRT_LOGGER = trt.Logger()
            with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:
            return self.build_trt_engine()


    @classmethod
    def convert_to_onnx(cls, model_name_or_path: str = None, onnx_model_path: str = None, opset_version = 14):
        # model = AutoModel.from_pretrained(model_name_or_path)
        # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        print(model_name_or_path)
        # pdb.set_trace()
        base_model=BaseEmbedder(model_name_or_path=model_name_or_path)
        model=base_model.model
        tokenizer=base_model.tokenizer
        dummy_input = tokenizer("This is a dummy input", return_tensors="pt")

        torch.onnx.export(
            model,  
            (dummy_input['input_ids'],),  
            onnx_model_path,  
            opset_version=opset_version,  # ONNX opset 版本
            input_names=['input_ids'],  
            output_names=['output'],  
            dynamic_axes={'input_ids': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  
        )
        print(f"Model has been converted to ONNX and saved at {onnx_model_path}")

    def inference(
        self,
        inputs: Union[List[str], List[Tuple[str, str]], pd.DataFrame, Any],
        *args,
        **kwargs
    ):
        """
        Perform inference using the specified session type.

        Args:
            inputs (Union[List[str], List[Tuple[str, str]], pd.DataFrame, Any]): Input data for inference.
            session_type (str): The type of session to use ('tensorrt', 'onnx', 'normal').
            session (Any): The session object to use for inference.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The inference results.
        """
        if self.session == None:
            raise ValueError('Please run get_inference_session first!')
        
        session_type=self.config['infer_mode']
        if session_type == 'tensorrt':
            return self._inference_tensorrt(inputs, *args, **kwargs)
        elif session_type == 'onnx':
            return self._inference_onnx(inputs, *args, **kwargs)
        elif session_type == 'normal':
            return self._inference_normal(inputs, *args, **kwargs)
        else:
            raise ValueError(f"Unsupported session type: {session_type}")

    def _inference_tensorrt(self, inputs, *args, **kwargs):
        tokenizer = self.model.tokenizer
        encoded_inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
        input_ids = encoded_inputs['input_ids'].cuda().cpu().numpy().astype(np.float32) 
        TRT_LOGGER = trt.Logger()
        with self.session.create_execution_context() as context:
            # Allocate buffers
            input_shape = input_ids.shape
            input_size = trt.volume(input_shape) * trt.float32.itemsize
            output_shape=tuple([input_shape[0], input_shape[1], 768])
            output_size = trt.volume(output_shape) * trt.float32.itemsize

            d_input = cuda.mem_alloc(input_size)
            d_output = cuda.mem_alloc(output_size)

            bindings = [int(d_input), int(d_output)]
            input_binding_name = self.session.get_binding_name(0)  # 获取输入绑定名称
            context.set_input_shape(input_binding_name, input_shape)
            # Create a stream in which to copy inputs/outputs and run inference.
            stream = cuda.Stream()

            # Transfer input data to the GPU.
            cuda.memcpy_htod_async(d_input, input_ids, stream)

            # Run inference.
            context.execute_async_v3(stream_handle=stream.handle)

            # Transfer predictions back from the GPU.
            output_data = np.empty(output_shape, dtype=np.float32)
            cuda.memcpy_dtoh_async(output_data, d_output, stream)

            # Synchronize the stream
            stream.synchronize()

            return output_data

    def _inference_onnx(self, inputs, *args, **kwargs):
        tokenizer = self.model.tokenizer
        encoded_inputs = tokenizer(inputs, return_tensors="np", padding=True, truncation=True)
        input_ids = encoded_inputs['input_ids']

        input_feed = {self.session.get_inputs()[0].name: input_ids}

        outputs = self.session.run(None, input_feed)
        embeddings = outputs[0] # (1, 9, 768)
        cls_emb=embeddings[:, 0, :]
        cls_emb=cls_emb.squeeze()
        
        if kwargs.get('normalize', False) == True:
            norm = np.linalg.norm(cls_emb)
            normalized_cls_emb = cls_emb / norm
            return normalized_cls_emb
        
        return cls_emb

    def _inference_normal(self, inputs, *args, **kwargs):
        input_feed = {self.session.get_inputs()[0].name: inputs}

        outputs = self.session.run([self.session.get_outputs()[0].name], input_feed)
        embeddings = outputs[0]
        return embeddings
    
    def build_trt_engine(self, onnx_model_path: str = None, trt_model_path : str = None):
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        TRT_LOGGER = trt.Logger()
        onnx_file_path=self.config['onnx_model_path'] if onnx_model_path==None else onnx_model_path
        engine_file_path= self.config['trt_model_path'] if trt_model_path == None else trt_model_path
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            EXPLICIT_BATCH
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser, trt.Runtime(
            TRT_LOGGER
        ) as runtime:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)
            
            # Define optimization profile
            profile = builder.create_optimization_profile()
            profile.set_shape("input_ids", (1, 9), (1, 9), (1, 9))  # 修改为与输入张量一致的维度
            config.add_optimization_profile(profile)
            
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print(
                    "ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.".format(onnx_file_path)
                )
                exit(0)
            print("Loading ONNX file from path {}...".format(onnx_file_path))
            with open(onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors()):
                        print(parser.get_error(error))
                    return None

            print("Completed parsing of ONNX file")
            print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            if plan is None:
                print("Failed to build the serialized network!")
                return None
            engine = runtime.deserialize_cuda_engine(plan)
            if engine is None:
                print("Failed to deserialize the CUDA engine!")
                return None
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine


if __name__=='__main__':
    import pdb
    # sentences_1 = ["样例数据-1", "样例数据-2"]
    # sentences_2 = ["样例数据-3", "样例数据-4"]
    model = BaseEmbedder(model_name_or_path='/data2/OpenLLMs/bge-base-zh-v1.5', query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：", use_fp16=True, devices=['cuda:1','cuda:0']) # Setting use_fp16 to True speeds up computation with a slight performance degradation
    # pdb.set_trace()
    # embeddings_1 = model.encode(sentences_1)
    # embeddings_2 = model.encode(sentences_2)
    # similarity = embeddings_1 @ embeddings_2.T
    # print(similarity)

    # # for s2p(short query to long passage) retrieval task, suggest to use encode_queries() which will automatically add the instruction to each query
    # # corpus in retrieval task can still use encode_corpus(), since they don't need instruction
    # queries = ['query_1', 'query_2']
    # passages = ["样例文档-1", "样例文档-2"]
    # q_embeddings = model.encode_query(queries)
    # p_embeddings = model.encode_info(passages)
    # scores = q_embeddings @ p_embeddings.T
    # print(scores)
    
    # del model
    
    """
    below test BaseEmbedderInferenceEngine
    """
    # 1. convert model to onnx
    model_path='/data2/OpenLLMs/bge-base-zh-v1.5'
    args=AbsInferenceArguments(
        model_name_or_path=model_path,
        onnx_model_path='/data2/OpenLLMs/bge-base-zh-v1.5/onnx/model.onnx',
        trt_model_path='/data2/OpenLLMs/bge-base-zh-v1.5/trt/model.trt',
        infer_mode='tensorrt',
        infer_device='cuda:0',
        infer_batch_size=16
    )
    
    
    
    # print('test convert_to_onnx')
    # BaseEmbedderInferenceEngine.convert_to_onnx(model_name_or_path=model_path, onnx_model_path=args.onnx_model_path)
    s='This is a sentence.'
    s2='Is this a sentence?'
    # # 2. test normal session
    args.infer_mode='normal'
    inference_engine=BaseEmbedderInferenceEngine(args)
    s_e=inference_engine.inference(s)
    s2_e=inference_engine.inference(s2)
    print(f'test normal: {s_e @ s2_e.T}')
    
    # 3. test onnx session
    args.infer_mode='onnx'
    inference_engine_onnx=BaseEmbedderInferenceEngine(args)
    s_e=inference_engine_onnx.inference(s, normalize=True)
    s2_e=inference_engine_onnx.inference(s2, normalize=True)
    print(f'test onnx: {s_e @ s2_e.T}')
    
    # 4. test tensorrt session
    args.infer_mode='tensorrt'
    inference_engine_tensorrt=BaseEmbedderInferenceEngine(args)
    s_e=inference_engine_tensorrt.inference(s)
    s2_e=inference_engine_tensorrt.inference(s2)
    # print(f'test tensorrt: {s_e @ s2_e.T}')
    print(s_e.shape)
    