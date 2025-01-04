import os
import sys
import subprocess
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

    def run(self, output_names, input_feed, batch_size=None, encode_query=False, run_options=None):
            
        sentences=input_feed['sentences']    
        if encode_query:      
            embeddings = self.model.encode_query(sentences, batch_size=batch_size)
        else:
            embeddings = self.model.encode_info(sentences, batch_size=batch_size)
        
        if not isinstance(embeddings, list):
            embeddings = [embeddings]
            
        return embeddings
        
    


class BaseEmbedderInferenceEngine(InferenceEngine):
    def __init__(self, infer_args: AbsInferenceArguments, model:BaseEmbedder=None):
        super().__init__(infer_args)
        # normal model
        if not model:
            self.load_model()
        else:
            self.model=model
        # session
        self.batch_size= self.config['infer_batch_size']
        self.session = self.get_inference_session()
        
    def load_model(self, use_fp16=False):
        self.model = BaseEmbedder(model_name_or_path=self.config["model_name_or_path"],use_fp16=use_fp16, batch_size=self.config['infer_batch_size'], devices=self.config['infer_device'])

    def get_normal_session(self):
        if not self.model:
            self.load_model()
        return NormalSession(self.model)

    def get_ort_session(self) -> ort.InferenceSession:
        providers = ['CUDAExecutionProvider']
        if self.config['infer_device'] == 'cpu':
            providers = ["CPUExecutionProvider"]
        elif isinstance(self.config['infer_device'], int):
            providers = [("CUDAExecutionProvider", {"device_id": self.config['infer_device']})]
            
        onnx_model_path = self.config["onnx_model_path"]
        return ort.InferenceSession(onnx_model_path, providers=providers)

    def get_trt_session(self) -> trt.ICudaEngine:
        device=self.config['infer_device']
        if not isinstance(device, int):
            device=0
        # cuda.Device(device).make_context()
        engine_file_path=self.config['trt_model_path']
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())


    @classmethod
    def convert_to_onnx(cls, model_name_or_path: str = None, onnx_model_path: str = None, opset_version = 14, use_fp16=False):
        # model = AutoModel.from_pretrained(model_name_or_path)
        # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        print(model_name_or_path)
        os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)
        # pdb.set_trace()
        base_model=BaseEmbedder(model_name_or_path=model_name_or_path, use_fp16=False)
        model=base_model.model
        tokenizer=base_model.tokenizer
        dummy_input = tokenizer("This is a dummy input", return_tensors="pt", padding=True)
        dummy_input = (torch.LongTensor(dummy_input['input_ids']).view(1, -1), torch.LongTensor(dummy_input['attention_mask']).view(1, -1), torch.LongTensor(dummy_input['token_type_ids']).view(1, -1))
        print(dummy_input[0].shape)
        if use_fp16:
            model = model.half()  # 将模型权重转换为 FP16
            dummy_input = {key: value.half() for key, value in dummy_input.items()}  # 将输入数据转换为 FP16

        torch.onnx.export(
            model,  
            dummy_input,  
            onnx_model_path,  
            opset_version=opset_version,  # ONNX opset 版本
            input_names=['input_ids', 'attention_mask', 'token_type_ids'],  
            output_names=['output'],  
            dynamic_axes={'input_ids': {0: 'batch_size', 1: 'token_length'},'token_type_ids': {0: 'batch_size', 1: 'token_length'},'attention_mask': {0: 'batch_size', 1: 'token_length'}, 'output':{0: 'batch_size', 1: 'token_length'}}  
        )
        print(f"Model has been converted to ONNX and saved at {onnx_model_path}")

    def convert_to_tensorrt(self, onnx_model_path: str= None, trt_model_path: str = None):
        # use trtexec
        if not onnx_model_path or not trt_model_path:
            onnx_model_path = self.config['onnx_model_path']
            trt_model_path = self.config['trt_model_path']
            
        if not os.path.isfile(onnx_model_path):
            raise FileNotFoundError(f"ONNX model not exists: {onnx_model_path}")
        
        trt_save_dir = os.path.dirname(trt_model_path)
        if not os.path.exists(trt_save_dir):
            os.makedirs(trt_save_dir)

        trtexec_cmd = (
            f"trtexec --onnx={onnx_model_path} --saveEngine={trt_model_path} "
            f"--minShapes=input_ids:1x1,attention_mask:1x1,token_type_ids:1x1 "
            f"--optShapes=input_ids:{self.batch_size}x512,attention_mask:{self.batch_size}x512,token_type_ids:{self.batch_size}x512 "
            f"--maxShapes=input_ids:{self.batch_size}x512,attention_mask:{self.batch_size}x512,token_type_ids:{self.batch_size}x512 "
            "--verbose"
        )

        try:
            result = subprocess.run(trtexec_cmd, check=True, capture_output=True, text=True)
            print("DONE!")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print("FAIL!")
            print(f"ERROR MESSAGES: {e.stderr}")
            sys.exit(1)

    def inference(
        self,
        inputs: Union[List[str], List[Tuple[str, str]], pd.DataFrame, Any],
        encode_query=False,
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
            return self._inference_normal(inputs, encode_query=encode_query,*args, **kwargs)
        else:
            raise ValueError(f"Unsupported session type: {session_type}")

    def _inference_tensorrt(self, inputs, normalize=True, batch_size=None, *args, **kwargs):
        # prepaer inputs first
        
        if not batch_size:
            batch_size = self.batch_size
        
        if isinstance(inputs, str):
            inputs=[inputs]
            
        tokenizer=self.model.tokenizer        
        engine=self.session
        all_outputs=[]

        for idx in trange(0, len(inputs), batch_size, desc='Batch Inference'):
            batch_inputs=inputs[idx: idx+batch_size]
            
            encoded_inputs= tokenizer(batch_inputs, return_tensors="np", padding=True, truncation=True, max_length=512)
            inputs_feed={
                'input_ids':encoded_inputs['input_ids'], #(bs, max_length)
                'attention_mask':encoded_inputs['attention_mask'],
                'token_type_ids':encoded_inputs['token_type_ids']
            }
            with engine.create_execution_context() as context:
                stream = cuda.Stream()
                bindings = [0] * engine.num_io_tensors

                input_memory = []
                output_buffers = {}
                for i in range(engine.num_io_tensors):
                    tensor_name = engine.get_tensor_name(i)
                    dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
                    if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                        if -1 in tuple(engine.get_tensor_shape(tensor_name)):  # dynamic
                            # context.set_input_shape(tensor_name, tuple(engine.get_tensor_profile_shape(tensor_name, 0)[2]))
                            context.set_input_shape(tensor_name, tuple(inputs_feed[tensor_name].shape))
                        input_mem = cuda.mem_alloc(inputs_feed[tensor_name].nbytes)
                        bindings[i] = int(input_mem)
                        context.set_tensor_address(tensor_name, int(input_mem))
                        cuda.memcpy_htod_async(input_mem, inputs_feed[tensor_name], stream)
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
            
            output=output_buffers['output'][0]
            cls_output=output[:, 0, :].squeeze()
            all_outputs.extend(cls_output)
            
        if normalize:
            all_outputs= all_outputs / np.linalg.norm(all_outputs, axis = -1, keepdims = True)
        
        return all_outputs

    def _inference_onnx(self, inputs, normalize = True, batch_size = None, *args, **kwargs):
        if not batch_size:
            batch_size = self.batch_size
            
        if isinstance(inputs, str):
            inputs=[inputs]
            
        tokenizer = self.model.tokenizer
        all_outputs=[]
        for i in trange(0, len(inputs), batch_size, desc='Batch Inference'):
            batch_inputs= inputs[i:i+batch_size]
            encoded_inputs = tokenizer(batch_inputs, return_tensors="np", padding=True,  truncation=True, max_length=512)
            # input_ids = encoded_inputs['input_ids']
            input_feed={
                'input_ids':encoded_inputs['input_ids'], #(bs, max_length)
                'attention_mask':encoded_inputs['attention_mask'],
                'token_type_ids':encoded_inputs['token_type_ids']
            }

            outputs = self.session.run(None, input_feed)
            embeddings = outputs[0] # (1, 9, 768)
            cls_emb=embeddings[:, 0, :]
            cls_emb=cls_emb.squeeze()
            all_outputs.extend(cls_emb)
        
        if normalize == True:
            all_outputs = all_outputs / np.linalg.norm(all_outputs, axis=-1, keepdims=True)
            return all_outputs
        
        return cls_emb

    def _inference_normal(self, inputs,batch_size = None, encode_query = False, *args, **kwargs):
        if not batch_size:
            batch_size = self.batch_size
            
        input_feed = {self.session.get_inputs()[0].name: inputs}

        outputs = self.session.run([self.session.get_outputs()[0].name], input_feed, encode_query = encode_query ,batch_size = batch_size)
        
        embeddings = outputs[0]
        return embeddings

    def encode_query(self,
        inputs: Union[List[str], List[Tuple[str, str]], pd.DataFrame, Any],
        *args,
        **kwargs):
        
        return self.inference(inputs, encode_query=True ,*args, **kwargs)
    
    def encode_info(self,
        inputs: Union[List[str], List[Tuple[str, str]], pd.DataFrame, Any],
        *args,
        **kwargs):
        return self.inference(inputs, *args, **kwargs)

if __name__=='__main__':
    import pdb
    # sentences_1 = ["样例数据-1", "样例数据-2"]
    # sentences_2 = ["样例数据-3", "样例数据-4"]
    # model = BaseEmbedder(model_name_or_path='/data2/OpenLLMs/bge-base-zh-v1.5', use_fp16=True, devices=['cuda:1','cuda:0']) # Setting use_fp16 to True speeds up computation with a slight performance degradation
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
    model_path='/data2/OpenLLMs/bge-base-zh-v1.5'
    args=AbsInferenceArguments(
        model_name_or_path=model_path,
        onnx_model_path='/data2/OpenLLMs/bge-base-zh-v1.5/onnx/model.onnx',
        trt_model_path='/data2/OpenLLMs/bge-base-zh-v1.5/trt/model.trt',
        infer_mode='tensorrt',
        infer_device=0,
        infer_batch_size=16
    )
    
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "The Eiffel Tower is located in Paris, France.",
        "Python is a popular programming language.",
        "The Great Wall of China is one of the Seven Wonders of the World.",
        "Space exploration has led to many scientific discoveries.",
        "Climate change is a pressing global issue.",
        "The Mona Lisa is a famous painting by Leonardo da Vinci.",
        "Electric cars are becoming more common.",
        "The human brain is an incredibly complex organ."
    ]
    # 1. convert model to onnx
    # BaseEmbedderInferenceEngine.convert_to_onnx(model_name_or_path=model_path, onnx_model_path=args.onnx_model_path)
    # 2. test normal session
    args.infer_mode='normal'
    inference_engine=BaseEmbedderInferenceEngine(args)
    s_e_norm=inference_engine.encode_query(sentences, batch_size=10, normalize=True)
    print(s_e_norm.shape)
    
    # 3. test onnx session
    args.infer_mode = 'onnx'
    inference_engine_onnx = BaseEmbedderInferenceEngine(args)
    s_e_onnx = inference_engine_onnx.encode_query(sentences, normalize=True)
    print(s_e_onnx.shape)
    
    # 4. test tensorrt session
    # args.infer_mode='tensorrt'
    # inference_engine_tensorrt=BaseEmbedderInferenceEngine(args)
    # s_e_trt=inference_engine_tensorrt.inference(sentences, normalize=True)
    # print(s_e_trt.shape)
    # cuda.Context.pop()
    # s2_e=inference_engine_tensorrt.inference(s2)
    # print(f'test tensorrt: {s_e @ s2_e.T}')
    