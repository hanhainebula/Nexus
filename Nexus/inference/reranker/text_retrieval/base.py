import os
import sys
import subprocess
import torch
import math
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from typing import Any, List, Union, Tuple, Optional,Dict, Literal
from dataclasses import field, dataclass
import pdb
import tensorrt as trt
import pycuda.driver as cuda
import onnx
import onnxruntime as ort

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from Nexus.abc.inference import AbsReranker, InferenceEngine, AbsInferenceArguments

def sigmoid(x):
    x = x.item() if not isinstance(x, float) else x
    return float(1 / (1 + np.exp(-x)))


class BaseReranker(AbsReranker):
    """Base reranker class for encoder only models.

    Args:
        model_name_or_path (str): If it's a path to a local model, it loads the model from the path. Otherwise tries to download and
            load a model from HuggingFace Hub with the name.
        use_fp16 (bool, optional): If true, use half-precision floating-point to speed up computation with a slight performance 
            degradation. Defaults to :data:`False`.
        query_instruction_for_rerank (Optional[str], optional): Query instruction for retrieval tasks, which will be used with
            with :attr:`query_instruction_format`. Defaults to :data:`None`.
        query_instruction_format (str, optional): The template for :attr:`query_instruction_for_rerank`. Defaults to :data:`"{}{}"`.
        passage_instruction_format (str, optional): The template for passage. Defaults to "{}{}".
        cache_dir (Optional[str], optional): Cache directory for the model. Defaults to :data:`None`.
        devices (Optional[Union[str, List[str], List[int]]], optional): Devices to use for model inference. Defaults to :data:`None`.
        batch_size (int, optional): Batch size for inference. Defaults to :data:`128`.
        query_max_length (Optional[int], optional): Maximum length for queries. If not specified, will be 3/4 of :attr:`max_length`.
            Defaults to :data:`None`.
        max_length (int, optional): Maximum length of passages. Defaults to :data`512`.
        normalize (bool, optional): If True, use Sigmoid to normalize the results. Defaults to :data:`False`.
    """
    def __init__(
        self,
        model_name_or_path: str,
        use_fp16: bool = False,
        query_instruction_for_rerank: Optional[str] = None,
        query_instruction_format: str = "{}{}", # specify the format of query_instruction_for_rerank
        passage_instruction_for_rerank: Optional[str] = None,
        passage_instruction_format: str = "{}{}", # specify the format of passage_instruction_for_rerank
        trust_remote_code: bool = False,
        cache_dir: Optional[str] = None,
        devices: Optional[Union[str, List[str], List[int]]] = None, # specify devices, such as ["cuda:0"] or ["0"]
        # inference
        batch_size: int = 128,
        query_max_length: Optional[int] = None,
        max_length: int = 512,
        normalize: bool = False,
        **kwargs: Any,
    ):  
        import pycuda.autoinit
        self.model_name_or_path = model_name_or_path
        self.use_fp16 = use_fp16
        self.query_instruction_for_rerank = query_instruction_for_rerank
        self.query_instruction_format = query_instruction_format
        self.passage_instruction_for_rerank = passage_instruction_for_rerank
        self.passage_instruction_format = passage_instruction_format
        self.target_devices = self.get_target_devices(devices)
        
        self.batch_size = batch_size
        self.query_max_length = query_max_length
        self.max_length = max_length
        self.normalize = normalize

        for k in kwargs:
            setattr(self, k, kwargs[k])

        self.kwargs = kwargs

        self.pool = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, 
            cache_dir=cache_dir
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, 
            trust_remote_code=trust_remote_code, 
            cache_dir=cache_dir
        )

    def get_detailed_instruct(self, instruction_format: str, instruction: str, sentence: str):
        """Combine the instruction and sentence along with the instruction format.

        Args:
            instruction_format (str): Format for instruction.
            instruction (str): The text of instruction.
            sentence (str): The sentence to concatenate with.

        Returns:
            str: The complete sentence with instruction
        """
        return instruction_format.format(instruction, sentence)
    
    def get_detailed_inputs(self, sentence_pairs: Union[str, List[str]]):
        """get detailed instruct for all the inputs

        Args:
            sentence_pairs (Union[str, List[str]]): Input sentence pairs

        Returns:
            list[list[str]]: The complete sentence pairs with instruction
        """
        if isinstance(sentence_pairs, str):
            sentence_pairs = [sentence_pairs]

        if self.query_instruction_for_rerank is not None:
            if self.passage_instruction_for_rerank is None:
                return [
                    [
                        self.get_detailed_instruct(self.query_instruction_format, self.query_instruction_for_rerank, sentence_pair[0]),
                        sentence_pair[1]
                    ] for sentence_pair in sentence_pairs
                ]
            else:
                return [
                    [
                        self.get_detailed_instruct(self.query_instruction_format, self.query_instruction_for_rerank, sentence_pair[0]),
                        self.get_detailed_instruct(self.passage_instruction_format, self.passage_instruction_for_rerank, sentence_pair[1])
                    ] for sentence_pair in sentence_pairs
                ]
        else:
            if self.passage_instruction_for_rerank is None:
                return [
                    [
                        sentence_pair[0],
                        sentence_pair[1]
                    ] for sentence_pair in sentence_pairs
                ]
            else:
                return [
                    [
                        sentence_pair[0],
                        self.get_detailed_instruct(self.passage_instruction_format, self.passage_instruction_for_rerank, sentence_pair[1])
                    ] for sentence_pair in sentence_pairs
                ]

    def compute_score(
        self,
        sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
        **kwargs
    ):
        """Compute score for each sentence pair

        Args:
            sentence_pairs (Union[List[Tuple[str, str]], Tuple[str, str]]): Input sentence pairs to compute.

        Returns:
            numpy.ndarray: scores of all the sentence pairs.
        """
        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]
        sentence_pairs = self.get_detailed_inputs(sentence_pairs) # list[list[str]]

        if isinstance(sentence_pairs, str) or len(self.target_devices) == 1:
            return self.compute_score_single_gpu(
                sentence_pairs,
                device=self.target_devices[0],
                **kwargs
            )

        if self.pool is None:
            self.pool = self.start_multi_process_pool()
        scores = self.encode_multi_process(sentence_pairs,
                                           self.pool,
                                           **kwargs)
        return scores


    @torch.no_grad()
    def compute_score_single_gpu(
        self,
        sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
        batch_size: Optional[int] = None,
        query_max_length: Optional[int] = None,
        max_length: Optional[int] = None,
        normalize: Optional[bool] = None,
        device: Optional[str] = None,
        **kwargs: Any
    ) -> List[float]:
        """_summary_

        Args:
            sentence_pairs (Union[List[Tuple[str, str]], Tuple[str, str]]): Input sentence pairs to compute scores.
            batch_size (Optional[int], optional): Number of inputs for each iter. Defaults to :data:`None`.
            query_max_length (Optional[int], optional): Maximum length of tokens of queries. Defaults to :data:`None`.
            max_length (Optional[int], optional): Maximum length of tokens. Defaults to :data:`None`.
            normalize (Optional[bool], optional): If True, use Sigmoid to normalize the results. Defaults to :data:`None`.
            device (Optional[str], optional): Device to use for computation. Defaults to :data:`None`.

        Returns:
            List[float]: Computed scores of queries and passages.
        """
        if batch_size is None: batch_size = self.batch_size
        if max_length is None: max_length = self.max_length
        if query_max_length is None:
            if self.query_max_length is not None:
                query_max_length = self.query_max_length
            else:
                query_max_length = max_length * 3 // 4
        if normalize is None: normalize = self.normalize

        if device is None:
            device = self.target_devices[0]

        if device == "cpu": self.use_fp16 = False
        if self.use_fp16: self.model.half()

        self.model.to(device)
        self.model.eval()

        assert isinstance(sentence_pairs, list)
        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]
        
        # tokenize without padding to get the correct length
        all_inputs = []
        for start_index in trange(0, len(sentence_pairs), batch_size, desc="pre tokenize",
                                  disable=len(sentence_pairs) < 128):
            sentences_batch = sentence_pairs[start_index:start_index + batch_size]
            queries = [s[0] for s in sentences_batch]
            passages = [s[1] for s in sentences_batch]
            queries_inputs_batch = self.tokenizer(
                queries,
                return_tensors=None,
                add_special_tokens=False,
                max_length=query_max_length,
                truncation=True,
                **kwargs
            )['input_ids']
            passages_inputs_batch = self.tokenizer(
                passages,
                return_tensors=None,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
                **kwargs
            )['input_ids']
            for q_inp, d_inp in zip(queries_inputs_batch, passages_inputs_batch):
                item = self.tokenizer.prepare_for_model(
                    q_inp,
                    d_inp,
                    truncation='only_second',
                    max_length=max_length,
                    padding=False,
                )
                all_inputs.append(item)
        # sort by length for less padding
        length_sorted_idx = np.argsort([-len(x['input_ids']) for x in all_inputs])
        all_inputs_sorted = [all_inputs[i] for i in length_sorted_idx]

        # adjust batch size
        flag = False
        while flag is False:
            try:
                test_inputs_batch = self.tokenizer.pad(
                    all_inputs_sorted[:min(len(all_inputs_sorted), batch_size)],
                    padding=True,
                    return_tensors='pt',
                    **kwargs
                ).to(device)
                scores = self.model(**test_inputs_batch, return_dict=True).logits.view(-1, ).float()
                flag = True
            except RuntimeError as e:
                batch_size = batch_size * 3 // 4
            except torch.OutofMemoryError as e:
                batch_size = batch_size * 3 // 4

        all_scores = []
        for start_index in tqdm(range(0, len(all_inputs_sorted), batch_size), desc="Compute Scores",
                                disable=len(all_inputs_sorted) < 128):
            sentences_batch = all_inputs_sorted[start_index:start_index + batch_size]
            inputs = self.tokenizer.pad(
                sentences_batch,
                padding=True,
                return_tensors='pt',
                **kwargs
            ).to(device)

            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
            all_scores.extend(scores.cpu().numpy().tolist())

        all_scores = [all_scores[idx] for idx in np.argsort(length_sorted_idx)]

        if normalize:
            all_scores = [sigmoid(score) for score in all_scores]

        return all_scores

    def encode_multi_process(
        self,
        sentence_pairs: List,
        pool: Dict[Literal["input", "output", "processes"], Any],
        **kwargs
    ) -> np.ndarray:
        chunk_size = math.ceil(len(sentence_pairs) / len(pool["processes"]))

        input_queue = pool["input"]
        last_chunk_id = 0
        chunk = []

        for sentence_pair in sentence_pairs:
            chunk.append(sentence_pair)
            if len(chunk) >= chunk_size:
                input_queue.put(
                    [last_chunk_id, chunk, kwargs]
                )
                last_chunk_id += 1
                chunk = []

        if len(chunk) > 0:
            input_queue.put([last_chunk_id, chunk, kwargs])
            last_chunk_id += 1

        output_queue = pool["output"]
        results_list = sorted(
            [output_queue.get() for _ in trange(last_chunk_id, desc="Chunks")],
            key=lambda x: x[0],
        )
        scores = np.concatenate([result[1] for result in results_list])
        return scores

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
            name='sentence_pairs'
        )]
        
    def get_outputs(self):
        return [SessionParams(
            name='score'
        )]

    def run(self, output_names, input_feed, normalize=True, batch_size=10, run_options=None):
        sentence_pairs=input_feed['sentence_pairs']            
        score = self.model.compute_score(sentence_pairs, normalize=normalize, batch_size=batch_size)
        if not isinstance(score, list):
            score = [score]    
        return score
        
    


class BaseRerankerInferenceEngine(InferenceEngine):
    def __init__(self, infer_args: AbsInferenceArguments, model: BaseReranker = None):
        super().__init__(infer_args)
        # normal model
        import pycuda.autoinit
        if not model:
            self.load_model()
        else:
            self.model=model
        # session
        self.device = self.config['infer_device']
        self.batch_size = self.config['infer_batch_size']
        self.session = self.get_inference_session()
        
    def load_model(self, use_fp16=False):
        self.model = BaseReranker(model_name_or_path=self.config["model_name_or_path"],use_fp16=use_fp16, batch_size=self.config['infer_batch_size'], devices=self.config['infer_device'])

    def get_normal_session(self):
        if not self.model:
            self.load_model()
        return NormalSession(self.model)

    def get_ort_session(self) -> ort.InferenceSession:
        if self.config['infer_device'] == 'cpu':
            providers = ["CPUExecutionProvider"]
        elif isinstance(self.config['infer_device'], int):
            providers = ["CPUExecutionProvider", ("CUDAExecutionProvider", {"device_id": self.config['infer_device']})]
        else:
            providers = ['CUDAExecutionProvider', "CPUExecutionProvider"]
        onnx_model_path = self.config["onnx_model_path"]
        return ort.InferenceSession(onnx_model_path, providers=providers)

    def get_trt_session(self) -> trt.ICudaEngine:
        device=self.config['infer_device']
        if not isinstance(device, int):
            device=0
        cuda.Device(device).make_context()
        engine_file_path=self.config['trt_model_path']
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())


    @classmethod
    def convert_to_onnx(cls, model_name_or_path: str = None, onnx_model_path: str = None, opset_version = 14, use_fp16=False):

        print(model_name_or_path)
        os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)
        base_model=BaseReranker(model_name_or_path=model_name_or_path, use_fp16=False)
        model=base_model.model
        tokenizer=base_model.tokenizer

        queries = "What is the capital of France?"
        passages = "Paris is the capital and most populous city of France."
        dummy_input= tokenizer(queries, passages, padding=True , return_tensors='pt', truncation='only_second', max_length=512)

        dummy_input = (torch.LongTensor(dummy_input['input_ids']).view(1, -1), torch.LongTensor(dummy_input['attention_mask']).view(1, -1))
            
        torch.onnx.export(
            model,  
            dummy_input,  
            onnx_model_path,  
            opset_version=opset_version,  # ONNX opset 版本
            input_names=['input_ids', 'attention_mask'],  
            output_names=['output'],  
            dynamic_axes={'input_ids': {0: 'batch_size', 1:'token_length'}, 'output': {0: 'batch_size'}, 'attention_mask': {0: 'batch_size', 1: 'token_length'}}
        )
        
        if use_fp16:
            from onnxconverter_common.float16 import convert_float_to_float16
            import copy
            model_fp32 = onnx.load(onnx_model_path)
            model_fp16 = convert_float_to_float16(copy.deepcopy(model_fp32))
            onnx.save(model_fp16, onnx_model_path)
        
        print(f"Model has been converted to ONNX and saved at {onnx_model_path}")
        

    
    @classmethod
    def convert_to_tensorrt(cls, onnx_model_path: str = None, trt_model_path: str = None, batch_size=16, trt_path: str = None):
        if not os.path.isfile(onnx_model_path):
            raise FileNotFoundError(f"ONNX model not exists: {onnx_model_path}")
        
        trt_save_dir = os.path.dirname(trt_model_path)
        if not os.path.exists(trt_save_dir):
            os.makedirs(trt_save_dir)

        if trt_path:
            os.environ['LD_LIBRARY_PATH'] = f"{os.environ.get('LD_LIBRARY_PATH', '')}:{trt_path}/lib"
            os.environ['PATH'] = f"{os.environ.get('PATH', '')}:{trt_path}/bin"

        trtexec_cmd = (
            f"trtexec --onnx={onnx_model_path} --saveEngine={trt_model_path} "
            f"--minShapes=input_ids:1x1,attention_mask:1x1 "
            f"--optShapes=input_ids:{batch_size}x512,attention_mask:{batch_size}x512 "
            f"--maxShapes=input_ids:{batch_size}x512,attention_mask:{batch_size}x512 "
            "--verbose "
            "--fp16"
        )

        # Run trtexec command
        try:
            result = subprocess.run(trtexec_cmd, shell=True, check=True, capture_output=True, text=True)
            print("DONE!")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print("FAIL!")
            print(f"ERROR MESSAGES: {e.stderr}")
            sys.exit(1)

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

    def _inference_tensorrt(self, sentence_pairs, batch_size = None, normalize = True, *args, **kwargs):
        
        if not batch_size:
            batch_size = self.batch_size
        
        
        if not isinstance(sentence_pairs, list):
            sentence_pairs=[sentence_pairs]

        self.tokenizer = self.model.tokenizer
        engine=self.session
        queries= [s[0] for s in sentence_pairs]
        passages= [s[1] for s in sentence_pairs]
        all_scores=[]
        for i in trange(0, len(sentence_pairs), batch_size, desc='Batch encoding'):
            batch_queries=queries[i:i+batch_size]
            batch_passages=passages[i:i+batch_size]
                     
            encoded_inputs=self.tokenizer(batch_queries, batch_passages, padding=True , return_tensors='np', truncation='only_second', max_length=512)
            inputs={
                'input_ids':encoded_inputs['input_ids'], #(bs, max_length)
                'attention_mask':encoded_inputs['attention_mask']
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
                            context.set_input_shape(tensor_name, tuple(inputs[tensor_name].shape))
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
            
            scores=output_buffers['output'][0] # (bs, 1)
            all_scores.extend(scores)
            
        if normalize:
            all_scores = [sigmoid(s) for s in all_scores]

        all_scores = [float(score) for score in all_scores]
        return all_scores
    
    def _inference_onnx(self, sentence_pairs, batch_size = None, normalize = True,  *args, **kwargs):
        
        if not batch_size:
            batch_size = self.batch_size
        
        self.tokenizer=self.model.tokenizer   
        if not isinstance(sentence_pairs, list):
            sentence_pairs=[sentence_pairs]
        queries= [s[0] for s in sentence_pairs]
        passages= [s[1] for s in sentence_pairs]
        
        all_scores=[]
        for i in trange(0, len(sentence_pairs), batch_size, desc='Batch encoding'):
            batch_queries = queries[i:i+batch_size]
            batch_passages = passages[i:i+batch_size]
            encoded_inputs=self.tokenizer(batch_queries, batch_passages, padding=True , return_tensors='np', truncation='only_second', max_length=512)
            input_feed={
                'input_ids':encoded_inputs['input_ids'], #(bs, max_length)
                'attention_mask':encoded_inputs['attention_mask']
                }
            batch_all_scores = self.session.run(None, input_feed)[0]
            all_scores.extend(batch_all_scores)

        
        if normalize:
            all_scores = [sigmoid(score[0]) for score in all_scores]

        all_scores = [float(score) for score in all_scores]

        return all_scores

    def _inference_normal(self, inputs, normalize=True, batch_size=None, *args, **kwargs):
        input_feed = {self.session.get_inputs()[0].name: inputs}

        if not batch_size:
            batch_size = self.batch_size

        outputs = self.session.run([self.session.get_outputs()[0].name], batch_size = batch_size, normalize = normalize,input_feed=input_feed)
        scores = outputs
        return scores
    
    def compute_score(self, inputs, *args, **kwargs):
        return self.inference(inputs, *args, **kwargs)


if __name__=='__main__':
    from Nexus import AbsInferenceArguments, BaseRerankerInferenceEngine

    # trt path is path to TensorRT you have downloaded.
    trt_path='/root/TensorRT-10.7.0.23'

    model_path='/root/models/bge-reranker-base'
    trt_model_path ='/root/models/bge-reranker-base/trt/model_fp16.trt'
    onnx_model_path='/root/models/bge-reranker-base/onnx/model.onnx'

    qa_pairs = [
        ("What is the capital of France?", "Paris is the capital and most populous city of France."),
        ("Who wrote 'Pride and Prejudice'?","Edison wrote this." ),
        ("What is the largest planet in our solar system?", "May be our mother land."),
        ("Who is the current president of the United States?", "The current president of the United States is Joe Biden."),
        ("What is the chemical symbol for water?", "The chemical symbol for water is H2O."),
        ("What is the speed of light?", "The speed of light is approximately 299,792 kilometers per second."),
        ("Who painted the Mona Lisa?", "The Mona Lisa was painted by Leonardo da Vinci."),
        ("What is the tallest mountain in the world?", "Mount Everest is the tallest mountain in the world."),
        ("What is the smallest country in the world?", "Vatican City is the smallest country in the world."),
        ("Who discovered penicillin?", "Penicillin was discovered by Alexander Fleming.")
    ]

    args=AbsInferenceArguments(
        model_name_or_path=model_path,
        onnx_model_path=onnx_model_path,
        trt_model_path=trt_model_path,
        infer_mode='tensorrt',
        infer_device=0,
        infer_batch_size=48
    )

    # BaseRerankerInferenceEngine.convert_to_tensorrt(args.onnx_model_path, args.trt_model_path, args.infer_batch_size, trt_path)

    inference_engine_tensorrt = BaseRerankerInferenceEngine(args)

    score = inference_engine_tensorrt.inference(qa_pairs, normalize=True, batch_size=5)
    print(score)