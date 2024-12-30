import logging
from tqdm import tqdm, trange
from abc import ABC, abstractmethod
from typing import Any, Union, List, Dict, Literal, Optional

import queue
import multiprocessing as mp
from multiprocessing import Queue

import math
import gc
import torch
import numpy as np
from transformers import is_torch_npu_available

logger = logging.getLogger(__name__)


class AbsEmbedder(ABC):
    """
    Base class for embedder.
    Extend this class and implement :meth:`encode_queries`, :meth:`encode_corpus`, :meth:`encode` for custom embedders.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def stop_self_pool(self):
        if self.pool is not None:
            self.stop_multi_process_pool(self.pool)
            self.pool = None
        try:
            self.model.to('cpu')
            torch.cuda.empty_cache()
        except:
            pass
        gc.collect()

    @staticmethod
    def get_target_devices(devices: Union[str, int, List[str], List[int]]) -> List[str]:
        """

        Args:
            devices (Union[str, int, List[str], List[int]]): specified devices, can be `str`, `int`, list of `str`, or list of `int`.

        Raises:
            ValueError: Devices should be a string or an integer or a list of strings or a list of integers.

        Returns:
            List[str]: A list of target devices in format.
        """
        if devices is None:
            if torch.cuda.is_available():
                return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            elif is_torch_npu_available():
                return [f"npu:{i}" for i in range(torch.npu.device_count())]
            elif torch.backends.mps.is_available():
                return [f"mps:{i}" for i in range(torch.mps.device_count())]
            else:
                return ["cpu"]
        elif isinstance(devices, str):
            return [devices]
        elif isinstance(devices, int):
            return [f"cuda:{devices}"]
        elif isinstance(devices, list):
            if isinstance(devices[0], str):
                return devices
            elif isinstance(devices[0], int):
                return [f"cuda:{device}" for device in devices]
            else:
                raise ValueError("devices should be a string or an integer or a list of strings or a list of integers.")
        else:
            raise ValueError("devices should be a string or an integer or a list of strings or a list of integers.")

    @abstractmethod
    def encode_query(
        self,
        *args,
        **kwargs: Any
    ):
        """encode queries
        """
        return self.encode(*args, **kwargs)
    
    @abstractmethod
    def encode_info(
        self,
        *args,
        **kwargs: Any
    ):
        """
        encode info (corpus)
        """
        return self.encode(*args, **kwargs)


    @abstractmethod
    def encode(
        self,
        *args, **kwargs
    ):
        """encode
        """
        pass

        
    def __del__(self):
        self.stop_self_pool()

    @abstractmethod
    def encode_single_device(
        self,
        *args,
        **kwargs
    ):
        """
        This method should encode on a single device.
        """
        pass

    # adapted from https://github.com/UKPLab/sentence-transformers/blob/1802076d4eae42ff0a5629e1b04e75785d4e193b/sentence_transformers/SentenceTransformer.py#L807
    def start_multi_process_pool(
        self,
        process_target_func: Any,
    ) -> Dict[Literal["input", "output", "processes"], Any]:
        """
        Starts a multi-process pool to process the encoding with several independent processes
        via :meth:`SentenceTransformer.encode_multi_process <sentence_transformers.SentenceTransformer.encode_multi_process>`.

        This method is recommended if you want to encode on multiple GPUs or CPUs. It is advised
        to start only one process per GPU. This method works together with encode_multi_process
        and stop_multi_process_pool.

        Returns:
            Dict[str, Any]: A dictionary with the target processes, an input queue, and an output queue.
        """
        if self.model is None:
            raise ValueError("Model is not initialized.")

        logger.info("Start multi-process pool on devices: {}".format(", ".join(map(str, self.target_devices))))

        self.model.to("cpu")
        self.model.share_memory()
        ctx = mp.get_context("spawn")
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for device_id in tqdm(self.target_devices, desc='initial target device'):
            p = ctx.Process(
                target=process_target_func,
                args=(device_id, self, input_queue, output_queue),
                daemon=True,
            )
            p.start()
            processes.append(p)

        return {"input": input_queue, "output": output_queue, "processes": processes}

    # adapted from https://github.com/UKPLab/sentence-transformers/blob/1802076d4eae42ff0a5629e1b04e75785d4e193b/sentence_transformers/SentenceTransformer.py#L976
    @staticmethod
    def _encode_multi_process_worker(
        target_device: str, model: 'AbsEmbedder', input_queue: Queue, results_queue: Queue
    ) -> None:
        """
        Internal working process to encode sentences in multi-process setup
        """
        while True:
            try:
                chunk_id, sentences, kwargs = (
                    input_queue.get()
                )
                embeddings = model.encode_single_device(
                    sentences,
                    device=target_device,
                    **kwargs
                )

                results_queue.put([chunk_id, embeddings])
            except queue.Empty:
                break

    # copied from https://github.com/UKPLab/sentence-transformers/blob/1802076d4eae42ff0a5629e1b04e75785d4e193b/sentence_transformers/SentenceTransformer.py#L857
    @staticmethod
    def stop_multi_process_pool(pool: Dict[Literal["input", "output", "processes"], Any]) -> None:
        """
        Stops all processes started with start_multi_process_pool.

        Args:
            pool (Dict[str, object]): A dictionary containing the input queue, output queue, and process list.

        Returns:
            None
        """
        for p in pool["processes"]:
            p.terminate()

        for p in pool["processes"]:
            p.join()
            p.close()

        pool["input"].close()
        pool["output"].close()
        pool = None

    # adapted from https://github.com/UKPLab/sentence-transformers/blob/1802076d4eae42ff0a5629e1b04e75785d4e193b/sentence_transformers/SentenceTransformer.py#L877
    def encode_multi_process(
        self,
        sentences: List[str],
        pool: Dict[Literal["input", "output", "processes"], Any],
        **kwargs
    ):
        chunk_size = math.ceil(len(sentences) / len(pool["processes"]))

        input_queue = pool["input"]
        last_chunk_id = 0
        chunk = []

        for sentence in sentences:
            chunk.append(sentence)
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
        embeddings = self._concatenate_results_from_multi_process([result[1] for result in results_list])
        return embeddings

    def _concatenate_results_from_multi_process(self, results_list: List[Union[torch.Tensor, np.ndarray, Any]]):
        """concatenate and return the results from all the processes

        Args:
            results_list (List[Union[torch.Tensor, np.ndarray, Any]]): A list of results from all the processes.

        Raises:
            NotImplementedError: Unsupported type for results_list

        Returns:
            Union[torch.Tensor, np.ndarray]: return the embedding vectors in a numpy array or tensor.
        """
        if isinstance(results_list[0], torch.Tensor):
            return torch.cat(results_list, dim=0)
        elif isinstance(results_list[0], np.ndarray):
            return np.concatenate(results_list, axis=0)
        else:
            raise NotImplementedError("Unsupported type for results_list")

