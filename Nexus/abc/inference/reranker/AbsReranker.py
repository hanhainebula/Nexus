import logging
from abc import ABC, abstractmethod
from typing import Any, Union, List, Tuple, Dict, Literal, Optional

import multiprocessing as mp
from multiprocessing import Queue

import math
import gc
import torch
import numpy as np
from tqdm import tqdm, trange
from transformers import is_torch_npu_available

logger = logging.getLogger(__name__)


class AbsReranker(ABC):
    """
    Base class for Reranker.
    Extend this class and implement :meth:`compute_score_single_gpu` for custom rerankers.
    """

    def __init__(
        self,
        *args,
        **kwargs: Any,
    ):
        pass

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
            devices (Union[str, int, List[str], List[int]]): Specified devices, can be `str`, `int`, list of `str`, or list of `int`.

        Raises:
            ValueError: Devices should be a string or an integer or a list of strings or a list of integers.

        Returns:
            List[str]: A list of target devices in format
        """
        if devices is None:
            if torch.cuda.is_available():
                return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            elif is_torch_npu_available():
                return [f"npu:{i}" for i in range(torch.npu.device_count())]
            elif torch.backends.mps.is_available():
                return ["mps"]
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
    def compute_score(
        self,
        *args,
        **kwargs
    ):
        """Compute score for each sentence pair

        Returns:
            numpy.ndarray: scores of all the sentence pairs.
        """
        pass

    def __del__(self):
        self.stop_self_pool()

    @abstractmethod
    def compute_score_single_gpu(
        self,
        sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
        batch_size: int = 256,
        query_max_length: Optional[int] = None,
        max_length: int = 512,
        normalize: bool = False,
        device: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        This method should compute the scores of sentence_pair and return scores.
        """
        pass

    # copied from https://github.com/UKPLab/sentence-transformers/blob/1802076d4eae42ff0a5629e1b04e75785d4e193b/sentence_transformers/SentenceTransformer.py#L857
    def start_multi_process_pool(self) -> Dict[Literal["input", "output", "processes"], Any]:
        """
        Starts a multi-process pool to process the encoding with several independent processes
        via :meth:`SentenceTransformer.encode_multi_process <sentence_transformers.SentenceTransformer.encode_multi_process>`.

        This method is recommended if you want to encode on multiple GPUs or CPUs. It is advised
        to start only one process per GPU. This method works together with encode_multi_process
        and stop_multi_process_pool.

        Returns:
            Dict[str, Any]: A dictionary with the target processes, an input queue, and an output queue.
        """
        logger.info("Start multi-process pool on devices: {}".format(", ".join(map(str, self.target_devices))))

        self.model.to("cpu")
        self.model.share_memory()
        ctx = mp.get_context("spawn")
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for device_id in tqdm(self.target_devices, desc='initial target device'):
            p = ctx.Process(
                target=AbsReranker._encode_multi_process_worker,
                args=(device_id, self, input_queue, output_queue),
                daemon=True,
            )
            p.start()
            processes.append(p)

        return {"input": input_queue, "output": output_queue, "processes": processes}

    # copied from https://github.com/UKPLab/sentence-transformers/blob/1802076d4eae42ff0a5629e1b04e75785d4e193b/sentence_transformers/SentenceTransformer.py#L857

    # copied from https://github.com/UKPLab/sentence-transformers/blob/1802076d4eae42ff0a5629e1b04e75785d4e193b/sentence_transformers/SentenceTransformer.py#L857
    @staticmethod
    def _encode_multi_process_worker(
            target_device: str, model: 'AbsReranker', input_queue: Queue, results_queue: Queue
    ) -> None:
        """
        Internal working process to encode sentences in multi-process setup
        """
        while True:
            try:
                chunk_id, sentences, kwargs = (
                    input_queue.get()
                )
                embeddings = model.compute_score_single_gpu(
                    sentences,
                    device=target_device,
                    **kwargs
                )

                results_queue.put([chunk_id, embeddings])
            except:
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
