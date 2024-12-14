import logging
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Any

import pandas as pd

import onnx
import onnxruntime as ort
import tensorrt as trt

from .arguments import AbsInferenceArguments

logger = logging.getLogger(__name__)

class InferenceEngine(ABC):
    def __init__(self, infer_args: AbsInferenceArguments):
        self.config = infer_args.to_dict()

    @abstractmethod
    def load_model(self):
        pass

    @staticmethod
    def load_onnx_model(onnx_model_path: Union[str, Path]):
        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)
        logger.info(f"====== Loaded ONNX model from {onnx_model_path} ======")
        logger.info(onnx.helper.printable_graph(onnx_model.graph))
        return onnx_model

    @abstractmethod
    def get_normal_session(self):
        pass

    @abstractmethod
    def get_onnx_session(self) -> ort.InferenceSession:
        pass

    @abstractmethod
    def get_tensorrt_session(self) -> trt.ICudaEngine:
        pass

    def get_inference_session(self):
        if self.config["infer_mode"] == "normal":
            return self.get_normal_session()
        if self.config["infer_mode"] == "onnx":
            return self.get_onnx_session()
        elif self.config["infer_mode"] == "tensorrt":
            return self.get_tensorrt_session()
        else:
            raise ValueError(f"Invalid inference mode: {self.config['infer_mode']}")

    @abstractmethod
    def convert_to_onnx(self):
        pass

    @abstractmethod
    def inference(
        self,
        inputs: Union[List[str], List[Tuple[str, str]], pd.DataFrame, Any],
        *args,
        **kwargs
    ):
        pass
