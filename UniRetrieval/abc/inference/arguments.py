from dataclasses import dataclass, field
from typing import List, Union

from UniRetrieval.abc.arguments import AbsArguments


@dataclass
class AbsInferenceArguments(AbsArguments):
    stage: str = field(
        default=None,
        metadata={"help": "Stage of the inference process."}
    )
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Name or path to the model."}
    )
    output_topk: int = field(
        default=10,
        metadata={"help": "Number of top-k results to output."}
    )
    infer_device: Union[str, int, List[int], List[str]] = field(
        default="cpu",
        metadata={"help": "Device to perform inference.", "nargs": "+"}
    )
    infer_mode: str = field(
        default="normal",
        metadata={"help": "Inference mode: normal, onnx, tensorrt.", "choices": ["normal", "onnx", "tensorrt"]}
    )
    infer_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for inference."}
    )
