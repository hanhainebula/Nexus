from abc import ABC, abstractmethod
import inspect
from typing import Optional

import torch

from .dataset import AbsDataset
from .modeling import AbsModel
from .arguments import AbsTrainingArguments
from transformers import Trainer


def _patch_accelerate_unwrap_model_signature():
    try:
        from accelerate import Accelerator
    except ImportError:
        return

    signature = inspect.signature(Accelerator.unwrap_model)
    if "keep_torch_compile" in signature.parameters:
        return
    if getattr(Accelerator.unwrap_model, "_nexus_keep_torch_compile_patch", False):
        return

    original_unwrap_model = Accelerator.unwrap_model

    def compatible_unwrap_model(self, model, keep_fp32_wrapper: bool = True, keep_torch_compile: bool = False):
        return original_unwrap_model(self, model, keep_fp32_wrapper=keep_fp32_wrapper)

    compatible_unwrap_model._nexus_keep_torch_compile_patch = True
    Accelerator.unwrap_model = compatible_unwrap_model


_patch_accelerate_unwrap_model_signature()

class AbsTrainer(Trainer):
    @abstractmethod
    def _save(self, output_dir: Optional[str] = None, *args, **kwargs):
        return super()._save(output_dir, *args, **kwargs)

    @torch.no_grad()
    @abstractmethod
    def evaluate(self, eval_dataset: AbsDataset, *args, **kwargs):
        return super().evaluate(eval_dataset, **args, **kwargs)
