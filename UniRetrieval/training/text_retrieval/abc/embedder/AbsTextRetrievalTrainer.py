import logging
import torch
from typing import Optional
from abc import ABC, abstractmethod
from transformers.trainer import Trainer
from UniRetrieval.abc.training.trainer import AbsTrainer
from UniRetrieval.abc.training.dataset import AbsDataset
logger = logging.getLogger(__name__)


"""
此处未继承AbsTrainer，因为AbsTrainer的未implement的train和evaluate函数会覆盖Trainer的对应函数。
但保留了save_model这一abstractmethod，以保持和AbsTrainer的一致
"""
class AbsEmbedderTrainer(Trainer):
    # TODO 增加了save_model函数
    """
    Abstract class for the trainer of embedder.
    """
    @abstractmethod
    def save_model(self, output_dir: Optional[str] = None, state_dict=None):
        pass

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        
        Args:
            model (AbsEmbedderModel): The model being trained.
            inputs (dict): A dictionary of input tensors to be passed to the model.
            return_outputs (bool, optional): If ``True``, returns both the loss and the model's outputs. Otherwise,
                returns only the loss.
        
        Returns:
            Union[torch.Tensor, tuple(torch.Tensor, EmbedderOutput)]: The computed loss. If ``return_outputs`` is ``True``, 
                also returns the model's outputs in a tuple ``(loss, outputs)``.
        """

        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

