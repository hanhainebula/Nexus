import logging
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
class AbsRerankerTrainer(Trainer):
    """
    Abstract class for the trainer of reranker.
    """
    # TODO 增加save_model
    @abstractmethod
    def save_model(self, output_dir: Optional[str] = None, state_dict=None):
        pass

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        return self.save_model(output_dir, state_dict)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        
        Args:
            model (AbsRerankerModel): The model being trained.
            inputs (dict): A dictionary of input tensors to be passed to the model.
            return_outputs (bool, optional): If ``True``, returns both the loss and the model's outputs. Otherwise,
                returns only the loss. Defaults to ``False``.
        
        Returns:
            Union[torch.Tensor, tuple(torch.Tensor, RerankerOutput)]: The computed loss. If ``return_outputs`` is ``True``, 
                also returns the model's outputs in a tuple ``(loss, outputs)``.
        """

        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
