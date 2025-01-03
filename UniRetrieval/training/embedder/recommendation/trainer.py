import os
from UniRetrieval.abc.training.embedder import AbsEmbedderTrainer
import torch
from loguru import logger

class RetrieverTrainer(AbsEmbedderTrainer):
    def __init__(self, model, train=True, *args, **kwargs):
        super(RetrieverTrainer, self).__init__(model, *args, **kwargs)
        self.train_mode = train
        self.model_type = model.model_type
        # self._check_checkpoint_dir()
        if self.accelerator.is_main_process:
            print(model)
        
        self.item_vectors = None
        self.item_ids = None
        
    def compute_loss(self, model, batch, return_outputs=False,*args, **kwargs):
        outputs = model(batch=batch, cal_loss=True,*args, **kwargs)
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss
        
    def save_model(self, output_dir = None, **kwargs):
        """ Save the best model. """
        if self.accelerator.is_main_process:
            checkpoint_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(checkpoint_dir, exist_ok=True)
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save(checkpoint_dir)
            print(f"Model saved in {checkpoint_dir}.")
            
            checkpoint_dir = self.args.output_dir
            
            item_vectors_path = os.path.join(checkpoint_dir, 'item_vectors.pt')
            torch.save({'item_vectors': self.item_vectors, 'item_ids': self.item_ids}, item_vectors_path)
            logger.info(f'Item vectors saved.')
            
    
        