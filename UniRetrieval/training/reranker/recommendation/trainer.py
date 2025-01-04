import os
from loguru import logger

from UniRetrieval.abc.training.reranker import AbsRerankerTrainer

class RankerTrainer(AbsRerankerTrainer):
    def __init__(self, model, train=True, *args, **kwargs):
        super(RankerTrainer, self).__init__(model, *args, **kwargs)
        self.train_mode = train
        self.model_type = model.model_type
        if self.accelerator.is_main_process:
            print(model)
        
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
            logger.info(f"Model saved in {checkpoint_dir}.")
        

